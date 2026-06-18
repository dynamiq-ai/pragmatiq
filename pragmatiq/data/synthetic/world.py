"""The simulated world: calendar, merchants, transfer graph, campaigns, episodes.

``World.build(cfg)`` is phase A of generation: everything global and shared is
constructed here, deterministically from ``cfg.seed`` via namespaced
``np.random.default_rng((seed, stream))`` generators. Per-user simulation
(phase B) only reads from the world.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import WorldConfig
from .personas import PersonaTable, sample_personas

DAY_US = 86_400_000_000  # microseconds per day

# RNG stream namespaces. World streams are (seed, _NS_WORLD, k); per-user
# simulation streams are (seed, _NS_USER, user_idx). The distinct namespace
# field guarantees world build and user 1..5 never share a generator state.
_NS_WORLD = 1
_NS_USER = 0
_S_PERSONAS, _S_MERCHANTS, _S_GRAPH, _S_CAMPAIGNS, _S_EPISODES = 1, 2, 3, 4, 5


def user_rng(seed: int, user_idx: int) -> np.random.Generator:
    """Per-user generator, namespaced disjointly from world streams (rule 2)."""
    return np.random.default_rng((seed, _NS_USER, user_idx))


# --------------------------------------------------------------------------- calendar
@dataclass
class Calendar:
    """Day-resolution world calendar with paydays, holidays and drifts.

    All arrays are indexed by day offset from ``start`` (length ``n_days``).
    """

    start: np.datetime64  # day resolution
    n_days: int
    months: int
    month_start_day: np.ndarray  # int32[months+1], day offset of each month start
    day_of_week: np.ndarray  # int8[n_days], Monday=0
    is_weekend: np.ndarray  # bool[n_days]
    is_holiday: np.ndarray  # bool[n_days]
    season_mult: np.ndarray  # float64[n_days] spending seasonality
    inflation_mult: np.ndarray  # float64[n_days] mild price drift
    payday: np.ndarray  # int32[months], last business day of each month
    product_launch_month: dict[str, int] = field(default_factory=dict)

    HOUR_CURVE: np.ndarray = field(
        default_factory=lambda: _normalize24(
            np.array(
                # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                [0.3, 0.2, 0.15, 0.1, 0.1, 0.2, 0.5, 1.0, 1.6, 1.8, 1.9, 2.1, 2.6, 2.3, 1.9, 1.8, 2.0, 2.4, 2.6, 2.3, 1.8, 1.4, 0.9, 0.5]
            )
        )
    )
    DOW_MULT: np.ndarray = field(
        default_factory=lambda: np.array([0.95, 0.93, 0.95, 1.0, 1.15, 1.25, 1.05])
    )

    @classmethod
    def build(cls, cfg: WorldConfig) -> Calendar:
        """Construct the calendar for ``cfg.months`` starting at ``cfg.start_date``."""
        start_month = np.datetime64(cfg.start_date, "M")
        start = start_month.astype("datetime64[D]")
        month_starts = (start_month + np.arange(cfg.months + 1)).astype("datetime64[D]")
        month_start_day = (month_starts - start).astype(np.int32)
        n_days = int(month_start_day[-1])

        days = start + np.arange(n_days)
        # numpy: 1970-01-01 was a Thursday; ((days - epoch) + 3) % 7 gives Monday=0.
        day_of_week = ((days.astype("datetime64[D]").view("int64") + 3) % 7).astype(np.int8)
        is_weekend = day_of_week >= 5

        is_holiday = np.zeros(n_days, dtype=bool)
        md = np.stack(
            [(days.astype("datetime64[M]").view("int64") % 12) + 1, (days - days.astype("datetime64[M]")).view("int64") + 1],
            axis=1,
        )
        for mm, dd in ((1, 1), (4, 7), (5, 1), (8, 25), (12, 24), (12, 25), (12, 26), (12, 31)):
            is_holiday |= (md[:, 0] == mm) & (md[:, 1] == dd)

        month_of_year = md[:, 0]
        season_by_month = np.array([0.88, 0.94, 0.98, 1.0, 1.02, 1.04, 1.06, 1.05, 0.99, 1.0, 1.08, 1.28])
        season_mult = season_by_month[month_of_year - 1].astype(np.float64)

        day_idx = np.arange(n_days, dtype=np.float64)
        inflation_mult = (1.0 + 0.032) ** (day_idx / 365.25)  # ~3.2%/yr drift

        payday = np.zeros(cfg.months, dtype=np.int32)
        business = ~(is_weekend | is_holiday)
        for m in range(cfg.months):
            lo, hi = int(month_start_day[m]), int(month_start_day[m + 1])
            bdays = np.nonzero(business[lo:hi])[0]
            payday[m] = lo + (int(bdays[-1]) if len(bdays) else hi - lo - 1)

        launches = {"trading": min(4, cfg.months - 2), "premium": min(8, cfg.months - 1), "crypto": min(12, cfg.months - 1)}
        return cls(
            start=start, n_days=n_days, months=cfg.months, month_start_day=month_start_day,
            day_of_week=day_of_week, is_weekend=is_weekend, is_holiday=is_holiday,
            season_mult=season_mult, inflation_mult=inflation_mult, payday=payday,
            product_launch_month=launches,
        )

    def start_us(self) -> int:
        """World start as µs since epoch."""
        return int(self.start.astype("datetime64[us]").view("int64"))

    def day_to_us(self, day: np.ndarray | int) -> np.ndarray | int:
        """Convert day offsets to µs since epoch."""
        return self.start_us() + np.asarray(day, dtype=np.int64) * DAY_US if isinstance(day, np.ndarray) else self.start_us() + int(day) * DAY_US

    def month_of_day(self, day: np.ndarray) -> np.ndarray:
        """Month index for each day offset."""
        return np.searchsorted(self.month_start_day, np.asarray(day), side="right") - 1


def _normalize24(curve: np.ndarray) -> np.ndarray:
    return curve / curve.mean()


# --------------------------------------------------------------------------- merchants
# (mcc_code, key, description, lognormal mu, sigma, online_prob, weight)
MCC_CATALOG: list[tuple[str, str, str, float, float, float, float]] = [
    ("5411", "grocery", "Grocery Stores", 3.30, 0.55, 0.08, 0.190),
    ("5812", "restaurant", "Restaurants", 3.35, 0.65, 0.12, 0.110),
    ("5814", "fast_food", "Fast Food", 2.30, 0.45, 0.30, 0.110),
    ("5541", "fuel", "Service Stations", 3.85, 0.35, 0.02, 0.060),
    ("4111", "transport", "Local Transport", 1.45, 0.45, 0.55, 0.090),
    ("4121", "taxi", "Taxis/Rideshare", 2.55, 0.50, 0.95, 0.040),
    ("4511", "airline", "Airlines", 5.15, 0.65, 0.92, 0.012),
    ("7011", "hotel", "Hotels", 4.95, 0.60, 0.80, 0.014),
    ("5969", "online_retail", "Online Marketplaces", 3.45, 0.85, 1.00, 0.090),
    ("5732", "electronics", "Electronics", 4.55, 0.90, 0.65, 0.030),
    ("5651", "clothing", "Clothing", 3.85, 0.70, 0.45, 0.055),
    ("5912", "pharmacy", "Pharmacies", 2.60, 0.55, 0.10, 0.045),
    ("4814", "telecom", "Telecom", 3.30, 0.30, 0.90, 0.030),
    ("4900", "utilities", "Utilities", 4.45, 0.35, 0.95, 0.030),
    ("5815", "streaming", "Digital Media", 2.35, 0.25, 1.00, 0.045),
    ("5816", "gaming", "Digital Games", 2.65, 0.60, 1.00, 0.022),
    ("7995", "gambling", "Betting/Casino", 3.20, 0.95, 0.97, 0.022),
    ("6011", "atm", "ATM Withdrawal", 3.95, 0.40, 0.00, 0.040),
    ("5944", "jewelry", "Jewelry", 5.05, 0.85, 0.40, 0.005),
    ("7997", "gym", "Gyms/Clubs", 3.45, 0.30, 0.70, 0.020),
    ("5942", "books", "Book Stores", 2.70, 0.50, 0.60, 0.015),
    ("8011", "health", "Medical Services", 4.10, 0.60, 0.25, 0.025),
]
MCC_KEYS: tuple[str, ...] = tuple(row[1] for row in MCC_CATALOG)
MCC_IDX: dict[str, int] = {k: i for i, k in enumerate(MCC_KEYS)}

_BRAND_STEMS: dict[str, list[str]] = {
    "grocery": ["TESCO STORES", "SAINSBURYS", "ASDA SUPERSTORE", "LIDL GB", "ALDI", "WAITROSE", "MORRISONS", "CO-OP GROUP", "CARREFOUR", "BIEDRONKA", "EDEKA", "MERCADONA", "MAXIMA", "SPAR"],
    "restaurant": ["NANDOS", "PIZZA EXPRESS", "WAGAMAMA", "BELLA ITALIA", "THE IVY", "DISHOOM", "FRANCO MANCA", "BISTRO", "TRATTORIA", "BRASSERIE"],
    "fast_food": ["MCDONALDS", "KFC", "GREGGS", "SUBWAY", "BURGER KING", "PRET A MANGER", "DOMINOS PIZZA", "FIVE GUYS", "TACO BELL", "COSTA COFFEE", "STARBUCKS"],
    "fuel": ["SHELL", "BP CONNECT", "ESSO", "TEXACO", "ORLEN", "TOTAL ENERGIES", "CIRCLE K"],
    "transport": ["TFL TRAVEL", "TRAINLINE", "NATIONAL RAIL", "STAGECOACH", "MEGABUS", "SNCF", "DEUTSCHE BAHN", "RYANAIR RAIL"],
    "taxi": ["UBER", "BOLT", "FREENOW", "GETT", "LYFT"],
    "airline": ["RYANAIR", "EASYJET", "BRITISH AIRWAYS", "WIZZ AIR", "LUFTHANSA", "KLM", "VUELING"],
    "hotel": ["PREMIER INN", "TRAVELODGE", "HILTON", "IBIS", "BOOKING.COM", "AIRBNB", "MARRIOTT"],
    "online_retail": ["AMAZON MKTPLACE", "EBAY", "ETSY", "ALIEXPRESS", "ASOS.COM", "VINTED", "ZALANDO", "TEMU.COM", "SHEIN.COM"],
    "electronics": ["CURRYS", "APPLE STORE", "SAMSUNG SHOP", "MEDIA MARKT", "ARGOS", "RICHER SOUNDS", "CEX LTD"],
    "clothing": ["PRIMARK", "ZARA", "H&M", "NEXT RETAIL", "UNIQLO", "MARKS&SPENCER", "TK MAXX", "DECATHLON"],
    "pharmacy": ["BOOTS", "SUPERDRUG", "LLOYDS PHARMACY", "ROSSMANN", "DM DROGERIE", "APTEKA"],
    "telecom": ["VODAFONE", "O2 UK", "EE LIMITED", "THREE.CO.UK", "GIFFGAFF", "ORANGE", "T-MOBILE"],
    "utilities": ["BRITISH GAS", "EDF ENERGY", "OCTOPUS ENERGY", "THAMES WATER", "E.ON NEXT", "SSE ENERGY"],
    "streaming": ["SPOTIFY", "NETFLIX.COM", "DISNEY PLUS", "YOUTUBE PREMIUM", "APPLE.COM/BILL", "AUDIBLE UK", "HBO MAX", "NOW TV"],
    "gaming": ["STEAM PURCHASE", "PLAYSTATION NETWORK", "XBOX LIVE", "NINTENDO", "EPIC GAMES", "RIOT GAMES", "ROBLOX"],
    "gambling": ["BET365", "SKYBET", "PADDY POWER", "WILLIAM HILL", "LADBROKES", "POKERSTARS", "NATIONAL LOTTERY", "BETFAIR"],
    "atm": ["LINK ATM", "CASHZONE ATM", "NOTEMACHINE", "EURONET ATM", "BANKOMAT"],
    "jewelry": ["PANDORA", "GOLDSMITHS", "H SAMUEL", "SWAROVSKI", "TIFFANY & CO"],
    "gym": ["PUREGYM", "THE GYM GROUP", "DAVID LLOYD", "ANYTIME FITNESS", "NUFFIELD HEALTH"],
    "books": ["WATERSTONES", "WH SMITH", "BLACKWELLS", "FOYLES", "AMAZON KINDLE"],
    "health": ["BUPA", "SPECSAVERS", "VISION EXPRESS", "MYDENTIST", "PRIVATE CLINIC"],
}
_AGG_PREFIX = ["PAYPAL *", "SQ *", "SUMUP *", "IZ *", "ZTL*", "GOCARDLESS "]
_TOWNS = ["LONDON", "MANCHESTER", "LEEDS", "BRISTOL", "DUBLIN", "PARIS", "BERLIN", "MADRID", "WARSAW", "VILNIUS", "GLASGOW", "CARDIFF", "LYON", "MUNICH", "SEVILLE", "KRAKOW", "KAUNAS", "BIRMINGHAM", "LIVERPOOL", "EDINBURGH"]
_NOUNS = ["CORNER SHOP", "MINIMART", "FOODS", "TRADING", "SERVICES", "RETAIL", "STORES", "GROUP", "DIRECT", "EXPRESS", "SUPPLY", "STUDIO", "WORKSHOP", "MARKET", "DEPOT"]


@dataclass
class MerchantUniverse:
    """All merchants: MCC, Zipf popularity, noisy display names, countries.

    ``sample_in_mcc`` draws merchant ids within an MCC according to within-MCC
    Zipf popularity — the workhorse used by the spending process.
    """

    n_merchants: int
    mcc_idx: np.ndarray  # int16[n_merchants]
    names: np.ndarray  # object[n_merchants]
    countries: np.ndarray  # object[n_merchants]
    by_mcc: list[np.ndarray]  # merchant ids per MCC, popularity-sorted
    by_mcc_cum: list[np.ndarray]  # cumulative Zipf probs aligned with by_mcc
    mcc_mu: np.ndarray  # float64[n_mcc] lognormal amount mu
    mcc_sigma: np.ndarray
    mcc_online_p: np.ndarray

    @classmethod
    def build(cls, cfg: WorldConfig, rng: np.random.Generator) -> MerchantUniverse:
        """Sample the merchant universe (names/MCCs/popularity) from ``rng``."""
        n = cfg.n_merchants
        weights = np.array([row[6] for row in MCC_CATALOG])
        if cfg.mcc_weights:
            unknown = set(cfg.mcc_weights) - set(MCC_KEYS)
            if unknown:
                raise ValueError(f"mcc_weights has unknown MCC keys: {sorted(unknown)}")
            for k, v in cfg.mcc_weights.items():
                weights[MCC_IDX[k]] = float(v)
        weights = weights / weights.sum()
        n_mcc = len(MCC_CATALOG)
        if n < n_mcc:
            raise ValueError(f"n_merchants ({n}) must be >= number of MCCs ({n_mcc})")
        mcc_idx = rng.choice(n_mcc, size=n, p=weights).astype(np.int16)
        # Guarantee every MCC has >= 1 merchant so sample_in_mcc always has a
        # candidate: reassign the first slots to any missing MCC.
        missing = np.setdiff1d(np.arange(n_mcc), np.unique(mcc_idx))
        if len(missing):
            mcc_idx[: len(missing)] = missing.astype(np.int16)

        countries = np.array(list(cfg.country_mix), dtype=object)
        c_probs = np.array(list(cfg.country_mix.values()))
        merchant_country = countries[rng.choice(len(countries), size=n, p=c_probs / c_probs.sum())]

        names = np.empty(n, dtype=object)
        by_mcc: list[np.ndarray] = []
        by_mcc_cum: list[np.ndarray] = []
        for mi, key in enumerate(MCC_KEYS):
            ids = np.nonzero(mcc_idx == mi)[0]
            rng.shuffle(ids)
            n_mcc = len(ids)
            stems = _BRAND_STEMS[key]
            n_brand = min(n_mcc, max(1, int(n_mcc * 0.35)))  # top of the Zipf are brands
            stem_pick = rng.integers(0, len(stems), size=n_brand)
            store_no = rng.integers(1, 9800, size=n_brand)
            for j in range(n_brand):
                nm = f"{stems[stem_pick[j]]} {store_no[j]}"
                names[ids[j]] = nm
            town_pick = rng.integers(0, len(_TOWNS), size=n_mcc - n_brand)
            noun_pick = rng.integers(0, len(_NOUNS), size=n_mcc - n_brand)
            for j in range(n_mcc - n_brand):
                names[ids[n_brand + j]] = f"{_TOWNS[town_pick[j]]} {_NOUNS[noun_pick[j]]}"
            # Aggregator prefixes and statement-style truncation noise.
            agg_mask = rng.random(n_mcc) < 0.12
            trunc_mask = rng.random(n_mcc) < 0.15
            agg_pick = rng.integers(0, len(_AGG_PREFIX), size=n_mcc)
            trunc_len = rng.integers(11, 22, size=n_mcc)
            for j in range(n_mcc):
                mid = ids[j]
                if agg_mask[j]:
                    names[mid] = _AGG_PREFIX[agg_pick[j]] + str(names[mid])
                if trunc_mask[j]:
                    names[mid] = str(names[mid])[: trunc_len[j]]
            # Within-MCC Zipf popularity over the (already shuffled) ids.
            ranks = np.arange(1, n_mcc + 1, dtype=np.float64)
            zipf = 1.0 / ranks**1.07
            zipf /= zipf.sum()
            by_mcc.append(ids.astype(np.int32))
            by_mcc_cum.append(np.cumsum(zipf))

        mcc_mu = np.array([r[3] for r in MCC_CATALOG])
        mcc_sigma = np.array([r[4] for r in MCC_CATALOG])
        if cfg.mcc_amount_mean:
            unknown = set(cfg.mcc_amount_mean) - set(MCC_KEYS)
            if unknown:
                raise ValueError(f"mcc_amount_mean has unknown MCC keys: {sorted(unknown)}")
            for k, mean in cfg.mcc_amount_mean.items():
                i = MCC_IDX[k]
                # Lognormal moment matching: E[X]=exp(mu+sigma^2/2), sigma kept.
                mcc_mu[i] = float(np.log(max(float(mean), 0.01)) - 0.5 * mcc_sigma[i] ** 2)
        return cls(
            n_merchants=n, mcc_idx=mcc_idx, names=names, countries=merchant_country,
            by_mcc=by_mcc, by_mcc_cum=by_mcc_cum,
            mcc_mu=mcc_mu,
            mcc_sigma=mcc_sigma,
            mcc_online_p=np.array([r[5] for r in MCC_CATALOG]),
        )

    def sample_in_mcc(self, mcc: int, u: np.ndarray) -> np.ndarray:
        """Map uniforms ``u`` to merchant ids within MCC ``mcc`` (Zipf-weighted)."""
        pos = np.searchsorted(self.by_mcc_cum[mcc], u, side="right")
        pos = np.minimum(pos, len(self.by_mcc[mcc]) - 1)
        return self.by_mcc[mcc][pos]


# --------------------------------------------------------------------------- transfer graph
@dataclass
class MuleRing:
    """One injected mule ring as a multi-hop layering chain.

    Members are arranged into ``depth`` ordered layers. Funds fan in from
    ordinary senders onto the first (collector) layer, are forwarded hop by hop
    through the intermediate layers (the *layering* that hides the trail), and
    cash out from the last (distributor) layer. ``layer_of_member`` records each
    member's 0-based layer; the discriminative structure therefore spans the full
    chain depth, so it is recoverable by ≥2-hop message passing but only weakly
    by any single node's 1-hop degree.
    """

    ring_id: int
    members: np.ndarray  # user idx of the mules (labeled aml=1)
    senders: np.ndarray  # ordinary users who send the small credits
    layer_of_member: np.ndarray  # int, 0..depth-1 position of each member in the chain
    window_start_day: int
    window_len_days: int


@dataclass
class TransferGraph:
    """Watts–Strogatz social graph with pre-scheduled P2P transfers and mule rings.

    Transfers are fully scheduled in phase A (so ``transfers.parquet`` is a pure
    function of the seed); the per-user simulator emits matching events.
    """

    from_idx: np.ndarray  # int32[n_transfers], sorted by ts
    to_idx: np.ndarray
    ts_us: np.ndarray  # int64
    amount: np.ndarray  # float64 (2dp)
    is_mule_leg: np.ndarray  # int8: 0 normal, 1 fan-in, 2 fan-out
    rings: list[MuleRing]
    user_out_start: np.ndarray  # CSR over transfers sorted by from_idx
    user_out_order: np.ndarray
    user_in_start: np.ndarray  # CSR over transfers sorted by to_idx
    user_in_order: np.ndarray

    @classmethod
    def build(
        cls, cfg: WorldConfig, cal: Calendar, personas: PersonaTable, rng: np.random.Generator
    ) -> TransferGraph:
        """Build the organic graph, schedule edge transfers, inject mule rings."""
        n = cfg.n_users
        # Heavy-tailed organic degree via a sociability-driven configuration
        # model. A broad degree distribution lets mules blend in: a constant
        # degree (every organic user with ~4 distinct counterparties) would make
        # ring fan-in a near-oracle structural feature, so organic degree is
        # spread wide enough that ring fan-in is not distinctive on its own.
        soc_all = personas.traits["sociability"]
        # negative binomial (r=0.5): same mean as a Poisson at this rate but a
        # geometric-ish tail, so organic p99 degree (~20) overlaps ring fan-in.
        # A Poisson tail (p99 ~8) would instead leave mules structurally isolated.
        mean_k = cfg.organic_degree_dispersion * soc_all**2
        k_i = 2 + rng.negative_binomial(0.5, 0.5 / (0.5 + mean_k))
        k_i = np.clip(k_i, 1, 48).astype(np.int64)
        stubs = np.repeat(np.arange(n, dtype=np.int64), k_i)
        rng.shuffle(stubs)
        m_edges = len(stubs) // 2
        src, dst = stubs[:m_edges].copy(), stubs[m_edges:2 * m_edges].copy()
        keep = src != dst
        src, dst = src[keep], dst[keep]
        # drop parallel pairs so per-edge transfer counts stay meaningful
        first = np.unique(src * n + dst, return_index=True)[1]
        first.sort()
        src, dst = src[first], dst[first]

        # Edge transfer counts driven by endpoint sociability and overlap of
        # active windows; times uniform in the overlap.
        soc = personas.traits["sociability"]
        signup = personas.signup_day
        horizon = cal.n_days
        start_d = np.maximum(signup[src], signup[dst]) + 3
        overlap_days = np.maximum(horizon - start_d - 1, 0)
        rate = 0.55 * (soc[src] + soc[dst])  # transfers per 30d on the edge
        lam = rate * overlap_days / 30.0
        counts = rng.poisson(lam)
        total = int(counts.sum())
        e_src = np.repeat(src, counts).astype(np.int32)
        e_dst = np.repeat(dst, counts).astype(np.int32)
        e_lo = np.repeat(start_d, counts).astype(np.int64)
        e_span = np.repeat(overlap_days, counts).astype(np.int64)
        day = e_lo + (rng.random(total) * e_span).astype(np.int64)
        tod = (rng.beta(3.2, 2.2, size=total) * 86_400_000_000).astype(np.int64)  # day-tilted
        ts = cal.start_us() + day * DAY_US + tod
        flip = rng.random(total) < 0.5  # direction per transfer
        f, t = np.where(flip, e_dst, e_src), np.where(flip, e_src, e_dst)
        amt = np.round(np.clip(rng.lognormal(3.4, 0.9, size=total), 1.0, 2500.0), 2)
        leg = np.zeros(total, dtype=np.int8)

        rings = _inject_mule_rings(cfg, cal, personas, rng)
        ring_f, ring_t, ring_ts, ring_amt, ring_leg = _schedule_ring_transfers(
            rings, cal, rng, n, personas.signup_day
        )
        f = np.concatenate([f, ring_f])
        t = np.concatenate([t, ring_t])
        ts = np.concatenate([ts, ring_ts])
        amt = np.concatenate([amt, ring_amt])
        leg = np.concatenate([leg, ring_leg])

        order = np.argsort(ts, kind="stable")
        f, t, ts, amt, leg = f[order], t[order], ts[order], amt[order], leg[order]

        out_order = np.argsort(f, kind="stable").astype(np.int64)
        out_start = np.searchsorted(f[out_order], np.arange(n + 1))
        in_order = np.argsort(t, kind="stable").astype(np.int64)
        in_start = np.searchsorted(t[in_order], np.arange(n + 1))
        return cls(
            from_idx=f.astype(np.int32), to_idx=t.astype(np.int32), ts_us=ts.astype(np.int64),
            amount=amt, is_mule_leg=leg, rings=rings,
            user_out_start=out_start.astype(np.int64), user_out_order=out_order,
            user_in_start=in_start.astype(np.int64), user_in_order=in_order,
        )

    def user_outgoing(self, user_idx: int) -> np.ndarray:
        """Indices into the transfer arrays where ``user_idx`` is the sender."""
        s, e = self.user_out_start[user_idx], self.user_out_start[user_idx + 1]
        return self.user_out_order[s:e]

    def user_incoming(self, user_idx: int) -> np.ndarray:
        """Indices into the transfer arrays where ``user_idx`` is the receiver."""
        s, e = self.user_in_start[user_idx], self.user_in_start[user_idx + 1]
        return self.user_in_order[s:e]


def _inject_mule_rings(
    cfg: WorldConfig, cal: Calendar, personas: PersonaTable, rng: np.random.Generator
) -> list[MuleRing]:
    rings: list[MuleRing] = []
    n = cfg.n_users
    signup = personas.signup_day
    used: set[int] = set()  # no user belongs to two rings
    # Each ring needs up to 8 members + senders; clamp (loudly) for tiny smoke
    # worlds rather than draft the same users into several rings.
    n_rings = min(cfg.mule_ring_count, n // 12)
    if n_rings < cfg.mule_ring_count:
        logging.getLogger(__name__).warning(
            "mule_ring_count=%d too large for n_users=%d; building %d rings",
            cfg.mule_ring_count, n, n_rings)
    for r in range(n_rings):
        # A ring is a chain of ``depth`` layers (collectors → intermediaries →
        # distributors). Each layer holds 1-3 members; deeper chains make the
        # discriminative motif span more hops. The signal is the multi-hop
        # layering path, not any node's local degree.
        depth = int(rng.integers(3, 6))  # 3-5 layers
        per_layer = rng.integers(1, 4, size=depth)  # 1-3 members per layer
        size = int(per_layer.sum())
        # Draft ring members from the general population, keeping their original
        # personas. Real money mules are ordinary recruited accounts, so a mule's
        # own behaviour is statistically indistinguishable from a normal user's
        # and the only signal is ring membership in the transfer graph. A distinct
        # "mule" persona would expose AML membership as a single-field tell an
        # isolated embedding could read directly.
        taken = np.fromiter(used, dtype=np.int64, count=len(used)) if used else np.zeros(0, np.int64)
        candidates = np.setdiff1d(np.arange(n, dtype=np.int64), taken)
        if len(candidates) < size:
            raise ValueError(
                f"cannot fill mule ring {r}: only {len(candidates)} un-drafted users remain "
                f"(n_users={n}, mule_ring_count={cfg.mule_ring_count}); lower mule_ring_count")
        members = rng.choice(candidates, size=size, replace=False).astype(np.int64)
        layer_of_member = np.repeat(np.arange(depth), per_layer).astype(np.int64)
        used.update(int(u) for u in members)
        # Window must start after every member is onboarded — ring activity can
        # never predate an account_opened milestone.
        member_ready = int(signup[members].max()) + 5
        lo = max(int(cal.month_start_day[min(2, cfg.months - 2)]), member_ready)
        hi = cal.n_days - 12
        w0 = int(rng.integers(lo, max(lo + 1, hi)))
        # Long, dispersed laundering window (1-4 months; individual legs are
        # clamped to the book horizon when scheduled): the fan-in / layering /
        # fan-out legs spread out so a mule's OWN event stream is ordinary P2P
        # churn (no tight in-then-out burst) — the ring is visible in the transfer
        # GRAPH, not in any single member's behaviour.
        win_len = int(rng.integers(30, max(31, min(120, cal.n_days - w0 - 2))))
        # Senders must already be active when the fan-in happens. Fan-in is sized
        # so each collector's in-degree from the ring (≈ n_senders / collectors)
        # matches an ordinary sociable user's organic counterparty count — the
        # ring is NOT detectable from raw in-degree. Detection must come from the
        # multi-hop chain plus the faint per-member behavioural fingerprint, not a
        # trivially high degree.
        n_collectors = int(per_layer[0])
        eligible = np.nonzero(signup < w0)[0]
        eligible = eligible[~np.isin(eligible, members)]
        # Few distinct senders per collector (2-3), each sending several credits
        # over the window. This keeps a collector's distinct-counterparty count
        # in the ordinary range, so even a hand-crafted distinct-in feature is
        # only weakly predictive — the ring is not a high-fan-in oracle.
        n_senders = min(int(rng.integers(2, 4)) * n_collectors, len(eligible))
        senders = rng.choice(eligible, size=n_senders, replace=False) if n_senders > 0 else np.zeros(0, np.int64)
        rings.append(
            MuleRing(ring_id=r, members=members, senders=senders.astype(np.int64),
                     layer_of_member=layer_of_member,
                     window_start_day=w0, window_len_days=win_len)
        )
    return rings


def _schedule_ring_transfers(
    rings: list[MuleRing], cal: Calendar, rng: np.random.Generator, n_users: int,
    signup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_l: list[np.ndarray] = []
    t_l: list[np.ndarray] = []
    ts_l: list[np.ndarray] = []
    a_l: list[np.ndarray] = []
    leg_l: list[np.ndarray] = []

    def organic_amt(k: int) -> np.ndarray:
        # Laundering legs draw amounts from the SAME lognormal as organic P2P
        # (simulator ``_p2p`` uses lognormal(3.4, 0.9) clipped to [1, 2500]). So
        # the per-node amount/volume statistics of a mule are indistinguishable
        # from an ordinary sociable user's — hand-crafted volume features carry
        # no ring signal. The ring is encoded in the multi-hop *topology* and the
        # faint behavioural fingerprint, not in transfer magnitudes.
        return np.round(np.clip(rng.lognormal(3.4, 0.9, size=k), 1.0, 2500.0), 2)

    for ring in rings:
        if len(ring.members) == 0 or len(ring.senders) == 0:
            continue
        members = ring.members
        layer = ring.layer_of_member
        depth = int(layer.max()) + 1
        members_by_layer = [members[layer == d] for d in range(depth)]
        w0_us = cal.start_us() + ring.window_start_day * DAY_US
        span_us = ring.window_len_days * DAY_US

        # Fan-in: ordinary senders credit the COLLECTOR layer only (layer 0). Few
        # senders, each sending 1-2 credits, so both a collector's in-degree and
        # its distinct-counterparty count stay in the ordinary range — raw degree
        # is not a ring oracle.
        collectors = members_by_layer[0]
        n_in = rng.integers(1, 3, size=len(ring.senders))
        tot_in = int(n_in.sum())
        s_idx = np.repeat(ring.senders, n_in)
        m_idx = collectors[rng.integers(0, len(collectors), size=tot_in)]
        ts_in = w0_us + (rng.random(tot_in) * span_us).astype(np.int64)  # over the whole (long) window
        amt_in = organic_amt(tot_in)
        f_l.append(s_idx)
        t_l.append(m_idx)
        ts_l.append(ts_in)
        a_l.append(amt_in)
        leg_l.append(np.ones(tot_in, dtype=np.int8))

        # Layering chain: each layer forwards to the NEXT layer (hop by hop). This
        # directed multi-hop path is the discriminative motif — a collector's
        # 1-hop neighbourhood looks like ordinary P2P, but the length-``depth``
        # source→…→sink chain is recoverable only by ≥2-hop message passing over
        # informative node features. Each forwarding member sends to 1-2 members
        # of the next layer, so no node accrues anomalous out-degree.
        for d in range(depth - 1):
            src_layer = members_by_layer[d]
            dst_layer = members_by_layer[d + 1]
            if len(src_layer) == 0 or len(dst_layer) == 0:
                continue
            fan = rng.integers(1, 3, size=len(src_layer))  # 1-2 forwards each
            src_m = np.repeat(src_layer, fan)
            dst_m = dst_layer[rng.integers(0, len(dst_layer), size=int(fan.sum()))]
            k = len(src_m)
            # Each forward happens after the funds have had time to arrive (the
            # window is long and dispersed), so the chain never time-travels.
            ts_layer = w0_us + (0.2 + 0.6 * rng.random(k)) * span_us
            f_l.append(src_m.astype(np.int64))
            t_l.append(dst_m.astype(np.int64))
            ts_l.append(ts_layer.astype(np.int64))
            a_l.append(organic_amt(k))
            leg_l.append(np.full(k, 3, dtype=np.int8))  # 3 = layering leg

        # Cash-out: the DISTRIBUTOR layer (last) forwards to ordinary onboarded
        # accounts. Each distributor sends 1-2 legs of organic-sized amounts —
        # degree and volume match an ordinary user, so the cash-out is invisible
        # to hand-crafted node statistics.
        distributors = members_by_layer[-1]
        fwd_day = ring.window_start_day + ring.window_len_days
        ready = np.nonzero(signup <= fwd_day)[0]
        ready = ready[~np.isin(ready, members)]
        for m in distributors:
            n_out = int(rng.integers(1, 3))
            if len(ready):
                dests = rng.choice(ready, size=n_out)
            else:
                pool = members[members != m]
                dests = rng.choice(pool if len(pool) else members, size=n_out)
            ts_out = w0_us + (0.6 + 0.4 * rng.random(n_out)) * span_us
            f_l.append(np.full(n_out, m, dtype=np.int64))
            t_l.append(dests.astype(np.int64))
            ts_l.append(ts_out.astype(np.int64))
            a_l.append(organic_amt(n_out))
            leg_l.append(np.full(n_out, 2, dtype=np.int8))
    if not f_l:
        z = np.zeros(0)
        return z.astype(np.int64), z.astype(np.int64), z.astype(np.int64), z, z.astype(np.int8)
    # Clamp every ring leg to the book horizon: a long window or a fan-out delay
    # can schedule a transfer past the last simulated day, and (unlike user
    # events, which simulation time bounds) these go straight to transfers.parquet
    # — so no laundering leg may fall after the simulation end.
    horizon_us = cal.start_us() + cal.n_days * DAY_US - 1
    return (
        np.concatenate(f_l).astype(np.int64),
        np.concatenate(t_l).astype(np.int64),
        np.minimum(np.concatenate(ts_l).astype(np.int64), horizon_us),
        np.concatenate(a_l).astype(np.float64),
        np.concatenate(leg_l),
    )


# --------------------------------------------------------------------------- campaigns
@dataclass
class CampaignCalendar:
    """Marketing campaigns; each user gets BOTH potential outcomes for uplift."""

    campaign_id: np.ndarray  # object[str]
    ts_us: np.ndarray  # int64, send moment
    target_frac: np.ndarray  # float64
    channel: np.ndarray  # object: push | email
    template: np.ndarray  # object
    uplift: np.ndarray  # float64, max uplift on the 48h-app-open outcome

    @classmethod
    def build(cls, cfg: WorldConfig, cal: Calendar, rng: np.random.Generator) -> CampaignCalendar:
        """Schedule ~``campaigns_per_month × months`` campaigns on business days."""
        n_c = max(1, int(round(cfg.campaigns_per_month * cfg.months)))
        days = rng.integers(10, cal.n_days - 7, size=n_c)
        hours = rng.integers(9, 19, size=n_c)
        ts = cal.start_us() + days.astype(np.int64) * DAY_US + hours.astype(np.int64) * 3_600_000_000
        order = np.argsort(ts)
        templates = np.array(
            ["cashback_offer", "premium_trial", "trading_promo", "referral_bonus", "savings_nudge", "travel_fx"],
            dtype=object,
        )
        return cls(
            campaign_id=np.array([f"camp_{i:04d}" for i in range(n_c)], dtype=object),
            ts_us=ts[order],
            target_frac=rng.uniform(0.3, 0.8, size=n_c),
            channel=np.where(rng.random(n_c) < 0.65, "push", "email").astype(object),
            template=templates[rng.integers(0, len(templates), size=n_c)],
            uplift=rng.uniform(0.03, 0.14, size=n_c),
        )


# --------------------------------------------------------------------------- episodes (assignment)
@dataclass
class EpisodeAssignment:
    """Phase-A decisions about who gets fraud/stress/mule episodes and when."""

    fraud_user: np.ndarray  # bool[n_users]
    fraud_start_day: np.ndarray  # int32 (valid where fraud_user)
    fraud_len_days: np.ndarray  # int8 1..3
    stress_user: np.ndarray  # bool[n_users]
    stress_start_month: np.ndarray  # int16
    stress_len_months: np.ndarray  # int8 2..6
    stress_severity: np.ndarray  # float64 in [0.35, 1]
    mule_member: np.ndarray  # bool[n_users] (aml ground truth)
    mule_window: np.ndarray  # int32[n_users, 2] day window (start, end), -1 if none
    mule_layer: np.ndarray  # int8[n_users] chain layer of a mule (-1 if none)
    mule_depth: np.ndarray  # int8[n_users] chain depth of the mule's ring (0 if none)

    @classmethod
    def build(
        cls, cfg: WorldConfig, cal: Calendar, personas: PersonaTable,
        rings: list[MuleRing], rng: np.random.Generator,
    ) -> EpisodeAssignment:
        """Sample which users get which episode, weighted by latent traits."""
        n = cfg.n_users
        w = cfg.trait_noise

        # Fraud: probability proportional to fraud_vulnerability (noise-blended).
        vul = personas.traits["fraud_vulnerability"]
        p = (1 - w) * vul + w * vul.mean()
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1.0 / n)
        n_fraud = min(n, max(1, int(round(cfg.fraud_rate * n))))
        fraud_ids = rng.choice(n, size=n_fraud, replace=False, p=p)
        fraud_user = np.zeros(n, dtype=bool)
        fraud_user[fraud_ids] = True
        fraud_start = np.zeros(n, dtype=np.int32)
        lo = personas.signup_day[fraud_ids] + 45
        hi = cal.n_days - 5
        fraud_start[fraud_ids] = (lo + rng.random(n_fraud) * np.maximum(hi - lo, 1)).astype(np.int32)
        fraud_len = np.zeros(n, dtype=np.int8)
        fraud_len[fraud_ids] = rng.integers(1, 4, size=n_fraud).astype(np.int8)

        # Stress arcs: rate × insolvency conversion (~0.25) lands near cfg.default_rate;
        # non-converting arcs are realistic hard negatives (recovered stress).
        stress = personas.traits["financial_stress"]
        arc_rate = min(0.85, cfg.default_rate * 4.0)
        n_arc = max(1, int(round(arc_rate * n)))
        ps = (1 - w) * stress**1.5 + w * np.full(n, (stress**1.5).mean())
        ps = ps / ps.sum()
        arc_ids = rng.choice(n, size=n_arc, replace=False, p=ps)
        stress_user = np.zeros(n, dtype=bool)
        stress_user[arc_ids] = True
        # A recession wave concentrates arc onsets around the credit eval point
        # (some arcs then show pre-eval early-warning behavior, others start
        # after eval and stay irreducibly unpredictable — both are needed for
        # a realistic AUC ceiling).
        stress_start = np.zeros(n, dtype=np.int16)
        wave_lo = max(3, cfg.eval_month_credit - 5)
        wave_hi = max(wave_lo + 1, cfg.eval_month_credit)
        in_wave = rng.random(n_arc) < 0.55
        starts = np.where(
            in_wave,
            rng.integers(wave_lo, wave_hi, size=n_arc),
            rng.integers(3, max(4, cfg.months - 5), size=n_arc),
        )
        stress_start[arc_ids] = starts.astype(np.int16)
        stress_len = np.zeros(n, dtype=np.int8)
        stress_len[arc_ids] = rng.integers(2, 7, size=n_arc).astype(np.int8)
        stress_sev = np.zeros(n, dtype=np.float64)
        stress_sev[arc_ids] = rng.uniform(0.55, 1.0, size=n_arc)

        mule_member = np.zeros(n, dtype=bool)
        mule_window = np.full((n, 2), -1, dtype=np.int32)
        mule_layer = np.full(n, -1, dtype=np.int8)
        mule_depth = np.zeros(n, dtype=np.int8)
        for ring in rings:
            mule_member[ring.members] = True
            mule_window[ring.members, 0] = ring.window_start_day
            mule_window[ring.members, 1] = ring.window_start_day + ring.window_len_days + 3
            mule_layer[ring.members] = ring.layer_of_member.astype(np.int8)
            mule_depth[ring.members] = np.int8(int(ring.layer_of_member.max()) + 1)
        return cls(
            fraud_user=fraud_user, fraud_start_day=fraud_start, fraud_len_days=fraud_len,
            stress_user=stress_user, stress_start_month=stress_start,
            stress_len_months=stress_len, stress_severity=stress_sev,
            mule_member=mule_member, mule_window=mule_window,
            mule_layer=mule_layer, mule_depth=mule_depth,
        )


# --------------------------------------------------------------------------- world
@dataclass
class World:
    """Everything phase B needs, deterministic from ``cfg.seed`` (read-only)."""

    cfg: WorldConfig
    calendar: Calendar
    merchants: MerchantUniverse
    personas: PersonaTable
    transfers: TransferGraph
    campaigns: CampaignCalendar
    episodes: EpisodeAssignment

    @classmethod
    def build(cls, cfg: WorldConfig) -> World:
        """Phase A: construct the full world from the config seed."""
        cal = Calendar.build(cfg)
        personas = sample_personas(cfg, np.random.default_rng((cfg.seed, _NS_WORLD, _S_PERSONAS)))
        merchants = MerchantUniverse.build(cfg, np.random.default_rng((cfg.seed, _NS_WORLD, _S_MERCHANTS)))
        transfers = TransferGraph.build(cfg, cal, personas, np.random.default_rng((cfg.seed, _NS_WORLD, _S_GRAPH)))
        campaigns = CampaignCalendar.build(cfg, cal, np.random.default_rng((cfg.seed, _NS_WORLD, _S_CAMPAIGNS)))
        episodes = EpisodeAssignment.build(
            cfg, cal, personas, transfers.rings, np.random.default_rng((cfg.seed, _NS_WORLD, _S_EPISODES))
        )
        return cls(cfg=cfg, calendar=cal, merchants=merchants, personas=personas,
                   transfers=transfers, campaigns=campaigns, episodes=episodes)
