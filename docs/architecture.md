# pragmatiq architecture: model and temporal encoding

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv [2604.08649](https://arxiv.org/abs/2604.08649)).
> It is not affiliated with or endorsed by Revolut.

This document is for an ML engineer who wants to understand *how* the model
works, not just how to run it. It walks the data from raw banking events all the
way to a per-user embedding, and explains the choices that make the architecture
fit irregular, heterogeneous event histories. Every claim here is grounded in the
source; file references are given so you can read the code alongside.

Source map for this document:

| Concern | File |
| --- | --- |
| Tokenization, time transform, percentile binning, caps | `pragmatiq/data/tokenizer.py` |
| Padding-free varlen packing | `pragmatiq/data/collate.py` |
| Token embedding, TimeRoPE, calendar embedding | `pragmatiq/models/embeddings.py` |
| Transformer blocks, varlen attention, SDPA fallback | `pragmatiq/models/layers.py` |
| Encoder stack, forward pipeline, segment assembly | `pragmatiq/models/pragmatiq.py` |
| MLM head, text reconstruction, losses | `pragmatiq/models/heads.py` |
| Frozen text encoders (Nemotron variant) | `pragmatiq/models/text_encoder.py` |
| MLM masking | `pragmatiq/training/masking.py` |

---

## 1. Overview

The problem: a bank holds, per customer, a long and *irregular* stream of
heterogeneous events — card transactions, in-app navigation, trades, comms —
plus a slowly-changing profile (attributes and lifelong milestones like account
opening). These streams differ in schema, fire at wildly different rates, and
carry numbers, categorical codes, and free text. The goal is to compress one
customer's entire history into a single dense vector (a *user embedding*) that
downstream tasks (fraud, credit, churn, AML) can probe or fine-tune on.

pragmatiq does this with a **four-encoder stack** plus a tokenizer that turns
every field into a (key, value, time) triple:

```
raw history ──► tokenizer ──► PackedBatch (no padding, cu_seqlens)
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  TokenEmbedding   E(key) + E(value) + within-field sin/cos pos  │
   └───────────────────────────────────────────────────────────────┘
                                   │  per-token vectors x
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                      ▼
   EventEncoder (per event, [EVT] marker)            ProfileStateEncoder
   within-event attention; + CalendarEmbedding        ([USR] marker,
        │  ẑ_e (tokens), z_e (per event)              TimeRoPE on lifelong)
        │                                                      │  z_a
        └──────────────┬───────────────────────────────────────┘
                       ▼
              HistoryEncoder over [z_a, z_e…]
              TimeRoPE on log-seconds-to-last-event
                       │
                       ▼
          z_h[USR] = the user embedding   ·   z_h[event] = per-event states
```

The four encoders are: `TokenEmbedding` (not attention — just the input
embedding), `EventEncoder`, `ProfileStateEncoder`, and `HistoryEncoder`. All
three attention encoders are **bidirectional, pre-norm, GELU MLP with ffn = 4·d,
dropout 0.1** (`pragmatiq/models/layers.py`). The `[USR]` slot output of the
history encoder *is* the user embedding (`PragmaModel.embed_users`).

---

## 2. Tokenization — the key–value–time scheme

`pragmatiq/data/tokenizer.py` fits one vocabulary over the whole dataset, then
`encode(UserRecord) → TokenizedRecord` turns one customer's history into flat
arrays plus CSR offsets. The scheme emits **one key token per field**, paired
with a value representation chosen by the field's *kind*. Keys and values share a
single vocabulary space (the model later ties one embedding table across both).

### Field kinds

The `_finalize` step classifies each observed key into one of three kinds:

- **Numeric** → `PercentileBinner`. A key is treated as a continuous numeric only
  when nearly all of its values parse as finite floats (`ratio >= 0.995`, at least
  32 observations) *and* it is high-cardinality (more distinct values than
  `numeric_min_cardinality`, default `4 × n_buckets`). The binner learns
  percentile bucket edges from a per-key sample and adds a **dedicated zero
  bucket** (bucket 0); total buckets are `len(edges) + 2`. Out-of-range
  magnitudes clip into the end buckets, so inference never fails on an unseen
  amount. The high-cardinality gate keeps low-cardinality numeric *codes* (MCC
  `"5411"`, version `"10.23"`) out of the binner — they stay categorical — while a
  genuinely continuous quantity (amount, price, balance) is binned. `force_numeric`
  / `force_categorical` override the heuristic per key.
- **Categorical** (low-cardinality string, distinct count ≤ `categorical_threshold`,
  default 1000) → **one categorical token per value**, assigned in
  frequency-descending order.
- **Text** (high-cardinality string, distinct count > the threshold) → BPE by
  default, or a single frozen-embedding sentinel in the Nemotron variant
  (see below).

Per token the tokenizer stores `key_id`, `value_id`, and a within-field
`position` (0 for a single-token value; `0..n` for BPE sub-word pieces).

### Text: BPE (default) vs the frozen-embedding sentinel (Nemotron variant)

The text path is governed by `TokenizerConfig.text_value_mode`:

- **`"bpe"` (default):** a byte-level BPE (HF `tokenizers`) is trained over the
  textual-field corpus, sized so the total vocabulary lands near `target_vocab`
  (default ≈ 28k). A text value encodes to multiple sub-word piece tokens, each
  sharing the same `key_id` but with increasing within-field `position`.
- **`"embed"` (the PRAGMA+Nemotron variant):** an event text field emits **one
  sentinel token** (`value_id = [UNK]`, `position = 0`) and the raw string is
  carried in a compact `text_values` list, with `is_text[token] = 1`. At model
  time a *frozen* text encoder maps the string to a vector. This applies only to
  **event** text fields; **profile text stays BPE**.

### Time and calendar features

Time is encoded two ways per record. The core transform is the paper's
log-seconds:

```python
def time_encode(delta_seconds):
    return 8.0 * np.log1p(delta_seconds / 8.0)   # 8 · ln(1 + Δt/8)
```

- **Events:** each event gets the log-seconds *to the most recent event*
  (`Δt = ts_last - ts`, clamped ≥ 0), so the most recent event sits near 0 and
  older events grow. Each event also gets **calendar features**: hour-of-day,
  day-of-week, day-of-month, derived from the event's instant in
  `calendar_tz` (default UTC; settable to a local zone, DST-correct).
- **Profile:** static attributes get time 0. Lifelong milestones get log-seconds
  *since they occurred*, measured from the profile snapshot (`as_of`), so the
  lifelong axis shares the event stream's recency orientation.

### Special tokens

`[PAD]=0, [MASK]=1, [UNK]=2, [USR]=3, [EVT]=4`. `[PAD]` exists for the dense
reformulation; the varlen path never pads. Unseen keys/values at encode time map
to `[UNK]` with a one-time logged warning — never a `KeyError`.

### Pre-training caps

Real histories are heavy-tailed, so `encode` applies three caps (paper defaults;
each `None` disables it). At synthetic scale none of them bind, so output is
unchanged; they only act on large real-world records:

| Cap | Default | Effect |
| --- | --- | --- |
| `max_event_tokens` | 24 | each event keeps its first 24 tokens |
| `max_profile_tokens` | 200 | profile keeps the first whole items fitting 200 tokens |
| `max_events_per_user` | 6500 | a longer history keeps only its most recent events |

---

## 3. Temporal encoding

The defining choice: time enters the attention via **rotary embeddings over a
continuous position**, where that position is the log-seconds time feature — not
an integer index. This is `TimeRoPE` in `pragmatiq/models/embeddings.py`.

### Continuous RoPE

Standard RoPE rotates query/key pair `i` by `n · inv_freq[i]` for integer
position `n`. `TimeRoPE` instead uses a real-valued position `p` (the log-seconds
feature), so a token at log-seconds `p` rotates pair `i` by `p · inv_freq[i]`:

```python
freqs = position[:, None].float() * self.inv_freq[None, :]   # position is a real number
```

The relative-position structure that makes RoPE work is preserved (the dot
product of two rotated vectors depends on their *difference* in position), so
attention scores naturally encode the *elapsed time between two events* rather
than their ordinal distance. This is what fits **irregular event spacing**: two
events one second apart and two events one month apart get genuinely different
relative rotations, whereas integer positions would treat "the previous event"
identically regardless of whether it was a second or a month ago. The frequency
ladder `base` (`rope_base`, default 10 000) is a tunable GUESS exposed in config.

Positions are kept in fp32 even under bf16 training (`assemble_segments` upcasts
the position array) so log-second resolution is never quantized; the cos/sin are
cast back to the activation dtype only inside `rotate`.

### Where each axis is applied

- **Event encoder:** `use_rope=False`. Within an event there is no meaningful
  ordering of fields, so the event encoder uses no positional rotation at all
  (within-field order for BPE pieces is already carried by the additive
  sinusoidal position in `TokenEmbedding`).
- **Profile encoder:** `use_rope=True`. The per-item position is the lifelong
  log-seconds (static attributes at 0); the `[USR]` marker is anchored at
  log-seconds 0.
- **History encoder:** `use_rope=True`. The per-event position is
  `event_time_log` (log-seconds to the last event); the `[USR]` marker is again
  anchored at 0.

### Calendar embedding

Calendar features take a separate, additive path (`CalendarEmbedding`): for each
event, `(hour, dow, dom)` become sin/cos features (so 12am is adjacent to 11pm,
etc.), through a 2-layer MLP (`Linear(6→d) → GELU → Linear(d→d)`), and the result
is **added to the per-event vector** `z_e` (see §4). This lets the model use
day/night and weekday/weekend structure independently of the elapsed-time RoPE.

---

## 4. The encoder stack

`PragmaModel.forward` (`pragmatiq/models/pragmatiq.py`) runs three encoders in
order. A small helper, `assemble_segments`, implements the **prefix-marker
mechanism** used twice below: given per-segment lengths and a per-segment prefix
vector, it prepends one prefix row to each segment in the flat layout and returns
index tensors (`prefix_idx`, `token_dst`) that recover the prefix rows and the
token rows afterward, plus a per-row RoPE position array.

### 4.1 Token embedding

```python
x = E(key_ids) + E(value_ids) + sinusoidal(positions)
```

One shared table `E` embeds both keys and values; a fixed sinusoidal table adds
the within-field position (clamped to `max_position`, default 64). The table is
tied to the MLM output projection (§6). In the Nemotron variant, the projected
frozen text vector is *added* onto fed text tokens here (`text_proj` is the only
trainable piece of the text input path).

### 4.2 Event encoder → z_e

Each **event is a segment** (`cu_seqlens_event`). A learned `[EVT]` marker is
prepended per event, the stack runs with **within-event (block-diagonal)
attention**, and two outputs are read out:

- `ẑ_e = h[token_dst]` — the per-token outputs (used by the MLM head).
- the `[EVT]` output `h[prefix_idx]`, to which the **calendar embedding** is
  added to form the per-event vector:

```python
z_e = h[prefix_idx] + calendar(event_hour, event_dow, event_dom)   # [E, d]
```

Because attention is confined to within each event, every event is encoded
**independently** — an event's representation never depends on its neighbours at
this stage.

### 4.3 Profile-state encoder → z_a

Each **user's profile tokens form a segment**. A `[USR]` marker is prepended;
`TimeRoPE` is applied on the per-token log-seconds (each token inherits its
profile item's time; `[USR]` anchored at 0). The `[USR]` output is the profile
state:

```python
z_a = h[prefix_idx]   # [n_users, d]
```

### 4.4 History encoder → z_h (and the user embedding)

Now each **user is a segment**, and the segment's elements are
`[z_a, z_e…]` — the profile state followed by that user's per-event vectors. The
`[USR]` slot is `z_a` itself (it is passed in as the prefix vector). `TimeRoPE`
runs on the per-event log-seconds-to-last-event (`[USR]` at 0). The encoder reads
out:

- `z_h[USR] = h[prefix_idx]` — **the user embedding** `[n_users, d]`.
- `z_h[event] = h[token_dst]` — per-event history states `[E, d]`.

These three representations (`ẑ_e`, `z_h[event]`, `z_h[USR]`) are exactly what the
MLM head consumes.

---

## 5. Padding-free varlen attention

`VarlenCollator` (`pragmatiq/data/collate.py`) packs a batch of records with **no
padding anywhere**. Token-level arrays are concatenated across every event of
every user; event-level arrays across every user; structure is recovered by
cumulative-sequence-length vectors (the flash-attn varlen convention):

- `cu_seqlens_event` — token boundaries per event (`len = n_events + 1`)
- `cu_seqlens_history` — event boundaries per user (`len = n_users + 1`)
- `cu_seqlens_profile_item` / `cu_seqlens_profile` — the profile equivalents

The event encoder attends within each `cu_seqlens_event` segment; the history
encoder within each `cu_seqlens_history` segment. Each encoder is told the
longest segment (`max_seqlen`) so the fallback can size its block.

`varlen_self_attention` (`pragmatiq/models/layers.py`) has two paths that are
*numerically equivalent within a precision*:

- **flash-attn varlen** on CUDA in fp16/bf16, when `flash_attn_varlen_func` is
  available;
- **SDPA fallback** otherwise (always on CPU): each segment is scattered into a
  padded `[n_seg, max_len, H, hd]` block, and a key-padding mask hides the
  padding so attention is confined to real tokens.

Because attention is **within-segment** in both paths, padding in the SDPA block
is purely structural — the mask removes it entirely from the softmax. That is why
the fp32 SDPA forward matches a naive padded per-segment forward to atol 1e-4
(the padding-equivalence guarantee). The flash-attn path runs in
fp16/bf16 and matches the SDPA path only to bf16 precision (~1e-2). On CUDA,
fp32 inference (no autocast) also takes the SDPA path, since flash kernels accept
only half precision.

---

## 6. Pre-training objective

The pretraining head is `MLMHead` (`pragmatiq/models/heads.py`). For each token
it predicts, it builds a **3·d context** by concatenating three views of that
token:

```
context = [ ẑ_e(token),  z_h(its event),  z_h(USR) ]  ∈ R^{3d}
            within-event   per-event        whole-user
            token state     history state    embedding
```

This is projected `Linear(3d → d)`, then read out to vocabulary logits through
the **tied** token-embedding weights (`h @ embedding_weight.t()`). Training is
cross-entropy with **label smoothing 0.1** (`mlm_loss`). Giving each prediction
all three context levels means a masked value is reconstructed jointly from its
local event, the user's history of that event, and the user's global state.

### Masking (`pragmatiq/training/masking.py`)

Three selection modes are unioned per batch:

- **token**: each token independently with `p = 0.15`;
- **event**: each event with `p = 0.10` → all of its tokens masked;
- **key**: per user, each key with `p = 0.10` → all of that user's tokens with
  that key masked ("all values of sampled keys").

Priority when modes overlap is **event > key > token** (recorded in `mask_type`
so the trainer can log per-mode loss). Of the selected positions, **10% become
`[UNK]` and are excluded from the loss** (label `-100`); the other 90% become
`[MASK]` with the original `value_id` as the CE target. The **key token is kept**
— we know the key and predict its value. Non-selected positions are also `-100`.

### Text reconstruction (Nemotron variant)

In the Nemotron variant, text tokens carry a frozen embedding rather than a vocab
id, so a masked text token is reconstructed with **MSE against the frozen text
embedding** instead of cross-entropy. Such a token is recorded in `text_loss_idx`
(never in the CE `labels`). A parallel `Linear(3d → text_dim)` head
(`reconstruct_text`) predicts the frozen vector from the same 3·d context, and
the trainer combines the two terms:

```
loss = CE + λ · MSE        # λ = text_loss_weight, default 1.0
```

(`pragmatiq/training/pretrainer.py`). `feed_text` gates which text tokens may
still supply their embedding to the input — every text token except those masked
or `[UNK]`-dropped this step — so the model must reconstruct the hidden ones. The
frozen encoder is never trained or saved; its outputs are the regression targets.

The frozen encoders (`pragmatiq/models/text_encoder.py`) are: `hash` (a
deterministic, dependency-free, *non-semantic* stand-in seeded per string, for
CI/CPU) and `nemotron` (the paper's frozen Nemotron embedder via 🤗
`transformers`, mean-pooled last hidden state under `no_grad`). The built
encoder's `.dim` is authoritative for the MSE target width.

---

## 7. Model sizes

`ModelConfig.preset` (`pragmatiq/models/pragmatiq.py`) defines four sizes. Depths
are `(profile / event / history)` block counts. `dim = d`, `n_heads`, and every
block uses ffn = 4·d, dropout 0.1.

| Preset | dim | heads | depth (prof/event/hist) | target params | Note |
| --- | --- | --- | --- | --- | --- |
| `nano` | 64 | 2 | 1 / 2 / 1 | ~1M | **not in the paper**; CPU/CI + `quickstart` |
| `small` | 192 | 3 | 1 / 5 / 2 | 10M | matches paper 10M |
| `medium` | 512 | 8 | 3 / 16 / 6 | 100M | matches paper 100M |
| `large` | 1024 | 16 | 9 / 45 / 18 | 1B | matches paper 1B |

`small` / `medium` / `large` correspond to the paper's 10M / 100M / 1B sizes.
`nano` is pragmatiq's own addition so the gates and `pragmatiq quickstart` run
end-to-end on a CPU in minutes. `overrides` lets callers tune any architecture
field (notably `rope_base`, `dropout`) on top of the size table.

---

## 8. What is ours, not the paper

- **From the PRAGMA paper:** the core representation — key–value–time
  tokenization, the `8·ln(1+Δt/8)` time transform, continuous-time RoPE, the
  profile/event/history encoder stack, the 3·d MLM head with tied logits and
  label smoothing, and the token/event/key masking scheme. The **PRAGMA+Nemotron
  variant** (frozen text-embedding reconstruction with MSE) is also from the
  paper.
- **pragmatiq's own extension:** the **AML GraphSAGE work** — building a transfer
  graph from `transfers.parquet` and running a GraphSAGE GNN over frozen pragmatiq
  embeddings to recover money-mule ring membership — is **NOT in the PRAGMA
  paper**. It is an independent extension built on top of the paper's
  representation. The `nano` size and the synthetic data generator are likewise
  pragmatiq's own.
