'use client';

import { useState } from 'react';
import { ChevronRight } from 'lucide-react';

type Stage = {
  id: string;
  label: string;
  tag: string;
  blurb: string;
  inputs: string;
  output: string;
  facts: string[];
};

const STAGES: Stage[] = [
  {
    id: 'token',
    label: 'Token embedding',
    tag: 'produces token vectors',
    blurb:
      'Every field becomes a (key, value, time) token. One shared table embeds both keys and values; a sinusoidal within-field position is added for multi-piece text.',
    inputs: 'key_ids, value_ids, positions',
    output: 'x — per-token vectors [T, d]',
    facts: [
      'x = E(key) + E(value) + sinusoidal(position)',
      'The table is tied to the MLM output projection',
      'Numeric → percentile bucket, categorical → id, text → BPE pieces (or one frozen-embedding sentinel in the Nemotron variant)',
    ],
  },
  {
    id: 'event',
    label: 'Event encoder',
    tag: 'produces a per-event vector',
    blurb:
      'Each event is encoded independently. A learned [EVT] marker is prepended and attention is confined within the event; calendar features are added to the event vector.',
    inputs: 'x, cu_seqlens_event',
    output: 'ẑ_e (per-token) · z_e (per-event) [E, d]',
    facts: [
      'Within-event (block-diagonal) attention — no cross-event leakage at this stage',
      'z_e = [EVT] output + CalendarEmbedding(hour, day-of-week, day-of-month)',
      'use_rope = false (field order inside an event is not meaningful)',
    ],
  },
  {
    id: 'profile',
    label: 'Profile-state encoder',
    tag: 'produces the profile state',
    blurb:
      "Each user's profile tokens (static attributes + lifelong milestones) are encoded under a [USR] marker, with TimeRoPE on each item's log-seconds.",
    inputs: 'profile tokens, per-item log-seconds',
    output: 'z_a — profile state [n_users, d]',
    facts: [
      'TimeRoPE positions = log-seconds since each lifelong milestone',
      'Static attributes sit at time 0; the [USR] marker is anchored at 0',
      'Profile text stays BPE even in the Nemotron variant',
    ],
  },
  {
    id: 'history',
    label: 'History encoder',
    tag: 'produces history states',
    blurb:
      'A user is a sequence [z_a, z_e…] — profile state followed by per-event vectors. TimeRoPE runs on log-seconds-to-the-most-recent-event; the [USR] slot output is the user embedding.',
    inputs: '[z_a, z_e…], event_time_log',
    output: 'z_h[USR] · z_h[event] [E, d]',
    facts: [
      'Bidirectional, pre-norm, GELU, ffn = 4·d, dropout 0.1 (all three encoders)',
      'Continuous-time RoPE encodes elapsed time between events, not ordinal distance',
      'z_h[USR] is the per-user output; z_h[event] are per-event history states',
    ],
  },
  {
    id: 'embedding',
    label: 'User embedding',
    tag: 'the output vector',
    blurb:
      'The single dense vector for the whole history — what downstream tasks consume. The MLM head reconstructs masked values from ẑ_e, z_h(event), and z_h(USR) during pretraining.',
    inputs: 'z_h[USR]',
    output: 'embedding [n_users, d]',
    facts: [
      'Probed (gradient boosting), LoRA-fine-tuned, used as GNN node features, or served',
      'Pretraining head: concat [ẑ_e, z_h(event), z_h(USR)] ∈ R^{3d} → Linear(3d→d) → tied logits',
      'Cross-entropy + label smoothing 0.1; + MSE on text in the Nemotron variant',
    ],
  },
];

export function ArchitectureExplorer() {
  const [active, setActive] = useState(0);
  const s = STAGES[active];

  return (
    <div className="my-6 rounded-xl border border-fd-border bg-fd-card not-prose">
      {/* Stage flow */}
      <div className="flex flex-wrap items-center gap-1.5 border-b border-fd-border p-3">
        {STAGES.map((stage, i) => (
          <div key={stage.id} className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setActive(i)}
              aria-pressed={i === active}
              className={`cursor-pointer rounded-lg px-3 py-2 text-left text-sm font-medium transition-colors ${
                i === active
                  ? 'bg-fd-primary text-fd-primary-foreground'
                  : 'bg-fd-muted/60 hover:bg-fd-accent'
              }`}
            >
              <span className="block">{stage.label}</span>
              <span
                className={`block text-[11px] ${
                  i === active ? 'opacity-90' : 'text-fd-muted-foreground'
                }`}
              >
                {stage.tag}
              </span>
            </button>
            {i < STAGES.length - 1 ? (
              <ChevronRight className="size-4 shrink-0 text-fd-muted-foreground" aria-hidden />
            ) : null}
          </div>
        ))}
      </div>

      {/* Detail panel */}
      <div className="p-5">
        <p className="text-sm leading-relaxed text-fd-muted-foreground">{s.blurb}</p>

        <div className="mt-4 grid overflow-hidden rounded-lg border border-fd-border sm:grid-cols-[1fr_auto_1fr]">
          <div className="p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-fd-muted-foreground">
              Input
            </div>
            <div className="mt-2 font-mono text-[13px] leading-relaxed">{s.inputs}</div>
          </div>
          <div className="hidden items-center justify-center px-2 text-fd-muted-foreground sm:flex" aria-hidden>
            <ChevronRight className="size-4" />
          </div>
          <div className="border-t border-fd-border p-4 sm:border-t-0 sm:border-l">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-fd-muted-foreground">
              Output
            </div>
            <div className="mt-2 font-mono text-[13px] leading-relaxed text-fd-primary">{s.output}</div>
          </div>
        </div>

        <ul className="mt-5 space-y-2 text-sm">
          {s.facts.map((f) => (
            <li key={f} className="flex gap-2.5 leading-relaxed">
              <span className="mt-[7px] size-1.5 shrink-0 rounded-full bg-fd-primary" aria-hidden />
              <span>{f}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
