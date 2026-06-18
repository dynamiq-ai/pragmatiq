'use client';

import { useState } from 'react';

type Kind = 'numeric' | 'categorical' | 'text';
type FieldSpec = {
  key: string;
  value: string;
  kind: Kind;
  repr: string; // conceptual id/bucket shown for numeric & categorical
  bpe?: string[]; // sub-word pieces for the text → BPE path
};
type Sample = { id: string; label: string; source: string; delta: string; fields: FieldSpec[] };

// Illustrative tokenizations (conceptual ids/buckets) — the real tokenizer fits these
// from data; this shows the SHAPE of the key–value–time scheme.
const SAMPLES: Sample[] = [
  {
    id: 'txn',
    label: 'Card transaction',
    source: 'transaction',
    delta: '8·ln(1+Δt/8) ≈ 5.1',
    fields: [
      { key: 'amount', value: '42.10', kind: 'numeric', repr: 'bucket 41 / 64' },
      { key: 'mcc', value: '5411', kind: 'categorical', repr: 'id #812' },
      { key: 'currency', value: 'GBP', kind: 'categorical', repr: 'id #3' },
      {
        key: 'merchant',
        value: 'TESCO STORES 4521',
        kind: 'text',
        repr: '⟨text⟩ → frozen vector',
        bpe: ['TESCO', '▁STORES', '▁45', '21'],
      },
    ],
  },
  {
    id: 'app',
    label: 'App session',
    source: 'app',
    delta: '8·ln(1+Δt/8) ≈ 2.4',
    fields: [
      { key: 'screen', value: 'home', kind: 'categorical', repr: 'id #1' },
      { key: 'action', value: 'view', kind: 'categorical', repr: 'id #2' },
      { key: 'duration_s', value: '37', kind: 'numeric', repr: 'bucket 18 / 64' },
    ],
  },
  {
    id: 'transfer',
    label: 'P2P transfer',
    source: 'transaction',
    delta: '8·ln(1+Δt/8) ≈ 7.8',
    fields: [
      { key: 'amount', value: '250.00', kind: 'numeric', repr: 'bucket 55 / 64' },
      { key: 'direction', value: 'out', kind: 'categorical', repr: 'id #2' },
      {
        key: 'counterparty',
        value: 'dev_7611194361',
        kind: 'text',
        repr: '⟨text⟩ → frozen vector',
        bpe: ['dev', '_', '761', '119', '4361'],
      },
    ],
  },
];

const KIND_COLOR: Record<Kind, string> = {
  numeric: 'border-sky-400/50 bg-sky-400/10',
  categorical: 'border-emerald-400/50 bg-emerald-400/10',
  text: 'border-amber-400/50 bg-amber-400/10',
};

function Token({ k, v, kind }: { k: string; v: string; kind?: Kind }) {
  return (
    <span
      className={`inline-flex flex-col rounded-md border px-2 py-1 text-center font-mono text-xs ${
        kind ? KIND_COLOR[kind] : 'border-fd-border bg-fd-muted/50'
      }`}
    >
      <span className="text-[10px] text-fd-muted-foreground">{k}</span>
      <span>{v}</span>
    </span>
  );
}

export function TokenizationWalkthrough() {
  const [sampleId, setSampleId] = useState(SAMPLES[0].id);
  const [embed, setEmbed] = useState(false);
  const sample = SAMPLES.find((s) => s.id === sampleId)!;

  return (
    <div className="my-6 rounded-xl border border-fd-border bg-fd-card p-4 not-prose">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {SAMPLES.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setSampleId(s.id)}
              aria-pressed={s.id === sampleId}
              className={`cursor-pointer rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                s.id === sampleId
                  ? 'bg-fd-primary text-fd-primary-foreground'
                  : 'bg-fd-muted/60 hover:bg-fd-accent'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={embed}
            onChange={(e) => setEmbed(e.target.checked)}
            className="size-4 accent-[var(--brand-teal)]"
          />
          Nemotron text mode
        </label>
      </div>

      {/* Token stream */}
      <div className="mt-4 flex flex-wrap items-start gap-2">
        <Token k="source" v={sample.source} />
        {sample.fields.map((f) => {
          if (f.kind === 'text') {
            if (embed) {
              return (
                <span key={f.key} className="inline-flex flex-col items-center gap-0.5">
                  <Token k={f.key} v="⟨text⟩" kind="text" />
                  <span className="text-[10px] text-fd-muted-foreground">→ frozen vector</span>
                </span>
              );
            }
            return (
              <span
                key={f.key}
                className="inline-flex flex-col items-center gap-0.5 rounded-md border border-dashed border-amber-400/50 p-1"
              >
                <span className="flex flex-wrap gap-1">
                  {f.bpe!.map((piece, i) => (
                    <Token key={i} k={i === 0 ? f.key : '·'} v={piece} kind="text" />
                  ))}
                </span>
                <span className="text-[10px] text-fd-muted-foreground">BPE pieces</span>
              </span>
            );
          }
          return <Token key={f.key} k={f.key} v={f.repr} kind={f.kind} />;
        })}
        <Token k="time" v={sample.delta} />
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-3 text-xs text-fd-muted-foreground">
        <span><span className="mr-1 inline-block size-2 rounded-sm bg-sky-400/60" />numeric → percentile bucket</span>
        <span><span className="mr-1 inline-block size-2 rounded-sm bg-emerald-400/60" />categorical → id</span>
        <span><span className="mr-1 inline-block size-2 rounded-sm bg-amber-400/60" />text → {embed ? 'one frozen-embedding sentinel' : 'BPE sub-word pieces'}</span>
      </div>
      <p className="mt-3 text-xs text-fd-muted-foreground">
        Conceptual ids/buckets — the real tokenizer fits the vocabulary and bucket edges
        from your data. Toggle <em>Nemotron text mode</em> to see high-cardinality text
        collapse from BPE pieces to a single sentinel a frozen encoder maps to a vector.
      </p>
    </div>
  );
}
