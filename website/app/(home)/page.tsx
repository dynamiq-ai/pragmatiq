import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Boxes, Database, GitBranch, Gauge } from 'lucide-react';
import { ArchitectureExplorer } from '@/components/viz/ArchitectureExplorer';

const PILLARS = [
  {
    icon: Database,
    title: 'Synthetic banking data',
    body: 'A deterministic, agent-based generator: transactions, app sessions, transfers, profiles, and causal fraud / default / AML labels — same seed, byte-identical output.',
  },
  {
    icon: Boxes,
    title: 'Key–value–time model',
    body: 'A padding-free PyTorch encoder stack (profile · event · history) with TimeRoPE on continuous log-seconds and a tied masked-LM head. Runs on CPU first.',
  },
  {
    icon: Gauge,
    title: 'Evaluate & serve',
    body: 'Gradient-boosting probes (ROC-AUC + PR-AUC), LoRA fine-tuning, ONNX/Triton serving, and a one-command deploy — the full path from data to inference.',
  },
  {
    icon: GitBranch,
    title: 'AML over the graph',
    body: "Our own extension (not in the paper): a GraphSAGE ablation over the transfer graph that recovers money-mule rings a per-user embedding can't see.",
  },
];

const FIDELITY: [string, string][] = [
  ['Key–value–time tokenization + 8·ln(1+Δt/8) time transform', 'Paper'],
  ['Profile / event / history encoders + tied 3d MLM head', 'Paper'],
  ['PRAGMA+Nemotron frozen-text-embedding variant (MSE)', 'Paper · opt-in'],
  ['Synthetic data generator', 'Our addition'],
  ['AML transfer-graph GraphSAGE ablation', 'Our addition'],
  ['Gradient-boosting downstream probe', 'Our default'],
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      {/* Hero */}
      <section className="relative mx-auto w-full max-w-5xl px-6 pt-20 pb-16 text-center">
        <div className="mx-auto mb-6 w-fit rounded-2xl bg-white p-3 shadow-sm ring-1 ring-black/5">
          <Image src="/brand/mark.png" alt="pragmatiq" width={72} height={72} priority />
        </div>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">pragmatiq</h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg text-fd-muted-foreground">
          An open, reproducible implementation of the PRAGMA recipe for banking
          foundation models — turning timestamped key–value event histories into
          user embeddings you can probe, fine-tune, graph, and serve.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/docs"
            className="inline-flex items-center gap-2 rounded-lg bg-fd-primary px-5 py-2.5 font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
          >
            Get started <ArrowRight className="size-4" />
          </Link>
          <Link
            href="/docs/concepts/architecture"
            className="inline-flex items-center gap-2 rounded-lg border border-fd-border px-5 py-2.5 font-medium transition-colors hover:bg-fd-accent"
          >
            How it works
          </Link>
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2 text-sm">
          <code className="rounded-md bg-fd-muted px-2.5 py-1 font-mono">
            pip install -e &quot;.[dev]&quot;
          </code>
          <span className="text-fd-muted-foreground">·</span>
          <code className="rounded-md bg-fd-muted px-2.5 py-1 font-mono">pragmatiq quickstart</code>
        </div>
      </section>

      {/* What's included */}
      <section className="mx-auto w-full max-w-5xl px-6 pb-16">
        <div className="grid gap-4 sm:grid-cols-2">
          {PILLARS.map(({ icon: Icon, title, body }) => (
            <div
              key={title}
              className="rounded-xl border border-fd-border bg-fd-card p-5 text-left"
            >
              <Icon className="size-6 text-fd-primary" />
              <h3 className="mt-3 font-semibold">{title}</h3>
              <p className="mt-1.5 text-sm text-fd-muted-foreground">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* The stack, interactive */}
      <section className="mx-auto w-full max-w-5xl px-6 pb-16">
        <h2 className="text-center text-sm font-semibold uppercase tracking-wide text-fd-muted-foreground">
          From events to a user embedding
        </h2>
        <p className="mx-auto mt-2 mb-2 max-w-2xl text-center text-sm text-fd-muted-foreground">
          Click through the four-encoder stack — what each stage takes in, produces, and why.
        </p>
        <ArchitectureExplorer />
      </section>

      {/* Paper fidelity at a glance */}
      <section className="mx-auto w-full max-w-5xl px-6 pb-20">
        <h2 className="text-center text-sm font-semibold uppercase tracking-wide text-fd-muted-foreground">
          Paper fidelity at a glance
        </h2>
        <div className="mt-5 overflow-hidden rounded-xl border border-fd-border">
          {FIDELITY.map(([what, tag], i) => (
            <div
              key={what}
              className={`flex items-center justify-between gap-4 px-5 py-3 text-sm ${
                i % 2 ? 'bg-fd-card' : ''
              }`}
            >
              <span>{what}</span>
              <span
                className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  tag.startsWith('Our')
                    ? 'bg-fd-primary/10 text-fd-primary'
                    : 'bg-fd-accent text-fd-accent-foreground'
                }`}
              >
                {tag}
              </span>
            </div>
          ))}
        </div>
        <p className="mx-auto mt-10 max-w-2xl text-center text-xs text-fd-muted-foreground">
          Built by{' '}
          <a href="https://getdynamiq.ai" className="text-fd-primary hover:underline"
            target="_blank" rel="noreferrer">Dynamiq</a>. pragmatiq is an independent
          implementation inspired by the{' '}
          <a href="https://arxiv.org/abs/2604.08649" className="text-fd-primary hover:underline"
            target="_blank" rel="noreferrer">PRAGMA paper (arXiv 2604.08649)</a>{' '}
          and is not affiliated with or endorsed by Revolut.
        </p>
      </section>
    </main>
  );
}
