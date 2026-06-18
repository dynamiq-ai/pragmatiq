import type { ReactNode } from 'react';

type Variant = 'paper' | 'ours' | 'optional' | 'neutral';

const STYLES: Record<Variant, string> = {
  paper: 'bg-fd-accent text-fd-accent-foreground',
  ours: 'bg-fd-primary/10 text-fd-primary',
  optional: 'bg-amber-500/15 text-amber-700 dark:text-amber-400',
  neutral: 'bg-fd-muted text-fd-muted-foreground',
};

/** A small inline pill — e.g. "Paper", "Our addition", "Optional" — for fidelity tags. */
export function Badge({
  variant = 'neutral',
  children,
}: {
  variant?: Variant;
  children: ReactNode;
}) {
  return (
    <span
      className={`inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${STYLES[variant]}`}
    >
      {children}
    </span>
  );
}
