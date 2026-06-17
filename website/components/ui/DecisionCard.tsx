import type { ReactNode } from 'react';

/**
 * A "design decision" card in the reproducibility voice: what we chose, why, and the
 * alternative we considered. Used throughout the Concepts and Design-decisions sections
 * so every non-obvious default is explained rather than assumed.
 */
export function DecisionCard({
  title,
  why,
  alternative,
  children,
}: {
  title: string;
  why?: ReactNode;
  alternative?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="not-prose my-4 rounded-xl border border-fd-border bg-fd-card p-5">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
        <span className="inline-flex h-5 items-center rounded-md bg-fd-primary/10 px-2 text-[11px] font-semibold uppercase leading-none tracking-wide text-fd-primary ring-1 ring-fd-primary/20">
          Decision
        </span>
        <h4 className="m-0 text-base font-semibold leading-tight">{title}</h4>
      </div>
      {children ? <div className="mt-3 text-sm">{children}</div> : null}
      {why ? (
        <p className="mt-3 text-sm">
          <span className="font-semibold text-fd-primary">Why:</span> {why}
        </p>
      ) : null}
      {alternative ? (
        <p className="mt-1.5 text-sm">
          <span className="font-semibold text-fd-muted-foreground">Alternative considered:</span>{' '}
          {alternative}
        </p>
      ) : null}
    </div>
  );
}
