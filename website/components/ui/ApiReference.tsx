import facts from '@/data/facts.json';

type ApiFn = { name: string; signature: string; summary: string };

/**
 * Auto-generated Python API reference: renders every public `pragmatiq.api` function
 * (signature + summary) straight from website/data/facts.json, which scripts/docs_facts.py
 * emits from the live docstrings — so the reference cannot drift from the code.
 */
export function ApiReference() {
  const api = facts.api as ApiFn[];
  return (
    <div className="my-4 flex flex-col gap-4 not-prose">
      {api.map((fn) => (
        <div key={fn.name} id={`api-${fn.name}`} className="rounded-xl border border-fd-border bg-fd-card p-4">
          <h3 className="m-0 font-mono text-base font-semibold text-fd-primary">
            api.{fn.name}
          </h3>
          <p className="mt-1 mb-3 text-sm text-fd-muted-foreground">{fn.summary}</p>
          <pre className="overflow-x-auto rounded-lg bg-fd-muted/60 p-3 text-xs">
            <code>{fn.signature}</code>
          </pre>
        </div>
      ))}
    </div>
  );
}
