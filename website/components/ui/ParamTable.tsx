import facts from '@/data/facts.json';

type Field = {
  name: string;
  type: string;
  default: unknown;
  required: boolean;
};

const SOURCES: Record<string, Field[]> = {
  train_config: facts.train_config as Field[],
  tokenizer_config: facts.tokenizer_config as Field[],
};

function fmtDefault(f: Field): string {
  if (f.required) return 'required';
  const d = f.default;
  if (d === null) return 'None';
  if (typeof d === 'boolean') return d ? 'True' : 'False';
  if (Array.isArray(d)) return `(${d.join(', ')})`;
  return String(d);
}

/**
 * Renders a config dataclass's fields (name · type · default) straight from
 * website/data/facts.json, so the documented defaults always match the code.
 * `only` narrows to a subset of fields, in the given order; `notes` adds a
 * per-field human description column.
 */
export function ParamTable({
  config,
  only,
  notes,
}: {
  config: 'train_config' | 'tokenizer_config';
  only?: string[];
  notes?: Record<string, string>;
}) {
  const all = SOURCES[config] ?? [];
  const byName = new Map(all.map((f) => [f.name, f]));
  const fields = only
    ? (only.map((n) => byName.get(n)).filter(Boolean) as Field[])
    : all;

  return (
    <div className="my-4 overflow-x-auto rounded-lg border border-fd-border">
      <table className="w-full border-collapse text-sm">
        <thead className="bg-fd-muted/50">
          <tr className="text-left">
            <th className="px-3 py-2 font-medium">Field</th>
            <th className="px-3 py-2 font-medium">Type</th>
            <th className="px-3 py-2 font-medium">Default</th>
            {notes ? <th className="px-3 py-2 font-medium">Notes</th> : null}
          </tr>
        </thead>
        <tbody>
          {fields.map((f) => (
            <tr key={f.name} className="border-t border-fd-border align-top">
              <td className="px-3 py-2 font-mono text-fd-primary">{f.name}</td>
              <td className="px-3 py-2 font-mono text-xs text-fd-muted-foreground">
                {f.type}
              </td>
              <td className="px-3 py-2 font-mono text-xs">
                <code>{fmtDefault(f)}</code>
              </td>
              {notes ? (
                <td className="px-3 py-2 text-fd-muted-foreground">{notes[f.name] ?? ''}</td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
