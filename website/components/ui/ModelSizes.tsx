import facts from '@/data/facts.json';

type Size = {
  name: string;
  dim: number;
  n_heads: number;
  depth_profile: number;
  depth_event: number;
  depth_history: number;
};

// Paper-nominal parameter targets per preset (small/medium/large = the paper's
// 10M/100M/1B; nano is pragmatiq's CPU/CI size, not in the paper).
const NOMINAL: Record<string, string> = {
  nano: '~1M · CPU/CI',
  small: '10M',
  medium: '100M',
  large: '1B',
};

/** The S/M/L/nano size table, rendered from facts.json (ModelConfig.preset). */
export function ModelSizes() {
  const sizes = facts.model_sizes as Size[];
  return (
    <div className="my-4 overflow-x-auto rounded-lg border border-fd-border not-prose">
      <table className="w-full border-collapse text-sm">
        <thead className="bg-fd-muted/50 text-left">
          <tr>
            <th className="px-3 py-2 font-medium">Preset</th>
            <th className="px-3 py-2 font-medium">dim</th>
            <th className="px-3 py-2 font-medium">heads</th>
            <th className="px-3 py-2 font-medium">depth (profile / event / history)</th>
            <th className="px-3 py-2 font-medium">Nominal</th>
          </tr>
        </thead>
        <tbody>
          {sizes.map((s) => (
            <tr key={s.name} className="border-t border-fd-border">
              <td className="px-3 py-2 font-mono text-fd-primary">{s.name}</td>
              <td className="px-3 py-2 font-mono">{s.dim}</td>
              <td className="px-3 py-2 font-mono">{s.n_heads}</td>
              <td className="px-3 py-2 font-mono">
                {s.depth_profile} / {s.depth_event} / {s.depth_history}
              </td>
              <td className="px-3 py-2 text-fd-muted-foreground">{NOMINAL[s.name] ?? ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
