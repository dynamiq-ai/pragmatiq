import defaultMdxComponents from 'fumadocs-ui/mdx';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import { Step, Steps } from 'fumadocs-ui/components/steps';
import type { MDXComponents } from 'mdx/types';
import { ParamTable } from '@/components/ui/ParamTable';
import { DecisionCard } from '@/components/ui/DecisionCard';
import { Badge } from '@/components/ui/Badge';
import { ApiReference } from '@/components/ui/ApiReference';
import { ModelSizes } from '@/components/ui/ModelSizes';
import { ArchitectureExplorer } from '@/components/viz/ArchitectureExplorer';
import { TokenizationWalkthrough } from '@/components/viz/TokenizationWalkthrough';

// Components available in every MDX page without an import. Fumadocs ships the base
// set (Callout, Cards/Card, code blocks, headings); we add layout helpers and the
// pragmatiq-specific, facts-driven components.
export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    Tab,
    Tabs,
    Step,
    Steps,
    ParamTable,
    DecisionCard,
    Badge,
    ApiReference,
    ModelSizes,
    ArchitectureExplorer,
    TokenizationWalkthrough,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
