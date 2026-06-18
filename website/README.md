# pragmatiq documentation site

The documentation + educational website for [pragmatiq](https://github.com/dynamiq-ai/pragmatiq),
built with **Next.js (App Router) + [Fumadocs](https://fumadocs.dev)** (MDX), TypeScript, and
Tailwind. It is a self-contained app in this `website/` folder of the main repo, deployed to
Vercel; the Python project is unaffected.

## Develop

```bash
cd website
pnpm install
pnpm dev          # http://localhost:3000
pnpm build        # production build (run before committing significant changes)
```

Requires Node 20+ (Node 24 used here) and pnpm.

## Content

- `content/docs/**/*.mdx` — all page content. The sidebar follows the file tree (and
  `meta.json` files for ordering).
- `components/` — the design system and the interactive React/SVG visualizers.
- `data/facts.json` — canonical numbers/API surface emitted from the Python code by
  `../scripts/docs_facts.py`; data-driven tables read from it so they cannot drift.
- `lib/`, `app/` — Fumadocs source adapter, layouts, search, OG-image generation.

## Deploy

The site is a standard **Next.js** app with this folder (`website/`) as its root; `pnpm build`
produces a production build deployable on any Next.js host. It is served at
**https://pragmatiq.getdynamiq.ai**.

Open Graph/Twitter cards resolve to that domain in production by default; set
`NEXT_PUBLIC_SITE_URL` to override.

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
