export const appName = 'pragmatiq';

// Canonical site URL for metadata, sitemap, robots, and structured data.
// NEXT_PUBLIC_SITE_URL overrides; production defaults to the custom domain.
export const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.NODE_ENV === 'production'
    ? 'https://pragmatiq.getdynamiq.ai'
    : 'http://localhost:3000');

export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

// Public repository the docs site documents.
export const gitConfig = {
  user: 'dynamiq-ai',
  repo: 'pragmatiq',
  branch: 'main',
};

// Independent-implementation attribution required across the project (verbatim).
export const attribution =
  'pragmatiq is an independent implementation inspired by the PRAGMA paper ' +
  '(arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.';
