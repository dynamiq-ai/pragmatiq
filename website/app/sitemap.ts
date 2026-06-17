import type { MetadataRoute } from 'next';
import { source } from '@/lib/source';
import { siteUrl } from '@/lib/shared';

// Home + every docs page, so search engines index the whole site.
export default function sitemap(): MetadataRoute.Sitemap {
  const pages = source.getPages().map((page) => ({
    url: `${siteUrl}${page.url}`,
    changeFrequency: 'weekly' as const,
    priority: 0.8,
  }));
  return [
    { url: siteUrl, changeFrequency: 'weekly', priority: 1 },
    ...pages,
  ];
}
