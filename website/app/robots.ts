import type { MetadataRoute } from 'next';
import { siteUrl } from '@/lib/shared';

// Allow all crawlers (search engines + LLM/AI crawlers for GEO visibility) and point
// them at the sitemap. llms.txt / llms-full.txt are also served for AI grounding.
export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: '*', allow: '/' }],
    sitemap: `${siteUrl}/sitemap.xml`,
    host: siteUrl,
  };
}
