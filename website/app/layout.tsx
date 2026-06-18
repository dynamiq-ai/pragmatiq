import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { siteUrl } from '@/lib/shared';

const inter = Inter({
  subsets: ['latin'],
});

const DESCRIPTION =
  'An open, reproducible implementation of the PRAGMA recipe for banking foundation ' +
  'models: key–value–time tokenization, a padding-free encoder stack, training, ' +
  'evaluation, ONNX/Triton serving, and an AML transfer-graph extension. Built for ML ' +
  'engineers and data scientists replicating it on their own data.';

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: 'pragmatiq — open banking foundation-model stack',
    template: '%s · pragmatiq',
  },
  description: DESCRIPTION,
  keywords: [
    'pragmatiq', 'PRAGMA', 'banking foundation model', 'event-sequence model',
    'transformer', 'masked language model', 'TimeRoPE', 'fraud detection',
    'credit risk', 'AML', 'GraphSAGE', 'synthetic data', 'PyTorch', 'embeddings',
  ],
  authors: [{ name: 'Dynamiq', url: 'https://getdynamiq.ai' }],
  creator: 'Dynamiq',
  alternates: { canonical: '/' },
  openGraph: {
    type: 'website',
    siteName: 'pragmatiq',
    url: siteUrl,
    title: 'pragmatiq — open banking foundation-model stack',
    description: DESCRIPTION,
  },
  twitter: { card: 'summary_large_image', title: 'pragmatiq', description: DESCRIPTION },
};

// Structured data (JSON-LD) for search engines + LLM grounding: who made it, what it
// is, where the code/docs live, and the license.
const jsonLd = {
  '@context': 'https://schema.org',
  '@graph': [
    {
      '@type': 'Organization',
      '@id': 'https://getdynamiq.ai/#org',
      name: 'Dynamiq',
      url: 'https://getdynamiq.ai',
    },
    {
      '@type': 'WebSite',
      '@id': `${siteUrl}/#website`,
      url: siteUrl,
      name: 'pragmatiq documentation',
      description: DESCRIPTION,
      publisher: { '@id': 'https://getdynamiq.ai/#org' },
    },
    {
      '@type': 'SoftwareSourceCode',
      name: 'pragmatiq',
      description: DESCRIPTION,
      codeRepository: 'https://github.com/dynamiq-ai/pragmatiq',
      programmingLanguage: 'Python',
      license: 'https://www.apache.org/licenses/LICENSE-2.0',
      author: { '@id': 'https://getdynamiq.ai/#org' },
    },
  ],
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        {/* Default to the sharp black dark theme; the toggle still switches to light. */}
        <RootProvider theme={{ defaultTheme: 'dark', enableSystem: false }}>
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
