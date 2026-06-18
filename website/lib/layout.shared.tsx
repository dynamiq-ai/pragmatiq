import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import { appName, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="inline-flex items-center gap-2 font-semibold">
          <span className="flex size-6 items-center justify-center rounded-[6px] bg-white p-0.5 ring-1 ring-black/5">
            <Image src="/brand/mark.png" alt="" width={20} height={20} />
          </span>
          {appName}
        </span>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
    links: [{ text: 'Built by Dynamiq', url: 'https://getdynamiq.ai', external: true }],
  };
}
