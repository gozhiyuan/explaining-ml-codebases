#!/usr/bin/env python3
"""Render one article using the packaged/shared HTML runtime (stdlib only)."""
import argparse
import runpy
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('article', type=Path)
    ap.add_argument('-o', '--out', type=Path)
    ap.add_argument('--media-mode', choices=['auto', 'inline', 'bundle'], default='auto')
    ap.add_argument('--max-inline-mb', type=float, default=20)
    ap.add_argument('--link-base')
    args = ap.parse_args(argv)
    here = Path(__file__).resolve()
    candidates = [here.parent / 'render_html.py', here.parents[3] / 'scripts/render_html.py']
    runtime = next((p for p in candidates if p.is_file()), None)
    if runtime is None:
        print('ERROR shared renderer missing; use the repository install_skills.py helper', file=sys.stderr)
        return 2
    article = args.article.resolve()
    command = [str(article.parent), '--entry', article.name, '--layout', 'article',
               '-o', str(args.out or article.with_suffix('.html')), '--media-mode', args.media_mode,
               '--max-inline-mb', str(args.max_inline_mb)]
    if args.link_base:
        command += ['--link-base', args.link_base]
    return runpy.run_path(str(runtime))['main'](command)


if __name__ == '__main__':
    sys.exit(main())
