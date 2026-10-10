#!/usr/bin/env python3
"""Stage a portable Jekyll post and assets without modifying the target site."""
import argparse
import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

IMAGES={'.svg','.png','.jpg','.jpeg','.webp','.gif','.avif'}
VIDEOS={'.mp4','.webm','.ogg','.mp3','.wav'}
MEDIA=IMAGES|VIDEOS|{'.html','.htm','.json','.csv','.txt','.log','.py','.yaml','.yml','.ipynb','.zip'}
LINK=re.compile(r'(!?\[[^\]]*\]\(\s*)<?([^\s)>]+)>?(\s*(?:"[^"]*")?\))')
TAG=re.compile(r'<(?:img|video|audio|source|iframe|a)\b[^>]*>',re.I|re.S)
ATTR=re.compile(r'(\b(?:src|poster|href)\s*=\s*)(["\x27])(.*?)\2',re.I|re.S)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('article',type=Path)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--slug',required=True)
    p.add_argument('--date',required=True)
    p.add_argument('--title'); p.add_argument('--subtitle')
    p.add_argument('--categories',nargs='+')
    p.add_argument('--site',type=Path,help='Existing site, checked read-only for layout availability')
    p.add_argument('--out',type=Path,required=True,help='New staging directory')
    args=p.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',args.slug): p.error('slug must use lowercase words and hyphens')
    try: datetime.date.fromisoformat(args.date)
    except ValueError: p.error('date must be YYYY-MM-DD')
    if args.out.exists(): p.error('staging directory already exists; use a new --out')
    try:
        profile=json.loads(args.profile.read_text()); text=args.article.read_text()
        front=dict(profile['front_matter'])
        if args.site and not (args.site/'_layouts'/str(front['layout']+'.html')).is_file():
            raise ValueError('profile layout does not exist in target site')
        if text.startswith('---\n'): raise ValueError('use platform-neutral Markdown without existing front matter')
        h1=re.search(r'^#\s+(.+)$',text,re.M)
        front['title']=args.title or (h1.group(1) if h1 else args.slug.replace('-',' ').title())
        if args.subtitle: front['subtitle']=args.subtitle
        if args.categories: front['categories']=args.categories
        if profile.get('strip_first_heading') and h1:
            text=text[:h1.start()]+text[h1.end():]
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='jekyll-stage-',dir=args.out.parent) as temp:
            stage=Path(temp)/'site'; stage.mkdir()
            assets={}
            def asset(url):
                parts=urlsplit(html.unescape(url))
                if parts.scheme or url.startswith(('#','//','{{','{%')): return url
                path=(args.article.parent/unquote(parts.path)).resolve()
                if path.suffix.lower() not in MEDIA:
                    if path.suffix.lower()=='.md':
                        raise ValueError(f'local Markdown cross-link needs a published URL/post_url: {url}')
                    return url
                if not path.is_file(): raise ValueError(f'missing asset: {url}')
                if path not in assets:
                    if path.suffix.lower() in {'.html','.htm'}:
                        # Players are deliberately standalone: don't silently ship broken trees.
                        refs=re.findall(r'\b(?:src|href)=["\x27]([^"\x27]+)',path.read_text(),re.I)
                        if any(not urlsplit(r).scheme and not r.startswith(('#','//','data:')) for r in refs):
                            raise ValueError(f'HTML player must be standalone: {url}')
                    kind='image' if path.suffix.lower() in IMAGES else 'video' if path.suffix.lower() in VIDEOS else 'interactive' if path.suffix.lower() in {'.html','.htm'} else 'data'
                    root=Path(profile['asset_roots'][kind])
                    if root.is_absolute() or '..' in root.parts: raise ValueError('asset_roots must be site-relative')
                    name=re.sub(r'[^\w.-]','-',path.stem)[:60]+'-'+hashlib.sha256(path.read_bytes()).hexdigest()[:12]+path.suffix.lower()
                    rel=root/args.slug/name
                    dest=stage/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dest)
                    assets[path]='/'+rel.as_posix()
                return "{{ '"+assets[path]+"' | relative_url }}"+('#'+parts.fragment if parts.fragment else '')
            def link(m): return m.group(1)+asset(m.group(2))+m.group(3)
            def tag(m):
                return ATTR.sub(lambda a:a.group(1)+a.group(2)+asset(a.group(3))+a.group(2),m.group(0))
            lines=text.splitlines(keepends=True); output=[]; fence=None; pending=[]
            def flush():
                prose=''.join(pending); pending.clear()
                prose=re.sub(r'^\$\$\s*\n([\s\S]*?)\n\$\$\s*$',lambda m:'<div class="math-block">\n$$'+m.group(1)+'$$\n</div>',prose,flags=re.M)
                chunks=re.split(r'(`+[^`]*`+)',prose)
                for i,chunk in enumerate(chunks):
                    if i%2:
                        if '{{' in chunk or '{%' in chunk: chunks[i]='{% raw %}'+chunk+'{% endraw %}'
                    else: chunks[i]=TAG.sub(tag,LINK.sub(link,chunk))
                output.append(''.join(chunks))
            for line in lines:
                marker=re.match(r'^\s*(`{3,}|~{3,})',line)
                if marker:
                    if fence is None:
                        flush()
                        fence=marker.group(1)[0]; output.append('{% raw %}\n'+line)
                    elif marker.group(1)[0]==fence:
                        fence=None; output.append(line+'{% endraw %}\n')
                    else: output.append(line)
                    continue
                if fence: output.append(line); continue
                pending.append(line)
            if fence: raise ValueError('unclosed Markdown code fence')
            flush()
            body=''.join(output).strip()+'\n'
            selector=profile.get('content_selector')
            if selector:
                # Override theme media limits within the selected article container.
                # Keep full worked figures readable and video within the article width.
                body+='\n<style>\n'+selector+' video {display:block;width:100%;height:auto;}\n'+selector+' iframe {width:100%;max-height:none;border:0;}\n'+selector+' img:not(.emoji) {max-width:100%;height:auto;max-height:none;}\n'+selector+' figure {margin:2rem 0;}\n'+selector+' figcaption {font-size:.9em;line-height:1.5;margin-top:.6rem;}\n</style>\n'
            if profile.get('math')=='katex':
                body+='\n<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">\n<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>\n<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.querySelector(\'.post-content\') || document.querySelector(\'.post\'), {delimiters:[{left:\'$$\',right:\'$$\',display:true},{left:\'$\',right:\'$\',display:false}],throwOnError:false})"></script>\n'
            post=stage/'_posts'/f'{args.date}-{args.slug}.md'; post.parent.mkdir()
            post.write_text('---\n'+'\n'.join(k+': '+json.dumps(v,ensure_ascii=False) for k,v in front.items())+'\n---\n\n'+body)
            (stage/'asset-manifest.json').write_text(json.dumps({'post':post.relative_to(stage).as_posix(),'assets':list(assets.values()),'profile':profile.get('name'),'publication':'staged locally'},indent=2)+'\n')
            shutil.move(stage,args.out)
        print(f'Staged {post.name} and {len(assets)} assets in {args.out}')
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.error(str(exc))


if __name__=='__main__': raise SystemExit(main())
