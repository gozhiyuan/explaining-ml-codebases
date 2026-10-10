#!/usr/bin/env python3
"""Package independently usable skills from this checkout into a new destination."""
import argparse
from pathlib import Path
import shutil
import tempfile

CORE='explaining-ml-codebases'
BLOG='writing-ml-codebase-blogs'
IGNORE=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store','.git')
SHARED=['render_html.py','render_trace.py','export_trace_video.mjs','render_manim.py','manim_scenes.py','animation_math.py']


def package(root,stage,selection):
    if selection in {'core','all'}:
        out=stage/CORE; out.mkdir()
        for file in root.glob('*.md'): shutil.copyfile(file,out/file.name)
        for name in ['scripts','assets','agents']:
            shutil.copytree(root/name,out/name,ignore=IGNORE)
        for name in ['LICENSE','THIRD_PARTY_NOTICES.md']:
            shutil.copyfile(root/name,out/name)
    if selection in {'blog','all'}:
        out=stage/BLOG
        shutil.copytree(root/'skills'/BLOG,out,ignore=IGNORE)
        for name in SHARED: shutil.copyfile(root/'scripts'/name,out/'scripts'/name)
        shutil.copytree(root/'assets',out/'assets',dirs_exist_ok=True,ignore=IGNORE)
        for name in ['LICENSE','THIRD_PARTY_NOTICES.md']:
            shutil.copyfile(root/name,out/name)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest',type=Path,help='Agent skills directory; refuses to replace installed skills')
    p.add_argument('--archive',type=Path,help='New .zip containing installable skill folders')
    p.add_argument('--skill',choices=['core','blog','all'],default='core')
    args=p.parse_args(argv)
    if bool(args.dest)==bool(args.archive): p.error('choose exactly one of --dest or --archive')
    names=[CORE,BLOG] if args.skill=='all' else [CORE if args.skill=='core' else BLOG]
    if args.dest and any((args.dest/name).exists() for name in names):
        p.error('an installed skill already exists; select a new destination or move its folder before updating')
    if args.archive and (args.archive.exists() or args.archive.suffix!='.zip'):
        p.error('archive must be a new .zip')
    root=Path(__file__).resolve().parents[1]
    try:
        with tempfile.TemporaryDirectory(prefix='ml-skills-package-') as temp:
            stage=Path(temp)/'skills'; stage.mkdir(); package(root,stage,args.skill)
            if args.archive:
                args.archive.parent.mkdir(parents=True,exist_ok=True)
                archive=shutil.make_archive(str(Path(temp)/'release'),'zip',stage)
                shutil.move(archive,args.archive)
                print(f'Packaged {", ".join(names)} into {args.archive}')
            else:
                args.dest.mkdir(parents=True,exist_ok=True)
                moved=[]
                try:
                    for name in names:
                        shutil.move(stage/name,args.dest/name); moved.append(args.dest/name)
                except OSError:
                    for path in moved: shutil.rmtree(path)
                    raise
                print(f'Installed {", ".join(names)} into {args.dest}')
        return 0
    except OSError as exc: p.error(str(exc))


if __name__=='__main__': raise SystemExit(main())
