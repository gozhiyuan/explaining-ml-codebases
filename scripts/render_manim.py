#!/usr/bin/env python3
"""Render an original scene, its final poster, and numerical provenance."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from animation_math import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('spec',type=Path)
    p.add_argument('-o','--output',type=Path,required=True)
    p.add_argument('--quality',choices=['low','medium','high'],default='medium')
    p.add_argument('--python',default=sys.executable,help='Python environment containing Manim')
    args=p.parse_args()
    out=args.output.resolve()
    poster=out.with_suffix('.poster.png'); metadata=out.with_suffix('.metadata.json')
    spec_copy=out.with_suffix('.scene.json'); scene_copy=out.with_suffix('.scene.py'); math_copy=out.parent/'animation_math.py'
    source_bundle=out.with_suffix('.sources.zip')
    targets=[out,poster,metadata,spec_copy,scene_copy,source_bundle]
    if out.suffix.lower() != '.mp4' or any(x.exists() for x in targets):
        p.error('output must be a new .mp4; video/poster/metadata/source files must not exist')
    try:
        spec=json.loads(args.spec.read_text())
        values=validate(spec)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.error(str(exc))
    if not shutil.which('ffmpeg'):
        p.error('FFmpeg is needed for poster extraction; keep static figures when unavailable')
    root=Path(__file__).resolve().parent
    scene={'attention':'AttentionComputation','token-flow':'TokenFlow','cache-reuse':'CacheReuse'}[spec['template']]
    env=os.environ.copy(); env['ML_EXPLAIN_SCENE_SPEC']=str(args.spec.resolve())
    env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONDONTWRITEBYTECODE']='1'
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='ml-explain-manim-') as tmp:
            cmd=[args.python,'-m','manim',str(root/'manim_scenes.py'),scene,'-q'+{'low':'l','medium':'m','high':'h'}[args.quality],'--media_dir',tmp,'--disable_caching','-o','clip']
            subprocess.run(cmd,env=env,check=True)
            clips=list(Path(tmp).rglob('clip.mp4'))
            if len(clips)!=1:
                raise ValueError('Manim did not produce exactly one final clip')
            scratch=Path(tmp)/'poster.png'
            subprocess.run(['ffmpeg','-v','error','-sseof','-0.2','-i',str(clips[0]),'-frames:v','1','-update','1',str(scratch)],check=True)
            if not scratch.exists():
                raise ValueError('poster extraction failed')
            shutil.copyfile(clips[0],out); shutil.copyfile(scratch,poster)
        spec_copy.write_text(json.dumps(spec,indent=2)+'\n')
        shutil.copyfile(root/'manim_scenes.py',scene_copy)
        if not math_copy.exists():
            shutil.copyfile(root/'animation_math.py',math_copy)
        elif math_copy.read_bytes() != (root/'animation_math.py').read_bytes():
            raise ValueError('existing animation_math.py differs; use a separate output folder')
        metadata.write_text(json.dumps({'template':spec['template'],'values':values,'spec':spec_copy.name,'scene':scene_copy.name,'source_bundle':source_bundle.name,'timing':'illustrative, not a measurement','command': ['python','render_manim.py',spec_copy.name,'-o',out.name,'--quality',args.quality]},indent=2)+'\n')
        with zipfile.ZipFile(source_bundle,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('render_manim.py','manim_scenes.py','animation_math.py'):
                z.write(root/name,name)
            z.writestr('spec.json',json.dumps(spec,indent=2)+'\n')
            license_file=next((p for p in (root.parent/'LICENSE',root/'LICENSE') if p.is_file()),None)
            if license_file is None:
                raise ValueError('scene source bundle needs the distributed LICENSE file')
            z.write(license_file,'LICENSE')
            z.writestr('README.md','Original editable scene sources. Requires Manim Community, FFmpeg, and LaTeX for attention.\n\nRender with: python render_manim.py spec.json -o new.mp4 --python /path/to/manim-env/bin/python\nPresentation timing is illustrative.\n')
    except (subprocess.CalledProcessError,OSError,ValueError) as exc:
        for target in targets:
            target.unlink(missing_ok=True)
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1
    print(f'Rendered {out.name}, {poster.name}, metadata and editable scene sources')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
