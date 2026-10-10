#!/usr/bin/env python3
"""Revalidate recorded public examples in a fresh output folder; does not call an agent."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nanogpt',type=Path,required=True)
    p.add_argument('--vllm',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--python',default=sys.executable,help='Python with torch and numpy for nanoGPT')
    args=p.parse_args()
    args.out=args.out.resolve()
    if args.out.exists(): p.error('--out must be a new directory')
    root=Path(__file__).resolve().parents[1]
    cases=json.loads((root/'evals/cases.json').read_text())['cases']
    repos={c['id']:getattr(args,c['id']).resolve() for c in cases}
    for case in cases:
        repo=repos[case['id']]
        head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
        status=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True)
        if head.returncode or head.stdout.strip()!=case['commit'] or status.returncode or status.stdout:
            p.error(f'{case["id"]}: need a clean checkout at {case["commit"]}')
    args.out.mkdir(parents=True)
    report=[]
    def run(command,cwd):
        result=subprocess.run(list(map(str,command)),cwd=cwd,capture_output=True,text=True)
        identity=f'{len(report):02d}'
        (args.out/f'check-{identity}.log').write_text(result.stdout+result.stderr)
        report.append({'check':identity,'command':[str(x).replace(str(args.out.resolve()),'OUTPUT').replace(str(root),'SKILL_REPO').replace(str(repos['nanogpt']),'NANOGPT_REPO').replace(str(repos['vllm']),'VLLM_REPO') for x in command], 'status':'pass' if result.returncode==0 else 'fail','returncode':result.returncode})
        if result.returncode: raise RuntimeError(f'check {identity} failed; see its log')
    try:
        for case in cases:
            name=case['id']; dest=args.out/name
            shutil.copytree(root/'examples'/name,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            if name=='nanogpt':
                run([args.python,dest/'tests/validate_step.py','--repo',repos[name],'--output',dest/'assets/measurement.json'],dest)
                shutil.copyfile(dest/'assets/measurement.json',dest/'blog/assets/measurement.json')
            else:
                run([args.python,dest/'tests/source_contracts.py',repos[name]],dest)
                shutil.copyfile(dest/'tests/source-contract-results.json',dest/'blog/assets/source-contract-results.json')
            run([sys.executable,root/'scripts/check_explain_doc.py',dest,'--repo',repos[name]],dest)
            run([sys.executable,root/'skills/writing-ml-codebase-blogs/scripts/check_article.py',dest/'blog/article.md','--manifest',dest/'blog/article.json','--repo',repos[name],'--evidence',dest/'evidence.json'],dest)
            run([sys.executable,root/'scripts/render_html.py',dest,'-o',dest/'explain.html'],dest)
            run([sys.executable,root/'skills/writing-ml-codebase-blogs/scripts/render_article.py',dest/'blog/article.md','-o',dest/'blog/article.html'],dest)
        print('Passed CPU fixtures, doc/article structure and regenerated HTML for both public cases')
        return 0
    except RuntimeError as exc:
        print(str(exc),file=sys.stderr); return 1
    finally:
        (args.out/'replay-report.json').write_text(json.dumps({'checks':report,'boundary':'Revalidation of recorded examples; not fresh agent generation, upstream GPU tests, browser validation or site publication'},indent=2)+'\n')


if __name__=='__main__': raise SystemExit(main())
