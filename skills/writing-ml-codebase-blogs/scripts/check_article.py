#!/usr/bin/env python3
"""Check provenance, asset paths, source references and retained uncertainty.

Structural checks do not establish that statements describe source correctly.
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


def symbols(tree, prefix=''):
    result = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = prefix + node.name
            result.add(name)
            result.update(symbols(node, name + '.'))
        else:
            result.update(symbols(node, prefix))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('article', type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--evidence', type=Path)
    args = ap.parse_args(argv)
    errors = []
    try:
        text = args.article.read_text()
        manifest = json.loads(args.manifest.read_text())
        for key in ('title', 'question', 'audience', 'scope'):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                errors.append(f'manifest needs {key}')
        repo = args.repo.resolve()
        commit = manifest.get('source', {}).get('commit', '')
        if not re.fullmatch('[0-9a-f]{7,40}', commit):
            errors.append('source.commit must be a Git SHA')
        else:
            actual = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
            if actual.returncode or not actual.stdout.strip().startswith(commit):
                errors.append('source commit does not match checkout')
        if args.evidence:
            evidence = json.loads(args.evidence.read_text())
            if evidence.get('source', {}).get('commit') != commit:
                errors.append('input evidence and article commits differ')
            retained = json.dumps(manifest.get('uncertainties', []))
            for subsystem in evidence.get('subsystems', []):
                for unknown in subsystem.get('facts', {}).get('unverified', []):
                    if str(unknown) not in retained:
                        errors.append(f'input uncertainty not retained: {unknown}')
        claims = manifest.get('claims')
        if not isinstance(claims, list) or not claims:
            errors.append('manifest needs nonempty claims'); claims = []
        seen = set()
        for claim in claims:
            identity = claim.get('id')
            if not identity or identity in seen:
                errors.append(f'empty/duplicate claim id: {identity}')
            seen.add(identity)
            status = claim.get('evidence_status')
            if status not in ('source-reviewed', 'fixture', 'measured', 'unverified'):
                errors.append(f'{identity}: invalid evidence_status')
            if status == 'measured' and not (claim.get('command') and claim.get('result')):
                errors.append(f'{identity}: measured claim needs command/result')
            if status == 'unverified' and not (claim.get('reason') and 'unverified' in text.lower()):
                errors.append(f'{identity}: qualify uncertainty in manifest/prose')
            if not claim.get('sources'):
                errors.append(f'{identity}: no source references')
            for reference in claim.get('sources', []):
                rel = Path(reference.get('file', ''))
                file = (repo / rel).resolve()
                if rel.is_absolute() or not file.is_relative_to(repo) or not file.is_file():
                    errors.append(f'{identity}: missing/nonportable source {rel}'); continue
                symbol = reference.get('symbol', '')
                content = file.read_text(errors='replace')
                if file.suffix == '.py':
                    try: found = symbols(ast.parse(content))
                    except SyntaxError: found = set()
                    exists = symbol == '<module>' or symbol in found or (symbol.isidentifier() and re.search(r'\b' + re.escape(symbol) + r'\b', content))
                else:
                    exists = bool(symbol and symbol in content)
                if not exists:
                    errors.append(f'{identity}: symbol missing: {rel}::{symbol}')
        for figure in manifest.get('figures', []):
            if not (args.article.parent / figure.get('asset', '')).is_file() or not figure.get('caption'):
                errors.append(f'figure needs existing asset/caption: {figure.get("id")}')
            if not set(figure.get('claim_ids', [])) <= seen:
                errors.append(f'figure references unknown claims: {figure.get("id")}')
        prose = re.sub(r'```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`]*`', '', text)
        targets = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', prose)
        targets += re.findall(r'<(?:img|video|source|iframe)\b[^>]*?\b(?:src|poster)=["\x27]([^"\x27]+)', prose, re.I)
        for target in targets:
            url = urlsplit(target.strip().strip('<>'))
            if not url.scheme and not target.startswith(('//', '#')) and not (args.article.parent / unquote(url.path)).is_file():
                errors.append(f'local asset missing: {target}')
        if re.search(r'/(?:Users|home|private/tmp|var/folders)/', text):
            errors.append('article contains workstation paths')
        for error in errors: print('ERROR ' + error)
        print(f'{len(errors)} errors; {len(claims)} claim references checked; semantic review still required')
        return bool(errors)
    except (OSError, ValueError, TypeError, AttributeError) as e:
        print(f'ERROR {e}', file=sys.stderr); return 1


if __name__ == '__main__':
    sys.exit(main())
