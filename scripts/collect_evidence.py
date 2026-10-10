#!/usr/bin/env python3
"""Preserve writer facts before removing .work (stdlib only).

Usage: collect_evidence.py DOCS_DIR --commit SHA --example example.json
       [--source-url URL] [--scope TEXT] [--visuals visuals.json]
The example JSON records configuration, values, assumptions and evidence status.
This consolidates evidence; it does not independently verify code claims.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('docs_dir', type=Path)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--example', type=Path, required=True)
    ap.add_argument('--source-url', default='')
    ap.add_argument('--scope', default='')
    ap.add_argument('--visuals', type=Path)
    args = ap.parse_args(argv)
    try:
        if not re.fullmatch(r'[0-9a-f]{7,40}|unversioned', args.commit):
            raise ValueError('--commit must be a Git SHA or unversioned')
        facts_files = sorted((args.docs_dir / '.work').glob('*.facts.json'))
        if not facts_files:
            raise ValueError('no .work/*.facts.json; retain existing evidence rather than replacing it with an empty ledger')
        example = json.loads(args.example.read_text())
        if not isinstance(example, dict) or example.get('evidence_status') not in ('illustrative', 'fixture', 'measured'):
            raise ValueError('example needs evidence_status: illustrative, fixture, or measured')
        for key in ('values', 'config', 'assumptions'):
            if key not in example:
                raise ValueError(f'example needs {key}')
        subsystems, seen = [], set()
        for file in facts_files:
            facts = json.loads(file.read_text())
            if not isinstance(facts, dict) or not isinstance(facts.get('steps'), list):
                raise ValueError(f'{file.name}: facts needs a steps array')
            for step in facts['steps']:
                if not isinstance(step, dict) or not isinstance(step.get('id'), str) or not step['id']:
                    raise ValueError(f'{file.name}: every step needs a nonempty id')
                if step['id'] in seen:
                    raise ValueError(f'duplicate step ID: {step["id"]}')
                seen.add(step['id'])
            doc = facts.get('doc', file.name.removesuffix('.facts.json') + '.md')
            if not isinstance(doc, str) or Path(doc).name != doc or not doc.endswith('.md') or not (args.docs_dir/doc).is_file():
                raise ValueError(f'{file.name}: doc must name an existing Markdown file; set facts.doc when its name differs')
            subsystems.append({'doc': doc, 'facts': facts})
        visuals = json.loads(args.visuals.read_text()) if args.visuals else []
        if not isinstance(visuals, list):
            raise ValueError('visuals JSON must be an array of visual briefs')
        for visual in visuals:
            if not isinstance(visual, dict):
                raise ValueError('each visual brief must be an object')
            doc = visual.get('doc')
            if doc and not (args.docs_dir/doc).is_file():
                raise ValueError(f'visual brief names a missing document: {doc}')
        ledger = {'schema_version': 1,
                  'source': {'url': args.source_url, 'commit': args.commit, 'scope': args.scope},
                  'running_example': example, 'subsystems': subsystems, 'visuals': visuals}
        # Reject workstation paths in a shareable ledger; use repo-relative paths.
        encoded = json.dumps(ledger, indent=2, ensure_ascii=False) + '\n'
        if re.search(r'/(?:Users|home|private/tmp|var/folders)/', encoded):
            raise ValueError('evidence contains workstation paths; replace them with repo-relative paths')
        out = args.docs_dir / 'evidence.json'
        temporary = out.with_suffix('.json.tmp')
        temporary.write_text(encoded)
        temporary.replace(out)
        print(f'wrote {out}: {len(subsystems)} subsystems, {len(seen)} steps; claims still require source review')
        return 0
    except (ValueError, OSError, TypeError) as e:
        print(f'ERROR {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
