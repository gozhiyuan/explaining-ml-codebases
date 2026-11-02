#!/usr/bin/env python3
"""Preserve writer facts before removing .work (stdlib only).

Usage: collect_evidence.py DOCS_DIR --commit SHA --example example.json
       [--source-url URL] [--scope TEXT] [--visuals visuals.json]
The example JSON records configuration, values, assumptions and evidence status.
This consolidates evidence; it does not independently verify code claims.

Revise mode: when DOCS_DIR/evidence.json already exists, new facts replace the
stored subsystem for the same doc, other subsystems are kept, and visual briefs
merge by id. --example is optional then, and the stored example is kept. With
no .work/*.facts.json, the stored subsystems are kept unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _check_example(example):
    if not isinstance(example, dict) or example.get("evidence_status") not in (
        "illustrative",
        "fixture",
        "measured",
    ):
        raise ValueError(
            "example needs evidence_status: illustrative, fixture, or measured"
        )
    for key in ("values", "config", "assumptions"):
        if key not in example:
            raise ValueError(f"example needs {key}")


def _add_step_ids(facts, label, seen):
    for step in facts.get("steps", []):
        if (
            not isinstance(step, dict)
            or not isinstance(step.get("id"), str)
            or not step["id"]
        ):
            raise ValueError(f"{label}: every step needs a nonempty id")
        if step["id"] in seen:
            raise ValueError(f"duplicate step ID: {step['id']}")
        seen.add(step["id"])


def main(argv=None):  # noqa: C901, PLR0912
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("docs_dir", type=Path)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--example", type=Path)
    ap.add_argument("--source-url", default="")
    ap.add_argument("--scope", default="")
    ap.add_argument("--visuals", type=Path)
    args = ap.parse_args(argv)
    try:
        if not re.fullmatch(r"[0-9a-f]{7,40}|unversioned", args.commit):
            raise ValueError("--commit must be a Git SHA or unversioned")
        out = args.docs_dir / "evidence.json"
        existing = json.loads(out.read_text()) if out.is_file() else None
        if existing is not None and not isinstance(existing, dict):
            raise ValueError("existing evidence.json must be a JSON object")
        facts_files = sorted((args.docs_dir / ".work").glob("*.facts.json"))
        if not facts_files and existing is None:
            raise ValueError(
                "no .work/*.facts.json and no evidence.json; nothing to collect"
            )
        if args.example:
            example = json.loads(args.example.read_text())
        elif existing is not None and "running_example" in existing:
            example = existing["running_example"]
        else:
            raise ValueError(
                "--example is required when evidence.json has no running_example"
            )
        _check_example(example)

        subsystems, seen = [], set()
        for file in facts_files:
            facts = json.loads(file.read_text())
            if not isinstance(facts, dict) or not isinstance(facts.get("steps"), list):
                raise ValueError(f"{file.name}: facts needs a steps array")
            _add_step_ids(facts, file.name, seen)
            doc = facts.get("doc", file.name.removesuffix(".facts.json") + ".md")
            if (
                not isinstance(doc, str)
                or Path(doc).name != doc
                or not doc.endswith(".md")
                or not (args.docs_dir / doc).is_file()
            ):
                raise ValueError(
                    f"{file.name}: doc must name an existing Markdown file; set facts.doc when its name differs"
                )
            subsystems.append({"doc": doc, "facts": facts})
        if existing is not None:
            fresh_docs = {s["doc"] for s in subsystems}
            kept = [
                s
                for s in existing.get("subsystems", [])
                if isinstance(s, dict) and s.get("doc") not in fresh_docs
            ]
            for s in kept:
                _add_step_ids(s.get("facts", {}), f"evidence.json {s.get('doc')}", seen)
            subsystems = kept + subsystems

        visuals = json.loads(args.visuals.read_text()) if args.visuals else []
        if not isinstance(visuals, list):
            raise ValueError("visuals JSON must be an array of visual briefs")
        if existing is not None:
            fresh_ids = {v.get("id") for v in visuals if isinstance(v, dict)}
            visuals = [
                v
                for v in existing.get("visuals", [])
                if isinstance(v, dict) and v.get("id") not in fresh_ids
            ] + visuals
        for visual in visuals:
            if not isinstance(visual, dict):
                raise ValueError("each visual brief must be an object")
            doc = visual.get("doc")
            if doc and not (args.docs_dir / doc).is_file():
                raise ValueError(f"visual brief names a missing document: {doc}")

        source = dict((existing or {}).get("source", {}))
        source.update({"commit": args.commit})
        if args.source_url or "url" not in source:
            source["url"] = args.source_url
        if args.scope or "scope" not in source:
            source["scope"] = args.scope
        ledger = {
            "schema_version": 1,
            "source": source,
            "running_example": example,
            "subsystems": subsystems,
            "visuals": visuals,
        }
        # Reject workstation paths in a shareable ledger; use repo-relative paths.
        encoded = json.dumps(ledger, indent=2, ensure_ascii=False) + "\n"
        if re.search(r"/(?:Users|home|private/tmp|var/folders)/", encoded):
            raise ValueError(
                "evidence contains workstation paths; replace them with repo-relative paths"
            )
        temporary = out.with_suffix(".json.tmp")
        temporary.write_text(encoded)
        temporary.replace(out)
        mode = "merged into existing ledger" if existing is not None else "new ledger"
        print(
            f"wrote {out} ({mode}): {len(subsystems)} subsystems, {len(seen)} steps; claims still require source review"
        )
        return 0
    except (ValueError, OSError, TypeError) as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
