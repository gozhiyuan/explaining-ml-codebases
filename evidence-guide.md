# Persistent explanation evidence

Preserve evidence so later doc refreshes, visual generation, and a separate
blog-writing skill can reuse verified work. Markdown remains the explanation;
`evidence.json` is its source ledger, not an independent authority.

Before deleting `.work/`, update facts after source review and test mapping.
Keep repository-relative paths and defining symbols. For each important state
transition retain input, output, caller/owner, condition, and supporting tests
in the existing step records. Internal line numbers may help review, but use
files and symbols in reader-facing citations. Keep uncertainty and test gaps.
Facts default to the matching Markdown filename (`train.facts.json` →
`train.md`). If they differ, set `"doc": "scheduler.md"` in the facts object.
The collector rejects a missing mapped document rather than preserve a broken
handoff to the article writer.

Write `.work/example.json` with:

```json
{
  "evidence_status": "illustrative",
  "config": {"block_size": 4},
  "values": {"request_A": [1, 2, 3, 4, 5]},
  "assumptions": ["Toy token IDs; no eviction during this trace"],
  "seed": null,
  "provenance": "Constructed example; transitions checked against the recorded source"
}
```

Use `fixture` for identified test-fixture values and `measured` only for actual
runs; add the exact command, environment and results. Keep configuration and
values separate. Where a trace mixes evidence kinds, label individual values
or transitions and retain those qualifications in prose and captions.

Then run:

```bash
python "$SKILL_DIR/scripts/collect_evidence.py" docs/explain \
  --commit <explained-sha> --source-url <repository-url> --scope <scope> \
  --example docs/explain/.work/example.json --visuals docs/explain/.work/visuals.json
```

Omit `--visuals` when no visuals were planned. Use `--commit unversioned` for
code without Git history and explain that limitation. The script checks basic
structure, duplicate step IDs, and common workstation paths; it does not check
semantic correctness. It preserves facts per subsystem to avoid flattening
away ownership or provenance. It writes `schema_version`, `source`,
`running_example`, `subsystems` and `visuals` to `evidence.json`.

After review, each visual brief should name its delivered assets, step IDs,
assumptions, source symbols and evidence status (see `visual-guide.md`). Link
`evidence.json` from the index's provenance or verification section; retain
editable visual sources under `assets/`. Only then clean the temporary drafts.

On revision, update evidence for affected subsystems and visuals. Preserve
unchanged records; do not replace the ledger with an empty one when `.work/`
is absent. Use the same format when writing facts sequentially without
subagents. A downstream writer should reopen source for new or disputed
claims and retain the explained commit. This package does not publish articles
or apply a personal site's formatting.
