# Subagent prompts

Subagents do not see your conversation. Each prompt must stand on its own. Fill
every `<placeholder>`. Launch independent writers in one message so they run in
parallel. Respect the available concurrency limit.

| Prompt | Subagent type (Cursor `Task`) | Writes files? |
|--------|-------------------------------|---------------|
| Recon helper (large repos only) | `explore` | no |
| Writer | `generalPurpose` | yes, only under `docs/explain/.work/` |
| Reviser (revise mode) | `generalPurpose` | yes, only its own doc |
| Test mapper | `generalPurpose` | yes, only `.work/tests.md` |
| Fact checker | `generalPurpose` or `explore` | no |

In Claude Code, use the `Agent` tool with the matching built-in type
(`Explore` for read-only, `general-purpose` for writers).

`SKILL_DIR` below is the actual folder containing this skill's `SKILL.md`.
Use the same prompts and facts contract when working sequentially.

---

## Recon helper (large repos only)

Use this when the repo is too large for you to skim the top-level layout
yourself. Run one helper per top-level area, in parallel.

```
You are mapping one area of an ML codebase for a documentation plan. Do not write files.

Repo: <abs path>   Commit: <sha>   Area: <dir or package>
Golden path: <one line, e.g. "online serving of one Llama-family model, TP=1, one request">

Report, naming the file and function or class for each item:
1. The 5-15 most important classes and functions in this area for the golden path, with one line each on what they do.
2. The entry point(s) into this area from the rest of the repo, and what this area calls outside itself.
3. The data that enters and leaves this area (types, schemas, tensor shapes if visible).
4. Test files that cover this area, and how to run them.
5. Branches that exist but are off the golden path (other backends, other models), as a list.
Keep the report under 400 words. Mark anything you inferred without reading the code as UNVERIFIED.
```

---

## Writer

```
You are writing one sub-doc of an explanation of an ML codebase. Read the code. Do not guess.

Repo: <abs path>   Commit: <sha>
Subsystem: <name>   Doc code: <code>   Step-ID prefix: <code>.   (IDs look like <code>.A1, <code>.B3)
Paths in scope: <list of files/dirs>
Entry point: <function or command>   Exit: <artifact or response>
Golden path: <one line>
Mode: <simple|deep>   Audience: <audience>
Running example (use these exact values; do not invent a different one):
<paste the running example>
Visual briefs: <assigned briefs from .work/visuals.json, or "none">.
Seed: <seed>. Every value that depends on random init, dropout, or sampling uses this seed and this config, and the doc states both.
Sibling docs (link to them, do not explain their content): <code: one-line scope, ...>
Do not open (independent mode only; work from the code, not from these docs): <paths, or "none">
Must-cover topics from existing docs (cover each one, or write why it is out of scope; check each against the code, because old docs drift):
<lines from must-cover.txt for this subsystem, or "none">
Path root: <folder>. Put `<!-- explain-root: <folder> -->` and "Paths are relative to `<folder>/`" near the top, then cite paths relative to it.

Before writing, read:
- SKILL_DIR/stage-playbooks.md, only these sections: <list, e.g. "6. Data loading", "7. Model architecture", "8. Training loop">
- SKILL_DIR/doc-templates.md, "Sub-doc template" and "Step section shape"
- SKILL_DIR/writing-modes.md, "Both modes" and "<mode> mode"

Rules:
- Headings carry step IDs: "## <code>.A <group name>", then "### <code>.A1 <step name> (`file`)". Every step ID gets its own ### heading.
- Every step section has Idea, Input, Output, Code, Example, and Tests. Add Math and Under the hood when they apply.
- Point at code by file and function, class, or config key. Never by line number. Write "`make_model` in `training.py`", and `# training.py` in call-tree comments. Use the shortest path that is unique in the repo, or the path root. Read the function before you name it.
- For every step with logic, math, or a non-obvious condition, paste the key 5-25 lines verbatim from the file. Trim with `# ...`. Put `# <file>  <Symbol>` on the first line. Add shape and running-example comments. Then explain the lines in order. Never put pseudo-code in a labeled excerpt.
- Write for a newcomer: what the step does, the code, then what the code does to the running example. Do not stack paths and names in a sentence.
- Show every step's output as the object itself, using the formats in writing-modes.md "Show the data, not only the code": the type, the shape or schema change (new columns with dtypes), where it lives (memory, local disk, S3), and a small sample from the running example (rows, an array slice, trimmed JSON, a file listing). Get real values by running the function on the running example on CPU, or from a test fixture. Label each sample measured, fixture, or illustrative.
- In the Overview call tree, end each node that produces something with `-> <artifact, type/shape, location>`. Under the tree, add the artifacts table.
- When a config key picks one implementation from a family (loss, optimizer, scheduler, sampler, augmentations, attention backend, metrics), follow stage-playbooks.md section 14: show the switch (key, default, factory), explain the golden variant in depth, and put the other variants in one comparison table with their files, differences, users, and tests.
- If your scope includes a model, first list every argument of its `forward` and every input channel built in `__init__`, and find what switches each one on. A channel that production data turns on (for example when a vocab file is not empty) is on the golden path even if its default is off. Explain it, and give the running example a value for it.
- If your scope includes a model, start its section with the architecture diagram from stage-playbooks.md section 7: a Mermaid `flowchart TB` from the input tensors to every output head, with quoted shapes on the edges (`-->|"[B, T, 256]"|`), the repeated stack drawn once as `× N`, and residuals and conditioning inputs shown. Add the ASCII shape flow and the classDiagram after it. For an architecture card, follow "Architecture card" in doc-templates.md.
- If this brief contradicts the code (a wrong default, class, or step count), follow the code, say so in the doc, and report it in your reply.
- When the code does not explain why a choice was made and a reader would ask, you may run up to about 5 targeted history lookups: `git log -L <start>,<end>:<file> --oneline -n 3` or `git blame -L <n>,<n> <file>`. Cite the short hash. Do not read the whole log.
- You may run cheap CPU-only snippets to confirm shapes or values (tiny config, fake batch). Do not use a GPU. Reuse the available environment; coordinate any authorized isolated dependency setup with the main agent. Do not edit any file in the source repo.
- <include only if allowed> You may run the golden path end to end in a scratch copy: `cp -r <repo> /tmp/explain-run-<code>`. Downloads under ~200 MB, CPU only, under ~10 minutes, existing environment only. Label the values from that run as measured, with the command, hardware, and versions. Delete the scratch copy when done.
- Selected visuals: read SKILL_DIR/visual-guide.md. Use shared values and explicit assumptions; suggest a figure/player only when it clarifies a reader question. A video needs a teaching objective, an advantage over static panels, and meaningful changing inputs/outputs or interacting states. Do not add a generic template clip merely because the tools exist. Preserve the brief, source symbols, evidence status and trace snapshots. Do not substitute illustrative timing for measurements.
- Diagrams: the ASCII call tree per step, plus the Mermaid views your playbook sections require (`classDiagram` for the model's module tree, `sequenceDiagram` for a training step or a request).
- If a claim cannot be confirmed from the code, write it as "UNVERIFIED: <claim> (<reason>)".
- For the Tests entries, list the test functions you find that cover each step (path::name) and what they assert. Do not run the test suite. The test mapper runs it later. Use status "unknown" for now. For important steps with no test, write a "Gap:" line with a proposed test name and assertion.
- If this subsystem is too big for one doc (more than ~1500 lines), stop after the Overview call tree and return a proposed split into child docs instead.

Write:
1. <repo>/docs/explain/.work/<code>.md   (the sub-doc)
2. <repo>/docs/explain/.work/<code>.facts.json with:
   {"entry_points": [...], "components": [{"name","file","line","role","calls":[...]}],
    "steps": [{"id","name","input","output","code","tests":[...], "gap": null|"..."}],
    "terms": [{"term","meaning","code_name"}], "unverified": [...], "off_golden_path": [...],
    "shapes": [{"name","shape","tiny","real","made_at"}], "surprises": ["facts that contradict a reasonable first guess"]}

Before consolidation, incorporate reviewer corrections and actual test outcomes into the facts; retain uncertainty and gaps. The parent preserves these in evidence.json before cleanup.

Reply with: the two file paths, the step IDs you defined, the number of UNVERIFIED claims, and any proposed split.
```

---

## Reviser (revise mode)

```
You are revising one existing doc of an ML codebase explanation. Fix the listed items. Leave everything else as it is.

Repo: <abs path>   Commit now: <sha>   Doc: <repo>/<docs dir>/<code>.md
Worklist for this doc (checker output, changed code, and user requests):
<items, one per line, with the step ID or line>
Running example and seed: use the values already in index.md. Do not invent new ones.
Do not open (independent mode only): <paths, or "none">

Before editing, read:
- SKILL_DIR/stage-playbooks.md, the sections for the stages your items touch (for example "7. Model architecture" for a model item). They define what "complete" means for that stage.
- SKILL_DIR/writing-modes.md, "Both modes" (especially "Point at code by name", "Show the code, then explain it", "Show the data, not only the code") and "<mode> mode"
- SKILL_DIR/doc-templates.md, "Step section shape" and "Diagram conventions"

Rules:
- Edit only the sections in the worklist. Do not reorder, rename, or rewrite sections that are not listed. Keep step IDs stable.
- For "shows no example output": run the step's function on the running example on CPU (or use a test fixture) and add the real output: type, shape or schema change, where it lives, and a small sample. Label it measured, fixture, or illustrative.
- For drifted excerpts or changed code: re-read the function, replace the excerpt verbatim, and fix every claim in that section that the change affects.
- Same citation, excerpt, and Mermaid rules as the writer prompt. Never add line numbers.
- Do not edit source files in the repo.

When done, run: python SKILL_DIR/scripts/check_explain_doc.py <repo>/<docs dir>/<code>.md --repo <repo>
Reply with: each worklist item and what you did, any item you could not fix and why, and the checker summary line.
```

---

## Test mapper

```
You are mapping and running tests for an explanation of an ML codebase. Do not edit source or test files.

Repo: <abs path>   Commit: <sha>
Step list: read every <repo>/docs/explain/.work/*.facts.json (field "steps").
Test runner: <pytest / unittest / other, and any setup such as "uv run pytest" or PYTHONPATH>

Do:
1. For each step, confirm the listed tests exist (path::name). Search for more tests that call the step's function or class.
2. Run the tests that are cheap: CPU-only, no network, no cloud credentials, under ~2 minutes each. Use a timeout. Run them by file or node ID so one failure does not hide others.
3. Do not run tests that need a GPU, network, AWS/GCP credentials, or large downloads. Mark them "not run (needs GPU|network|AWS|slow)". Pytest markers and skip decorators are good evidence.
   If a package does not import, use an existing suitable environment or coordinate isolated setup with the main agent when the user has authorized it. Otherwise mark tests "not run (needs deps: <package>)" and give the needed setup command. An installation attempt alone does not make a test pass.
4. For each step with no test, write a gap row: step ID, what is missing, a proposed test name, and the assertion (input -> expected output).

Write <repo>/docs/explain/.work/tests.md with two tables:
| Step | Test | Command | Status | Asserts |
| Step | Missing | Proposed test and assertion |
Status is one of: pass, fail (one-line error), not run (needs X), gap.

Reply with: totals for pass / fail / not run / gap, the exact commands you ran, and the environment you ran them in (Python version, how dependencies were available).
```

---

## Fact checker

Launch this as a fresh agent after `check_explain_doc.py` passes. It must not
be one of the writers.

```
You are fact-checking an explanation of an ML codebase against the code. Do not edit the docs. Report only.

Repo: <abs path>   Commit: <sha>
Docs: <repo>/docs/explain/*.md   Mode: <simple|deep>

Sample at least 15 claims per doc (more for long docs), spread across sections. Prefer claims that are easy to get wrong:
- tensor shapes and dims, default values, config keys
- call order in call trees, which function calls which
- formulas versus the code that computes them (check signs, normalization, masks, reductions)
- which process or thread runs a function, what crosses a process boundary
- test names and what they assert

For each sampled claim, open the named function or config key and decide: correct, wrong, or unsupported. Also sample code excerpts: is each one verbatim from the file it names, and does the explanation under it match what the lines do?
Also check:
- the running example has the same values in every doc
- one concept has one name across docs
- every numbered section has Input, Output, and Tests
- the configuration surface matches the code: each layer that exists, and which one wins
- every argument of each model's `forward` and every input channel built in `__init__` appears in the architecture diagram and the inputs inventory, or is marked off with a reason. Check how each channel is switched on: by config, or by the data (a vocab size, an encoder file). A data-switched channel that production uses must be explained
- <if existing docs were found> each topic in <repo>/docs/explain/.work/must-cover.txt is covered or listed as out of scope, and where the new docs and the old docs disagree, the code decides who is right
- <deep only> sentences follow SKILL_DIR/writing-modes.md "deep mode" structural rules

Reply with a table: doc, section, claim, verdict, evidence (file and function, plus the line you checked), fix. Then the counts of correct / wrong / unsupported.
```
