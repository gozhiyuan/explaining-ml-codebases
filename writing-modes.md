# Writing modes

Both modes share the rules in "Both modes". Then apply the rules for your mode.
`simple` borrows from the `eli5` skill. `deep` borrows from the `asd-ste100`
skill (Simplified Technical English). If those skills are installed, you may
read them for more detail. This file is enough on its own.

## Both modes

- **Purpose before mechanism.** Say why a step exists before you say how it works.
- **Code is the source of truth.** Every claim comes from code you read. When a doc, a comment, or a paper disagrees with the code, follow the code and record the mismatch. Templates specify structure, not repository facts. A worked example needs a named source/fixture or an explicit illustrative label; `pass`, `fail` and measured values require retained execution evidence.
- **One thing, one name.** Pick one term for each concept and use it everywhere. Do not switch between "snapshot", "row", "record", and "sample" for the same thing. Put the term in the glossary with its code name.
- **Code names stay exact.** Write identifiers in backticks, spelled as in the code: `KVCacheManager`, `GPT.forward`.
- **Keep hedges.** If the code or the evidence only supports "may", write "may". Never upgrade a guess to a fact. Mark what you could not confirm: `UNVERIFIED: <claim> (reason)`.
- **No marketing adjectives.** Drop "powerful", "robust", "seamless", "blazing-fast". Give the number or the mechanism instead.
- **Say it where it is used.** Do not make the reader jump to another section to decode a label. Repeat the short definition in place.
- **Concrete over abstract.** Every abstract sentence gets an example within a few lines.
- **No arrow chains in prose.** "A -> B -> C fails" goes in a diagram. In prose, write a sentence.

### Give the reader an easy entry into deep detail

Depth means complete evidence, not a difficult introduction. The mental model
starts with the system's job and the input it turns into a result. Follow with
3-6 main actions in execution order: who acts, what it receives, and what it
produces. Use ordinary words before introducing code names, counters or math.
Put branches, exceptions and invariants in the detailed sections that need them.
If several paths exist, introduce the selected path before comparing variants.

At each subsystem overview and each section that introduces a new mechanism:

1. Open with one sentence saying what this part accomplishes and why it exists.
2. Give a short ordered list of the meaningful actions, usually 2-4. Describe
   changes to data or state, not a list of function names.
3. Show a small input/output pair from the shared example, with its provenance.
4. Then add source excerpts, formulas, ownership rules, tests and alternatives.

Keep this proportional: a short single-action step needs only its existing
Idea/Input/Output labels. A glossary or test matrix needs no artificial process
list. For parallel work, describe the actors and dependencies rather than invent
a total execution order. Do not repeat the whole overview inside every step.

Check the opening alone: can a newcomer say what arrives, who changes it, and
what comes out without consulting the code? If not, simplify the opening while
retaining the technical detail below it. Explain essential terms where used.

### Point at code by name, not by line

Readers do not jump to line numbers, and line numbers go stale on the next
commit. Name the function, class, or config key, and the file once per section:

| Do | Don't |
|----|-------|
| `<symbol>` in `<source file>` | A bare line number without its symbol or context |
| `<config key>` with its value and defining class/file at the selected commit | A default copied from a different version |
| Call tree comment with the actual source file | A machine-local path or stale line number |

Rules:

- Name the file the first time a section mentions a symbol. After that, the symbol alone is enough.
- Use the shortest path that is unique in the repo (`src/serve.py`, not the full path), or declare a path root for the doc.
- A line number is allowed only when the code has no name to point at, for example one magic constant inside a long function. Then prefer an excerpt.
- Never put a path in a sentence just to prove you read it. Put it where a reader would open the file.

### Show the code, then explain it

A reader understands a step faster from 10 real lines than from a description
of them. For every step that does real work, paste the key part of the code:

- **Copy it verbatim** from the file. Keep the real names, the order, and the control flow. Do not paraphrase code into pseudo-code inside a code block.
- **Trim, don't rewrite.** Cut unrelated lines and replace each cut with `# ...`. Keep 5-25 lines.
- **Label it.** The first line is a comment with the file and symbol: `# model.py  GPT.forward`.
- **Annotate it.** Add end-of-line comments with shapes and running-example values: `# [B, T, 4]`, a comment with values from the shared example.
- **Explain after it.** Below the block, walk through what the lines do, in order, using the running example.

For a source excerpt paired with measured shapes, see
[nanoGPT's embedding step](https://github.com/gozhiyuan/explaining-ml-codebases/blob/main/examples/nanogpt/model.md#modela1-embed-tokens-and-positions-modelpy).
Its fixture uses `[B,T,D]=[2,4,8]`; that is an explicitly recorded tiny
configuration, not a shape to assume in another repository.

Pseudo-code is fine in prose or in a diagram, never dressed up as a real
excerpt. The doc checker compares each labeled excerpt with the file and warns
when the code has changed.

### Show the data, not only the code

In ML and data pipelines, the reader needs to see what each step produces:
its type, its shape, its new columns, and where it lives. "Returns the
features" teaches nothing. Every step shows its output as the reader would
see it in a debugger, a notebook, or `aws s3 ls`. Use the running example.

| Output kind | Show |
|-------------|------|
| pandas / Spark DataFrame | row count and grain ("one row per entity and time bucket"), the schema change (columns added, dropped, retyped, with dtypes), and 2-5 rows of the running example as a table. For Spark, say whether it is lazy or written |
| numpy array / tensor | dtype, shape with named dims, device, and a small slice: `x[0, :3]` |
| dict / dataclass / JSON | the keys with value types, then a trimmed real example (pretty JSON, `...` for long lists) |
| file on disk or S3 | the path pattern, format, partitioning, size or row count, a listing tree, and one record |
| model output | each tensor's name, shape, and one position's values (logits, probabilities, decoded class) |
| side effect only | what changed and where: a registry entry, an endpoint, a log line, a `_SUCCESS` marker |

Say where the object lives every time: in memory (one process or per Spark
executor), on local disk, or in S3. Label each sample "measured" (you ran the
code), "from a test fixture" (name it), or "illustrative" (built by hand to
match the code). Get real values when you can: run the function on the
running example in a CPU snippet, or read a test fixture.

Output schema (a template, not a measured dataset):

```text
<artifact or in-memory object> (<location>, <row grain>, <observed row count>)
+ <added column>    <dtype>    <source operation>
- <removed column> <dtype>    <reason>
<2-5 rows from the named fixture or measured running example>
```

## `simple` mode (ELI5-style)

### Pick the audience

Parse who the reader is. If the user did not say, use "smart newcomer who
knows nothing about this repo or this ML subfield".

| Audience | Frame it around | Analogy source |
|----------|-----------------|----------------|
| Kid, or non-technical friend or family | what it does for people | kitchen, school, mail, library |
| Manager, director | impact, cost, risk, what decisions it needs | business process, factory line |
| Product manager | user value, what the system can and cannot do | user stories |
| New engineer | where things live, how to run and change them | familiar software (web server, database) |
| Data scientist new to the stack | how the infrastructure maps to the ML they know | notebook equivalents |

### Shape of each stage

1. **What** in one sentence.
2. **Analogy** to something the audience knows.
3. **How it works** in 3-6 short steps.
4. **Why it matters** to this audience.
5. **Where in the code** (file names) and **how to check it** (one test command).

### Rules

- No jargon. If a term is essential, define it in the same sentence.
- One idea per sentence. Short sentences.
- Use "you" and the reader's world.
- Math: give the intuition in words. Show at most one formula per stage, and name each symbol in words.
- Numbers stay small and friendly ("3 requests", "a sentence of 5 words").
- One big-picture diagram per doc. Mermaid flowchart, under ~12 nodes, plain labels.
- **Never false.** Simplifying is fine. If a simplification drops something that matters, add a one-line `Simplified:` note that says what was left out.
- **Never talk down.** The reader is smart and new, not slow.
- Length: the whole `index.md` should fit in about 10 minutes of reading.

### Example (simple)

> **What.** The scheduler decides which requests the GPU works on next.
>
> **Analogy.** Think of a busy kitchen with one big stove. Orders keep
> arriving. The head chef does not cook one order start to finish. Every
> minute, the chef puts a bit of each open order on the stove, as many as fit.
>
> **Why it matters.** This is why the system can answer many users at once
> without each one waiting in a long line.
>
> Simplified: the real "minute" is one model step, and "fits on the stove"
> includes the configured token budget and available KV memory.

## `deep` mode (STE-style)

### Structural rules (apply all)

| Rule | Do | Don't |
|------|----|-------|
| Active voice | "`Scheduler.schedule` allocates the KV blocks." | "The KV blocks are allocated." |
| Sentence length | <= 20 words for steps, <= 25 words for descriptions | long sentences with several clauses |
| One idea per sentence | "The worker loads the weights. Then it captures CUDA graphs." | "...loads the weights and then, if enabled, captures..." |
| No semicolons in prose | two sentences | `;` (code and tables are fine) |
| Noun clusters <= 3 words | "the block table of the request" | "request KV cache block table entry" |
| No soft phrasal verbs | start, contact, read, stop | spin up, reach out, dive into, kick off |
| No dropped words | "Files that are not saved are lost." | "Files not saved lost." |
| Keep modality | "may preempt" stays "may preempt" | turning "may" into "does" |
| Simple tenses | "The step wrote the file." | "The step has written the file." unless the present result matters |
| Paragraphs | one topic, <= 6 sentences | mixed-topic walls |
| Lists for sequences | 3+ steps go in a numbered list or a call tree | a sequence buried in prose |

### Precision rules

- **Shapes** use named dims with values from the selected configuration and the running example. Define each dim once per doc in a dims table. Do not label an invented shape as production.
- **Units** on every number: hours, ms, bytes, tokens, rows.
- **Defaults** name their defining symbol and file at the selected commit. Read the value there instead of copying a default from this guide.
- **Formulas** define every symbol, then show a numeric example from the running example.
- **Call order** goes in a call tree, not in prose.
- **Counts** come from code or a run. If you estimate, say "about" and how you estimated.

### Worked deep example

See [vLLM's scheduler step](https://github.com/gozhiyuan/explaining-ml-codebases/blob/main/examples/vllm/scheduler.md) for concrete budgets,
request state, outputs and source excerpts. Its validation uses selected
statements with stand-ins; it does not prove that a complete scheduler or GPU
worker executed. Keep that boundary attached to the example.

### Lint

If the `asd-ste100` skill is installed, run its linter on the finished doc.
Resolve `STE_SKILL_DIR` to that skill's installed directory first. It skips
code blocks and table syntax:

```bash
python "$STE_SKILL_DIR/scripts/ste-lint.py" --disable passive-voice docs/explain/*.md
```

Fix hard violations in prose. Identifiers and quoted log lines may trip the
length or phrasal-verb checks. Leave those as they are.
