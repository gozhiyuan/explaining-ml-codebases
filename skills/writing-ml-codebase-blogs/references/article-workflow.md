# From code evidence to an article

Read the explanation index, relevant subsystems and `evidence.json`. Choose a
central question and the smallest configuration that exercises it. A useful
article often has a problem, mental model, worked trace, implementation choices
and tradeoffs; these are prompts, not fixed headings. Keep reference test
matrices in supporting docs when they interrupt the story.

Revisit source after planning. Verify constructor settings, branch conditions,
ownership and consumers. If code and docs differ, explain the selected version
and keep the mismatch in the ledger. Use connected prose, a shared example,
short real excerpts and explicit simplifications. Define equation variables
where they affect behavior. Never manufacture benchmark numbers.

## Article manifest

Write `article.json` beside Markdown:

```json
{
  "schema_version": 1,
  "title": "How one token window becomes training loss",
  "question": "What happens to a tiny batch during a training step?",
  "audience": "Python engineer new to this model",
  "scope": "One CPU batch; distributed training is deferred",
  "source": {"url": "https://github.com/karpathy/nanoGPT", "commit": "record-the-actual-sha"},
  "assumptions": ["Tiny configuration, dropout disabled"],
  "claims": [
    {"id": "loss", "statement": "Targets train every sequence position.",
     "sources": [{"file": "model.py", "symbol": "GPT.forward"}],
     "evidence_status": "source-reviewed"}
  ],
  "figures": [
    {"id": "token-window", "asset": "assets/token-window.svg",
     "caption": "Inputs and shifted targets share a token window.",
     "claim_ids": ["loss"], "evidence_status": "illustrative"}
  ],
  "uncertainties": [],
  "checks": []
}
```

Use the actual SHA. Claim status is `source-reviewed`, `fixture`, `measured`,
or `unverified`; figures additionally use `illustrative`. A measured claim
needs `command` and `result`, with paths relative to delivered artifacts. An
unverified claim needs `reason` and explicit qualification in the prose. The
checker never runs these commands. The `checks` array records actual command,
outcome and boundary, such as tiny CPU forward rather than a serving benchmark.
Retain input evidence uncertainties. Copy relevant test gaps to the article's
limitations or supporting evidence even when the main body is shortened.

Render the final article and check its central question, source accuracy,
responsive layout, figure labels, code overflow, math and media separately.
Correct Markdown or editable visuals before regenerating HTML. A successful
symbol lookup is only a structural check; inspect the actual branch and caller.

Use `symbol: "<module>"` for a module-level loop, assignment or call order.
This explicitly requests a file-level reference; verify the relevant statements
manually instead of inventing a function name for script-level behavior.
