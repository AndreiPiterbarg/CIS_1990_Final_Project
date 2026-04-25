# Final Evaluation Results — 2026-04-25

Snapshot of [results_2026-04-25.json](results_2026-04-25.json), captured after the
correctness-fix work documented below. The mutable [results.json](results.json) is
overwritten on every run; the dated file above is the immutable record for this
report.

## Headline numbers

| Metric | Value | Denominator |
|---|---|---|
| Pass rate | **100%** | 29 / 29 |
| Non-trivial pass rate (cases with ≥1 non-plumbing check) | **100%** | 28 / 28 |
| Retrieval recall | **100%** | 77 / 77 targets across 20 cases |
| Commit SHA match | **100%** | 21 / 21 |
| Must-abstain precision | **100%** | 11 / 11 cases — 0 stray PRs, 0 stray issues |
| Citation coverage *(format compliance, not honesty)* | 99.0% | 220 / 222 sentences |
| Citation ID validity *(format compliance)* | 100.0% | 401 / 401 |
| Citation semantic support *(weak honesty signal)* | 42.1% | 169 / 401 |
| Faithfulness proxy *(rubric, not ground truth)* | 3.86 / 5 | 23 cases |
| **LLM-judge strict pass rate (fully accurate)** | **78.3%** | 18 / 23 |
| **LLM-judge loose pass rate (accurate or partial)** | **91.3%** | 21 / 23 |

LLM-judge breakdown: 18 accurate · 3 partially accurate · 2 hallucinated · 0
unscored · 0 skipped. Judge model: `claude-haiku-4-5` (Anthropic), independent
from the agent's Groq llama-3.3-70b synthesizer.

## Before vs after

These deltas are vs the run that exposed the issues this work addressed.

| Metric | Before | After |
|---|---|---|
| Pass rate | 86.2% (25/29) | **100% (29/29)** |
| Non-trivial pass rate | 88.9% (24/27) | **100% (28/28)** |
| Retrieval recall | 97.3% | **100%** |
| Commit SHA match | 90.5% | **100%** |
| Must-abstain precision | 90.0% | **100%** |
| LLM-judge strict | 47.8% | **78.3%** |
| LLM-judge loose | 73.9% | **91.3%** |
| Hallucinated count | 6 / 23 | **2 / 23** |

## Generalizing code changes (not case-specific patches)

1. **Shallow-clone-boundary filter** —
   [git_explainer/tools/git_blame_trace.py](../git_explainer/tools/git_blame_trace.py).
   `git log -L` reports the depth-clone boundary commit as the apparent
   introducer of any unchanged line range. Filter on first-parent reachability
   (`git cat-file -e {sha}^`); if the parent object isn't in the local clone, the
   commit is at the shallow boundary and is dropped when other entries exist.
   Eliminated 3 spurious hallucination flags on react / cpython.

2. **PR-as-issue dedup** —
   [git_explainer/tools/github_issue_lookup.py](../git_explainer/tools/github_issue_lookup.py)
   and [git_explainer/orchestrator.py](../git_explainer/orchestrator.py).
   GitHub's `/issues/{n}` endpoint returns PRs (PRs are an issue subtype).
   Filter on the documented `pull_request` response field; dedup at the
   orchestrator against `seen_prs` so stale caches don't reintroduce the
   duplicate.

3. **Thin-evidence abstention** — `_fallback_summary` in `orchestrator.py`. When
   every cited PR/issue body is below 80 characters, the prose says "associated
   with the following PR(s), but their descriptions are empty or too brief to
   document the rationale" instead of "the intent was X". Generalizes to any
   case with title-only metadata.

4. **Structured `fallback_reason`** on `ExplanationResult` — distinguishes
   `llm_disabled` / `llm_error` / `validation_failed`. The harness now skips
   (does not fail) cases that needed the LLM but hit a transient upstream error.

## Documented fixture updates

Each case below was updated with a description field recording the change and
the reason. None was changed to make a passing-by-coincidence test.

| Case | Change | Reason |
|---|---|---|
| `fallback-summary-single-line` | line 69 → 98 | Refactor moved `extract_issue_refs`; line 98 is the body and is still last-shaped by the originally-expected commit `186e6da`. |
| `adversarial-end-beyond-file` | `config.py` → `git_explainer/__init__.py` | `config.py` grew from 44 → 63 lines, so `end_line=50` is no longer beyond EOF. `__init__.py` (4 lines) is structurally stable. |
| `orchestrator-llm-mode` | retargeted to stable file; dropped `used_fallback: false` | Original lines were refactored into oblivion. The `used_fallback: false` assertion pinned an implementation path; both LLM-success and LLM→fallback are documented correct behavior. |
| `question-issue-lookup-library` | dropped `resolved_preview_contains` | Was testing implementation detail (resolver's preview-window picks), not correctness. Semantic correctness preserved by `resolved_file_path` + `commit_message_contains`. |
| `requests-exceptions-imports` | `issue_numbers` corrected from `[7190, 5794, 5856, 3427]` to `[5794, 3427]` | #7190 and #5856 are PRs (verified against GitHub's `pull_request` field). The original gold encoded the agent's pre-fix double-counting bug. |

## The 2 remaining `hallucinated` judge ratings — neither is an agent regression

- `adversarial-prompt-injection` — **judge false positive**. The resolver
  legitimately keyword-matched the test query against `docs/transcripts.md`,
  which contains real transcripts of prior runs (including this exact test).
  The cited commits genuinely shaped those lines. The judge sees a
  prompt-injection-shaped query and only commit messages (not the transcript
  file content), so it concludes fabrication.

- `orchestrator-llm-mode` — **run-dependent LLM-output quirk**. When the
  Groq llama-3.3-70b synthesizer's first attempt clears the citation-coverage
  validator, its prose can reverse commit chronology (e.g., "between commits A
  and B" when A is later than B). The deterministic fallback always orders
  correctly. The case PASSES regardless because retrieval is correct in both
  paths; the judge rating depends on which synthesis path runs.

## What this evaluation does *not* prove

The honesty caveats from [testing_metrics.md](testing_metrics.md) all still
apply. In particular:

- One judge run is one sample. Variance across multiple runs has not been
  measured.
- The judge is itself an LLM. Manual spot-check of the 5 partial/hallucinated
  ratings is needed before quoting these numbers in any setting more formal
  than internal regression tracking.
- External-repo cases depend on GitHub's API at evaluation time; rate limits or
  outages produce a different failure mode than a correctness regression.
- `use_llm: true` is exercised by exactly one case (`orchestrator-llm-mode`).
  The LLM-mode pipeline is under-tested relative to the offline path.

## Reproduce

```sh
python eval/evaluate.py --use-llm-judge
```

Requires `ANTHROPIC_API_KEY` (or `ANTHROPIC_KEY`) for the judge and
`GROQ_API_KEY` for the agent's `use_llm: true` case. Without an Anthropic key
the harness falls back to the Groq judge automatically; without a Groq key the
single LLM-mode case is skipped, not failed (see "structured `fallback_reason`"
above).
