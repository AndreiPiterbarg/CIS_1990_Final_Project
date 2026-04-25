# Git Explainer Testing Metrics

This document explains what the evaluation metrics in `eval/evaluate.py` actually measure, what they do not measure, and where the honest boundaries are.

## Bottom line

The harness now separates **format compliance**, **evidence retrieval**, and **explanation honesty** so headline numbers from one bucket cannot impersonate another. Concretely:

- Pass rate is reported twice: overall and restricted to cases with at least one non-plumbing check. Cases whose only checks are `used_fallback`, `explanation_contains`, `resolved_file_path`, `resolved_matched_terms`, or `resolved_preview_contains` are flagged as trivial-checks-only and listed by ID in `summary.evaluation_honesty.trivial_check_case_ids`.
- Retrieval accuracy remains recall against hand-authored gold targets. It is reported with an explicit denominator: how many cases actually contributed.
- Retrieval precision is now reported for must-abstain cases. If a case asserts `expects_no_pull_requests`, `expects_no_issues`, or `pr_numbers: []` / `issue_numbers: []`, any returned PR or issue counts as noise and the case fails precision.
- Citation coverage is still reported, but clearly labeled as a format-compliance metric (the prompt and fallback both require a citation in every citable sentence, so this is near 100% by construction).
- Citation validity is clearly labeled as an ID-exists check, not a claim-support check.
- A new citation semantic-support metric is reported: the fraction of citations where the cited artifact's text shares at least one content word with the citing sentence. This is a weak signal against glued-on citations; it is not entailment.
- The faithfulness proxy is still reported, still labeled as a proxy, but is now honest:
  - Components with no signal return `null` instead of defaulting to 1.0. Previously `retrieval_support` silently defaulted to 1.0 on cases without gold targets.
  - `citation_grounding` uses the semantic support rate, falling back to validity. It no longer averages in citation coverage, which was near-100% by construction.
  - `scope_honesty` has been replaced with `abstention_correctness`, which scores thin-evidence cases by whether the explanation surfaces an abstention marker rather than by whether the limitations section happens to contain a `[commit:...]` token.
- When an LLM is configured, the LLM-as-judge pass runs by default and is the headline honesty metric. The proxy rubric is reported secondarily and is labeled `PROXY — not ground truth`. You can opt out with `--no-llm-judge` or `--no-llm`.

## Metric definitions

### Pass rate

`pass_rate = passed_cases / (passed + failed + errors)`

A case passes when every configured check in `expected` is true. Skipped cases are excluded from the denominator and reported separately.

**Non-trivial pass rate** is the same ratio restricted to cases with at least one non-plumbing check. This is the honest number to cite; raw pass rate is inflated by cases that only check whether the agent emitted citation-shaped output.

### Retrieval recall

`recall = matched_targets / total_targets`

Counted hits:

- `commit_message_contains`
- `expected_commit_shas`
- `pr_numbers`
- `issue_numbers`
- `resolved_file_path`
- `resolved_matched_terms`

Cases with no positive targets do not contribute — the report prints the number of cases that did (`cases_with_targets`) so the metric cannot be read as "all cases have high recall."

### Retrieval precision (must-abstain cases)

Precision is measured only on cases that assert no PRs or no issues should be returned (`expects_no_pull_requests`, `expects_no_issues`, or an explicit empty `pr_numbers: []` / `issue_numbers: []`). For every such case:

- Any returned PR or issue is counted as noise.
- The case passes precision iff noise == 0.

The summary prints `must_abstain_precision` with its denominator and the total unexpected PRs/issues across the run. We intentionally do not measure precision on positive-target cases because the gold lists are a lower bound, not exhaustive.

### Citation coverage (format compliance)

`coverage = cited_sentences / citable_sentences`

Near-100% by construction. The prompt and fallback both require a citation in every citable sentence. Reported, not celebrated.

### Citation ID validity (format compliance)

`validity = valid_citations / total_citations`

A citation is valid if the cited ID (commit SHA / PR number / issue number) appears in the returned evidence. This is a necessary condition — not evidence that the artifact supports the claim.

### Citation semantic support (weak honesty signal)

`support = supported_citations / total_citations`

A citation is considered supported if the cited artifact's text (commit message; PR/issue title + body) shares at least one content word with the sentence containing the citation. Stopwords and generic change verbs are stripped. This catches glued-on citations — e.g., `"The feature was added [commit:abc1234]"` where commit `abc1234` is about something else — but does not test entailment, and a single shared word can be coincidental.

### Faithfulness rubric (PROXY)

The proxy maps four component ratios into a 1.0-5.0 score and reports their average:

- `answer_completeness`: fraction of non-trivial `explanation_contains` phrases present in the explanation. Returns `null` when there are no non-trivial phrases to check, so cases with only `["filename.py", "[commit:"]` do not inflate this score.
- `retrieval_support`: the measured recall against gold targets. Returns `null` when no gold targets exist — it no longer defaults to 1.0 or to a `min_commits` heuristic.
- `citation_grounding`: uses semantic support rate when there are citations, else validity. Does not use coverage.
- `abstention_correctness`: cases with both PRs and issues in evidence score full marks; thin-evidence cases score full marks iff the explanation mentions an abstention marker (e.g., "No linked pull request", "no associated issue", "not found", "limited evidence").

The overall proxy is the average of the components with signal for that case. The summary reports how many cases produced a proxy score.

The proxy is clearly labeled `PROXY — not ground truth`. Prefer the LLM-as-judge number when reporting a headline honesty figure.

### LLM-as-judge faithfulness (headline honesty)

On each case, a held-out LLM is shown only the question, the retrieved PR/issue/commit evidence, and the agent's explanation, and asked to rate the explanation as `accurate`, `partially accurate`, or `hallucinated`, plus list any contradictions. The judge is instructed to not rely on outside knowledge.

The summary reports:

- `pass_rate`: accurate-or-partially-accurate / scored
- `strict_pass_rate`: accurate / scored
- counts of accurate / partially accurate / hallucinated / unscored / skipped

This is the headline honesty metric. It still has known limits (LLM judges have their own biases, the evidence-bounded prompt can mistake omissions for honesty), so manual spot checks on `hallucinated` and `partially accurate` cases remain necessary.

### Latency

Straightforward wall-clock timing per case, plus aggregate avg / p50 / p95. Compare only across similar run modes (local vs external repos, offline fallback vs LLM mode, warm vs cold cache).

## The `evaluation_honesty` self-audit block

Every results.json now contains a `summary.evaluation_honesty` block with:

- `trivial_check_case_ids`: cases whose only checks are plumbing
- `no_retrieval_target_case_ids`: cases that contribute no retrieval recall signal
- `no_citation_case_ids`: cases that produced no citations at all
- `headline.preferred_faithfulness_metric`: `"llm_judge"` when it was run, else `"none"`
- `caveats`: a list of short strings explaining exactly what each metric does and does not prove

Tools that render this benchmark are encouraged to show those caveats next to the headline numbers.

## What these numbers still do NOT prove

- They do not verify claim-level correctness of the explanation sentence by sentence. The LLM judge approximates this but is not a human rubric.
- They do not measure hallucination in the long tail where the model fabricates a plausible citation and the judge fails to catch it.
- They do not characterize variance: `use_llm=true` cases should be run multiple times and reported with variance bands before being used to compare model versions.
- External-repo cases still rely on the agent's remote GitHub queries at evaluation time, so a rate limit or a GitHub outage produces a different failure mode than a correctness regression.

A fair reading of this harness remains:

- It is a smoke test plus a regression suite for plumbing, guardrails, retrieval recall, and abstention.
- The format-compliance numbers (coverage, validity) should never be quoted as honesty.
- The proxy faithfulness number should never be quoted alone; always report the LLM-judge pass rate alongside it if it was run, and explicitly say "proxy" if it was not.
- The suite does not yet certify the agent as honest in a way a referee would accept without manual review of a sample of cases.
