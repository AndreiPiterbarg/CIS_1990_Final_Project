# Git Explainer Testing Metrics

This document explains what the evaluation metrics in `eval/evaluate.py` actually measure, what they do not measure, and how to make the benchmark more rigorous.

## Bottom line

The current harness is stricter than before, but the headline numbers still overstate factual confidence.

- `90.5%` pass rate (across cases that actually ran) does not mean `90.5%` factual accuracy, and the two failing cases (`requests-api-surface`, `requests-exceptions-imports`) are external-repo spans where the stricter `pr_numbers: []` / `issue_numbers: []` scoring now flags unexpected PR or issue hits.
- `100%` retrieval accuracy is computed only on cases that have hand-written positive retrieval targets.
- `98.6%` citation coverage and `100%` citation validity are still close to format-compliance metrics because the synthesizer is instructed to put citations in nearly every sentence.
- `4.99/5` faithfulness is a proxy score, not a human or independently verified honesty score.

As the benchmark is written today:

- There are `22` total cases.
- Only `10` cases have any positive retrieval targets at all.
- `3` external-repo cases currently contribute no positive retrieval targets.
- `1` case includes `commit_message_contains: []`, `7` include `pr_numbers: []`, and `10` include `issue_numbers: []`.
- `3` cases use `expects_no_pull_requests: true`, and `3` use `expects_no_issues: true`.
- `commit_message_contains: []` is skipped as vacuous, while `expects_no_*` is a real negative assertion. For backward compatibility, `pr_numbers: []` and `issue_numbers: []` now also mean "return none", not automatic passes.
- `use_llm: true` cases are skipped (not failed) when the harness is run with `--no-llm`, and are reported separately in the summary so pass rate is not diluted by forced-fallback runs.
- Many `explanation_contains` checks only require a filename or a citation marker such as `[commit:`.

## Metric definitions

### Pass rate

`pass_rate = passed_cases / (passed + failed + errors)`

A case passes when every configured check in `expected` is true. This is only as strong as the checks for that case. If a case only asks for a filename mention and a fallback flag, it can pass without proving the explanation is deeply correct. Skipped cases (for example, `use_llm: true` cases under `--no-llm`) are excluded from the pass-rate denominator and reported separately in the summary.

### Retrieval accuracy

`retrieval_accuracy = matched_targets / retrieval_targets`

The scorer currently counts hits for:

- `commit_message_contains`
- `pr_numbers`
- `issue_numbers`
- `resolved_file_path`
- `resolved_matched_terms`

Important limitations:

- This is a recall-style metric against hand-picked targets, not a precision metric.
- Cases without any positive retrieval targets contribute `n/a`, not `0`.
- Negative assertions such as `expects_no_pull_requests` and `expects_no_issues` strengthen pass/fail scoring, but they do not currently increase the retrieval target denominator.
- Matching commit-message substrings is weaker than matching exact commits or exact diff hunks.

### Citation coverage

`citation_coverage = cited_sentences / citable_sentences`

This checks whether citable sentences in the explanation include at least one bracketed citation token such as `[commit:abc1234]`.

Important limitation:

- The prompt and fallback summary both require citations in every sentence, so this mostly measures output formatting discipline.

### Citation validity

`citation_validity = valid_citations / citation_count`

A citation is considered valid when it points to a commit, PR, or issue that appears in the returned evidence object.

Important limitation:

- Validity does not prove that the cited artifact actually supports the sentence it appears in.

### Faithfulness rubric

The evaluator converts several component ratios into a `1.0` to `5.0` proxy score:

- `answer_completeness`
- `retrieval_support`
- `citation_grounding`
- `scope_honesty`

The overall score is the average of those component ratios, scaled onto a five-point rubric.

Important limitations:

- This is explicitly a proxy metric.
- `scope_honesty` mostly checks whether the `limitations` section is cited, not whether the model truly abstains from unsupported claims.
- When there are no gold retrieval targets, the scorer can still assign strong `retrieval_support` if any commits or a resolved target are returned, which makes the metric optimistic on weakly specified cases.

### Latency

Latency is straightforward wall-clock timing per case plus aggregate `avg`, `p50`, and `p95`.

This is the cleanest metric in the harness, but it should still be compared only across similar run modes:

- local vs external repos
- offline fallback vs LLM mode
- warm cache vs cold cache

## Why the current harness can still look too good

Several design choices can still push the numbers upward:

- The notebook previously claimed to default to an offline smoke run, but its config cell immediately overrode itself to run the full benchmark. That made the notebook behavior less transparent than the markdown around it suggested.
- Many cases check only for shallow evidence of success, such as a filename mention or the presence of `[commit:`.
- Legacy `commit_message_contains: []` placeholders are still vacuous, even though empty PR/issue lists are no longer automatic passes.
- Citation metrics are rewarded by construction because the system prompt and fallback summary enforce citation-heavy output.
- External-repo cases do not yet have frozen gold retrieval targets, so they mostly test "can the agent say something plausible with citations?" rather than "did it retrieve the right evidence?"

## Recommended rigor improvements

### 1. Split formatting from semantics

Report these separately:

- format compliance: JSON shape, section presence, citation syntax
- evidence retrieval: exact commit/PR/issue/path hits
- explanation correctness: whether claims are actually supported

This prevents citation-heavy prose from being mistaken for honesty.

### 2. Prefer explicit negative assertions over legacy empty lists

The scorer now treats `pr_numbers: []` and `issue_numbers: []` as "expect no returned PRs/issues" for backward compatibility, but new cases should prefer dedicated checks such as:

- `expects_no_pull_requests`
- `expects_no_issues`

`commit_message_contains: []` remains vacuous and is skipped entirely, so benchmark authors should avoid it unless the absence of a commit-message target is intentional.

### 3. Use stronger gold targets

Prefer exact or near-exact evidence over loose string containment:

- exact commit SHAs
- exact PR/issue IDs
- required diff hunks or changed symbols
- required blame ancestry for line-history cases

This makes retrieval harder to game with vaguely similar commits.

### 4. Add precision-oriented penalties

The harness should check not only whether the right evidence appeared, but also whether obviously wrong evidence appeared. Useful additions:

- extra unexpected commit penalty
- unsupported PR/issue penalty
- irrelevant citation rate

### 5. Judge explanation support sentence by sentence

For each explanation sentence, verify:

- does it make a factual claim?
- is there at least one relevant citation?
- does the cited artifact actually support that claim?

This can be done with a human rubric, or with a blinded judge plus manual spot checks.

### 6. Freeze external-repo fixtures

External repos are useful, but they need stable gold data. For each external case, store:

- known-good commit IDs
- expected commit count range
- expected symbol or diff evidence

Otherwise the benchmark mixes "real retrieval" with "plausible narration."

### 7. Continue adding abstention and uncertainty cases

The new abstention cases improve coverage, but the suite still needs broader support for cases where the agent should say:

- the evidence is insufficient
- no relevant commit was found
- the PR or issue link is missing

Without those cases, the harness still rewards completeness more than careful abstention.

### 8. Run repeated trials for LLM mode

Any case using `use_llm=True` should be run multiple times and reported with variance:

- pass count across runs
- retrieval variance
- citation variance
- faithfulness variance

One successful run is not enough to characterize stochastic behavior.

### 9. Report denominators in the notebook summary

Every aggregate metric should state how many cases actually contributed to it. For example:

- retrieval accuracy: `10/22` cases had positive retrieval targets
- faithfulness proxy: `16/22` cases produced a scored explanation

Without denominators, the numbers read as more comprehensive than they are.

## Practical interpretation

A fair reading of the current harness is:

- it can verify basic plumbing, guardrails, and some abstention behavior
- the fallback path is very good at producing citation-shaped output
- the stricter empty-list scoring now catches unsupported PR or issue links that previously slipped through
- the current benchmark still does not strongly prove honesty or deep factual faithfulness

That means the current results are useful as a smoke test plus a light regression suite, but not yet strong enough to present as a rigorous honesty evaluation.
