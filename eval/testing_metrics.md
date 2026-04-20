# Git Explainer Testing Metrics

This document explains what the evaluation metrics in [evaluate.py](/Users/alistairking/Projects/upenn-courses/cis-1990/CIS_1990_Final_Project/eval/evaluate.py) actually measure, what they do not measure, and how to make the benchmark more rigorous.

## Bottom line

The current results do not look outright fabricated, but they are easier to score well on than the headline numbers suggest.

- `95%` pass rate does not mean `95%` factual accuracy.
- `100%` retrieval accuracy is computed only on cases that have hand-written retrieval targets.
- `98.5%` citation coverage and `100%` citation validity are close to format-compliance metrics because the synthesizer is instructed to put citations in every sentence.
- `4.99/5` faithfulness is a proxy score, not a human or independently verified honesty score.

As the benchmark is written today:

- There are `20` total cases.
- Only `10` cases have any retrieval targets at all.
- `3` external-repo cases currently contribute no retrieval targets.
- `2` cases include `commit_message_contains: []`, `8` include `pr_numbers: []`, and `11` include `issue_numbers: []`; those checks are structurally always true under the current scorer.
- Many `explanation_contains` checks only require a filename or a citation marker such as `[commit:`.

## Metric definitions

### Pass rate

`pass_rate = passed_cases / total_cases`

A case passes when every configured check in `expected` is true. This is only as strong as the checks for that case. If a case only asks for a filename mention and a fallback flag, it can pass without proving the explanation is deeply correct.

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
- Cases without any retrieval targets contribute `n/a`, not `0`.
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

## Why the current notebook can look too good

Several design choices push the numbers upward:

- The notebook previously claimed to default to an offline smoke run, but its config cell immediately overrode itself to run the full benchmark. That made the notebook behavior less transparent than the markdown around it suggested.
- Many cases check only for shallow evidence of success, such as a filename mention or the presence of `[commit:`.
- Empty expected lists are treated as passing checks, which inflates the count of “all checks passed.”
- Citation metrics are rewarded by construction because the system prompt and fallback summary enforce citation-heavy output.
- External-repo cases do not yet have frozen gold retrieval targets, so they mostly test “can the agent say something plausible with citations?” rather than “did it retrieve the right evidence?”

## Recommended rigor improvements

### 1. Split formatting from semantics

Report these separately:

- format compliance: JSON shape, section presence, citation syntax
- evidence retrieval: exact commit/PR/issue/path hits
- explanation correctness: whether claims are actually supported

This prevents citation-heavy prose from being mistaken for honesty.

### 2. Replace empty-list checks with explicit negative assertions

Instead of:

- `pr_numbers: []`
- `issue_numbers: []`

use dedicated checks such as:

- `expects_no_pull_requests`
- `expects_no_issues`

That makes “no linked metadata” a real test instead of an always-true placeholder.

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

Otherwise the benchmark mixes “real retrieval” with “plausible narration.”

### 7. Add abstention and uncertainty cases

A strong explainer should sometimes say:

- the evidence is insufficient
- no relevant commit was found
- the PR/issue link is missing

Today the harness rewards completeness much more than careful abstention.

### 8. Run repeated trials for LLM mode

Any case using `use_llm=True` should be run multiple times and reported with variance:

- pass count across runs
- retrieval variance
- citation variance
- faithfulness variance

One successful run is not enough to characterize stochastic behavior.

### 9. Report denominators in the notebook summary

Every aggregate metric should state how many cases actually contributed to it. For example:

- retrieval accuracy: `10/20` cases had retrieval targets
- faithfulness proxy: `14/20` cases produced a scored explanation

Without denominators, the numbers read as more comprehensive than they are.

## Practical interpretation

A fair reading of the current notebook is:

- the harness can verify basic plumbing and guardrails
- the fallback path is very good at producing citation-shaped output
- the current benchmark does not yet strongly prove honesty or deep factual faithfulness

That means the current results are useful as a smoke test, but not yet strong enough to be presented as a rigorous honesty evaluation.
