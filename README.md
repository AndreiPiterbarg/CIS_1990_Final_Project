# Git Explainer Agent

This project implements a Git explainer agent that answers: why does this code exist?

Given a local repository and either a line range or a natural-language question, the agent:

- traces the relevant commits with `git log -L`
- looks up associated pull requests and linked issues on GitHub
- fetches review comments and issue comments
- optionally reads surrounding file context when commit metadata is ambiguous
- can resolve a question like "Why is requests used for GitHub issue lookups?" into a concrete file and line span before tracing history
- synthesizes a cited explanation, with a deterministic fallback when no LLM is configured

## CLI

```bash
python3 main.py /path/to/repo src/app.py 10 25 --owner octocat --repo-name hello
```

Question mode works without line numbers:

```bash
python3 main.py /path/to/repo --question "Why is requests used for GitHub issue lookups?"
```

You can also provide a file hint in question mode to narrow the search:

```bash
python3 main.py /path/to/repo git_explainer/tools/github_issue_lookup.py --question "Why is requests used here?"
```

Add `--no-llm` to force the fallback summary and `--enforce-public-repo` to reject private or missing GitHub repos.
