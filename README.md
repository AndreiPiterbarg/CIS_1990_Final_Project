# Git Explainer Agent

This project implements a Git explainer agent that answers: why does this code exist?

Given a local repository, file path, and line range, the agent:

- traces the relevant commits with `git log -L`
- looks up associated pull requests and linked issues on GitHub
- fetches review comments and issue comments
- optionally reads surrounding file context when commit metadata is ambiguous
- synthesizes a cited explanation, with a deterministic fallback when no LLM is configured

## CLI

```bash
python3 main.py /path/to/repo src/app.py 10 25 --owner octocat --repo-name hello
```

Add `--no-llm` to force the fallback summary and `--enforce-public-repo` to reject private or missing GitHub repos.
