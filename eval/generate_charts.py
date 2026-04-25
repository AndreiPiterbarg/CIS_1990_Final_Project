"""Render presentation charts from the immutable dated results snapshot + local cache.

Source of truth = eval/results_2026-04-25.json (sha256-pinned), NOT the mutable
results.json which gets overwritten on every run.

Outputs six PNGs to eval/charts/:
  01_cache_buckets.png        — what the agent has memoized
  02_headline_metrics.png     — pass / retrieval / citations / judge / faithfulness
  03_latency_distribution.png — per-case wall-clock, sorted
  04_latency_by_class.png     — mean latency grouped by case class
  05_before_after.png         — improvement deltas across 8 metrics
  06_llm_judge_breakdown.png  — accurate / partial / hallucinated counts

Run:  python eval/generate_charts.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Presentation styling — minimal, high-DPI, sans-serif, no chartjunk.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.titlepad": 18,
})

ACCENT = "#2563eb"   # blue
SUCCESS = "#16a34a"  # green
WARN = "#f59e0b"     # amber
MUTED = "#94a3b8"    # gray

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "eval" / "results_2026-04-25.json"  # immutable snapshot
CACHE_PATH = ROOT / ".git_explainer_cache.json"
OUT_DIR = ROOT / "eval" / "charts"
OUT_DIR.mkdir(exist_ok=True)

# Before/after deltas from eval/FINAL_RESULTS.md
BEFORE_AFTER = [
    ("Pass rate",              86.2, 100.0),
    ("Non-trivial pass rate",  88.9, 100.0),
    ("Retrieval recall",       97.3, 100.0),
    ("Commit SHA match",       90.5, 100.0),
    ("Must-abstain precision", 90.0, 100.0),
    ("LLM-judge strict",       47.8,  78.3),
    ("LLM-judge loose",        73.9,  91.3),
    ("Hallucinated count*",     6.0,   2.0),  # *raw count out of 23, not %
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def chart_cache_buckets() -> None:
    cache = _load_json(CACHE_PATH)
    buckets = [
        ("contexts",       len(cache.get("contexts", {}))),
        ("diffs",          len(cache.get("diffs", {}))),
        ("commit_prs",     len(cache.get("commit_prs", {}))),
        ("prs",            len(cache.get("prs", {}))),
        ("pr_comments",    len(cache.get("pr_comments", {}))),
        ("etags",          len(cache.get("etags", {}))),
        ("issues",         len(cache.get("issues", {}))),
        ("issue_comments", len(cache.get("issue_comments", {}))),
    ]
    buckets.sort(key=lambda x: -x[1])
    labels = [b[0] for b in buckets]
    values = [b[1] for b in buckets]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [ACCENT if v > 0 else MUTED for v in values]
    bars = ax.barh(labels, values, color=colors, height=0.7)
    ax.invert_yaxis()
    ax.set_title(f"Memory cache contents  ·  {total} entries total", loc="left")
    ax.set_xlabel("Cached entries")

    head = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(val + head * 0.012, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0, head * 1.15)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    out = OUT_DIR / "01_cache_buckets.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def chart_headline_metrics() -> None:
    s = _load_json(RESULTS_PATH)["summary"]
    executed = s["counts"]["passed"] + s["counts"]["failed"] + s["counts"]["errors"]
    judge = s["llm_judge"]
    metrics = [
        ("Pass rate",
         s["pass_rate"] * 100,
         f"{s['counts']['passed']}/{executed}",
         "hard"),
        ("Non-trivial pass rate",
         s["non_trivial_pass_rate"] * 100,
         f"{s['counts']['non_trivial_passed']}/{s['counts']['non_trivial_ran']}",
         "hard"),
        ("Retrieval recall",
         s["retrieval"]["accuracy"] * 100,
         f"{s['retrieval']['matched_count']}/{s['retrieval']['target_count']} targets",
         "hard"),
        ("Must-abstain precision",
         s["retrieval"]["must_abstain_precision"] * 100,
         f"{s['retrieval']['must_abstain_passed']}/{s['retrieval']['must_abstain_cases']} cases",
         "hard"),
        ("LLM-judge strict (accurate)",
         judge["strict_pass_rate"] * 100,
         f"{judge['accurate_count']}/{judge['scored_count']} cases",
         "hard"),
        ("LLM-judge loose (acc + partial)",
         judge["pass_rate"] * 100,
         f"{judge['accurate_count'] + judge['partially_accurate_count']}/{judge['scored_count']}",
         "hard"),
        ("Citation validity (format)",
         s["citation"]["validity"] * 100,
         f"{s['citation']['valid_citation_count']}/{s['citation']['citation_count']}",
         "format"),
        ("Citation coverage (format)",
         s["citation"]["coverage"] * 100,
         f"{s['citation']['cited_sentence_count']}/{s['citation']['citable_sentence_count']}",
         "format"),
        ("Citation semantic support",
         s["citation"]["support_rate"] * 100,
         f"{s['citation']['supported_citation_count']}/{s['citation']['citation_count']}",
         "weak"),
        ("Faithfulness (proxy rubric)",
         s["faithfulness_rubric"]["average"] / 5 * 100,
         f"{s['faithfulness_rubric']['average']:.2f}/5.0",
         "weak"),
    ]
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    notes = [m[2] for m in metrics]
    kinds = [m[3] for m in metrics]

    color_map = {"hard": ACCENT, "format": MUTED, "weak": WARN}
    colors = [color_map[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_title(f"Benchmark headline metrics  ·  {s['benchmark']['case_count']} cases · "
                 f"{s['benchmark']['repo_count']} repos", loc="left")
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    for bar, val, note in zip(bars, values, notes):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%   {note}",
                va="center", fontsize=10.5, fontweight="bold")

    handles = [
        Patch(facecolor=ACCENT, label="hard correctness"),
        Patch(facecolor=MUTED,  label="format compliance"),
        Patch(facecolor=WARN,   label="weak honesty signal"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)

    out = OUT_DIR / "02_headline_metrics.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def chart_latency_distribution() -> None:
    payload = _load_json(RESULTS_PATH)
    cases = payload["cases"]
    summary = payload["summary"]["latency"]

    rows = sorted(
        [(c["case_id"], c.get("elapsed_seconds", 0.0), bool(c.get("skipped")))
         for c in cases],
        key=lambda r: r[1],
    )
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    skipped = [r[2] for r in rows]

    def band_color(v: float, sk: bool) -> str:
        if sk:
            return MUTED
        if v < 1.0:
            return SUCCESS
        if v < 5.0:
            return ACCENT
        return WARN

    colors = [band_color(v, sk) for v, sk in zip(vals, skipped)]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.barh(labels, vals, color=colors, height=0.78)
    ax.set_title(f"Per-case latency  ·  {len(cases)} cases  ·  total {summary['total_seconds']:.1f}s",
                 loc="left")
    ax.set_xlabel("Seconds (wall clock)")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.set_xlim(0, max(vals) * 1.10 if vals else 1)

    # p50 / p95 reference lines drawn against the data x-axis but labeled in axes y-coords
    p50 = summary["p50_seconds"]
    p95 = summary["p95_seconds"]
    trans = ax.get_xaxis_transform()
    for x, label in [(p50, f"p50 = {p50:.2f}s"), (p95, f"p95 = {p95:.2f}s")]:
        ax.axvline(x, color="#475569", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(x, 1.01, f"  {label}", transform=trans,
                fontsize=10, color="#475569", va="bottom")

    handles = [
        Patch(facecolor=SUCCESS, label="< 1 s   local / abstention"),
        Patch(facecolor=ACCENT,  label="1–5 s   cached external + question mode"),
        Patch(facecolor=WARN,    label="> 5 s   cold external repos"),
        Patch(facecolor=MUTED,   label="skipped"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11)

    out = OUT_DIR / "03_latency_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def chart_latency_by_class() -> None:
    cases = _load_json(RESULTS_PATH)["cases"]

    def classify(cid: str) -> str:
        if cid.startswith("adversarial"):
            return "adversarial\n(rejected fast)"
        if cid.startswith("abstention") or cid == "no-github-metadata":
            return "abstention\n(local)"
        if cid.startswith("question"):
            return "question mode\n(local)"
        if cid.startswith("flask") or cid.startswith("requests"):
            return "flask / requests\n(large PR threads)"
        if cid.startswith("react"):
            return "react\n(external)"
        if cid.startswith("cpython"):
            return "cpython\n(external)"
        return "local repo\n(this codebase)"

    groups: dict[str, list[float]] = defaultdict(list)
    for c in cases:
        if c.get("skipped"):
            continue
        groups[classify(c["case_id"])].append(c.get("elapsed_seconds", 0.0))

    order = sorted(groups, key=lambda k: sum(groups[k]) / len(groups[k]))
    means = [sum(groups[k]) / len(groups[k]) for k in order]
    counts = [len(groups[k]) for k in order]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bars = ax.bar(order, means, color=ACCENT, width=0.55)
    ax.set_title("Mean latency by case class", loc="left")
    ax.set_ylabel("Seconds (wall clock)")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    head = max(means) if means else 1
    for bar, m, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + head * 0.025,
                f"{m:.2f}s\nn={n}",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_ylim(0, head * 1.25)
    out = OUT_DIR / "04_latency_by_class.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def chart_before_after() -> None:
    labels = [m[0] for m in BEFORE_AFTER]
    before = [m[1] for m in BEFORE_AFTER]
    after = [m[2] for m in BEFORE_AFTER]

    import numpy as np
    y = np.arange(len(labels))
    h = 0.38

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.barh(y - h / 2, before, height=h, color=MUTED, label="before")
    ax.barh(y + h / 2, after,  height=h, color=ACCENT, label="after")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Before vs after  ·  correctness-fix work (2026-04-25)", loc="left")
    ax.set_xlim(0, 118)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

    for yi, b, a in zip(y, before, after):
        suffix = "" if labels[int(yi)] != "Hallucinated count*" else "  /23"
        ax.text(b + 1.2, yi - h / 2, f"{b:.1f}{suffix}", va="center",
                fontsize=10, color="#475569")
        ax.text(a + 1.2, yi + h / 2, f"{a:.1f}{suffix}", va="center",
                fontsize=10, color=ACCENT, fontweight="bold")

    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.text(0.01, 0.01,
             "*Hallucinated count is a raw count out of 23 judged cases, not a percentage.",
             fontsize=9, color="#64748b", style="italic")

    out = OUT_DIR / "05_before_after.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def chart_llm_judge_breakdown() -> None:
    judge = _load_json(RESULTS_PATH)["summary"]["llm_judge"]
    cats = [
        ("accurate",            judge["accurate_count"],            SUCCESS),
        ("partially accurate",  judge["partially_accurate_count"],  WARN),
        ("hallucinated",        judge["hallucinated_count"],        "#dc2626"),
    ]
    labels = [c[0] for c in cats]
    values = [c[1] for c in cats]
    colors = [c[2] for c in cats]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_title(f"LLM-judge breakdown  ·  judge = claude-haiku-4-5  ·  {total} scored cases",
                 loc="left")
    ax.set_xlabel("Cases")
    ax.set_xlim(0, total + 2)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    for bar, v in zip(bars, values):
        pct = v / total * 100 if total else 0
        ax.text(v + 0.25, bar.get_y() + bar.get_height() / 2,
                f"{v}  ({pct:.1f}%)",
                va="center", fontsize=12, fontweight="bold")

    fig.text(0.01, 0.02,
             "Note: both 'hallucinated' ratings are explained as non-regressions in eval/FINAL_RESULTS.md  "
             "(judge false positive on prompt-injection test · run-dependent ordering quirk in LLM-mode).",
             fontsize=9, color="#64748b", style="italic", wrap=True)

    out = OUT_DIR / "06_llm_judge_breakdown.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def main() -> None:
    chart_cache_buckets()
    chart_headline_metrics()
    chart_latency_distribution()
    chart_latency_by_class()
    chart_before_after()
    chart_llm_judge_breakdown()


if __name__ == "__main__":
    main()
