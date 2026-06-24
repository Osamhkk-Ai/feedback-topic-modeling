# =====================================================================
# charts.py  —  Simple matplotlib charts (no seaborn).
#
# Uses the headless "Agg" backend and closes every figure to avoid
# memory leaks. Chart axis labels use topic IDs (English) because
# matplotlib cannot shape/right-to-left Arabic text correctly; the
# Arabic names live in the data sheets / CSVs instead.
# =====================================================================

import matplotlib
matplotlib.use("Agg")           # headless; must be set before pyplot import
import matplotlib.pyplot as plt


def _placeholder(out_path, message):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def top10_bar(sizes, out_path, title="Top 10 Topics by Count"):
    items = sorted([(t, c) for t, c in sizes.items() if t != -1],
                   key=lambda x: x[1], reverse=True)[:10]
    if not items:
        _placeholder(out_path, "No topics found")
        return
    labels = [f"T{t}" for t, _ in items]
    counts = [c for _, c in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, counts)
    ax.invert_yaxis()                       # largest bar on top
    ax.set_xlabel("Number of comments")
    ax.set_ylabel("topic_id")
    ax.set_title(title)
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pct_distribution(sizes, out_path, title="Topic Percentage Distribution"):
    items = sorted([(t, c) for t, c in sizes.items() if t != -1],
                   key=lambda x: x[1], reverse=True)
    if not items:
        _placeholder(out_path, "No topics found")
        return
    total = sum(c for _, c in items) or 1
    labels = [f"T{t}" for t, _ in items]
    pct = [100.0 * c / total for _, c in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, pct)
    ax.set_ylabel("% of assigned comments")
    ax.set_xlabel("topic_id")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def noise_bar(sizes, out_path, title="Assigned vs Noise comments"):
    assigned = sum(c for t, c in sizes.items() if t != -1)
    noise = sizes.get(-1, 0)
    total = assigned + noise or 1
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Assigned", "Noise"], [assigned, noise], color=["#4c72b0", "#c44e52"])
    for b, v in zip(bars, [assigned, noise]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({100*v/total:.0f}%)",
                ha="center", va="bottom")
    ax.set_ylabel("comments")
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


