#!/usr/bin/env python3
"""Generate bar charts from chess game statistics JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_stats(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_elo_distribution(stats: dict, output_path: Path):
    elo_data = stats.get("elo_distribution", [])
    if not elo_data:
        print("No ELO distribution data found")
        return

    labels = [f"{e['elo_min']}-{e['elo_max']}" for e in elo_data]
    counts = [e["count"] for e in elo_data]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), counts, color="#4a90d9")

    ax.set_xlabel("ELO Range")
    ax.set_ylabel("Game Count")
    ax.set_title(f"Game Distribution by Max ELO (Total: {stats['total_games']:,})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.annotate(
                f"{count:,}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ELO distribution chart to {output_path}")


def plot_time_control_distribution(stats: dict, output_path: Path):
    tc_data = stats.get("time_control_distribution", [])
    if not tc_data:
        print("No time control distribution data found")
        return

    total = sum(e["count"] for e in tc_data)
    labels = [e["category"].capitalize() for e in tc_data]
    counts = [e["count"] for e in tc_data]
    percentages = [(c / total * 100) if total > 0 else 0 for c in counts]

    colors = {
        "Bullet": "#e74c3c",
        "Blitz": "#f39c12",
        "Rapid": "#27ae60",
        "Classical": "#3498db",
        "Unknown": "#95a5a6",
    }
    bar_colors = [colors.get(label, "#95a5a6") for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color=bar_colors)

    ax.set_xlabel("Time Control")
    ax.set_ylabel("Game Count")
    ax.set_title(f"Game Distribution by Time Control (Total: {total:,})")

    for bar, count, pct in zip(bars, counts, percentages):
        ax.annotate(
            f"{count:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved time control chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate charts from stats JSON")
    parser.add_argument("stats_file", type=Path, help="Path to stats JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (defaults to figs/ alongside input file)",
    )
    args = parser.parse_args()

    if not args.stats_file.exists():
        print(f"Error: {args.stats_file} not found")
        return 1

    output_dir = args.output if args.output else args.stats_file.parent.parent / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.stats_file)
    stem = args.stats_file.stem.replace("_stats", "")

    plot_elo_distribution(stats, output_dir / f"{stem}_elo_distribution.png")
    plot_time_control_distribution(stats, output_dir / f"{stem}_time_control.png")

    return 0


if __name__ == "__main__":
    exit(main())
