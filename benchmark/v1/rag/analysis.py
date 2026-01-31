# SPDX-License-Identifier: Apache-2.0
# Third Party
import matplotlib
import pandas as pd

matplotlib.use("Agg")
# Standard
from typing import Dict, List
import argparse
import os

# Third Party
import matplotlib.pyplot as plt
import seaborn as sns

CATEGORY_COLORS = {
    "LMCache": "#1f77b4",
    "CacheBlend": "#ff7f0e",
}

CATEGORY_ORDER = ["LMCache", "CacheBlend"]


def assign_category(scenario_name: str) -> str:
    scenario_lower = scenario_name.lower()
    if "cacheblend" in scenario_lower:
        return "CacheBlend"
    elif "lmcache" in scenario_lower:
        return "LMCache"
    return "Other"


def plot_bar_metric(
    df: pd.DataFrame,
    metric_col: str,
    metric_title: str,
    y_label: str,
    ax: plt.Axes,
):
    mean_df = df.groupby("scenario")[metric_col].mean().reset_index()
    mean_df["category"] = mean_df["scenario"].apply(assign_category)
    mean_df["category"] = pd.Categorical(
        mean_df["category"], categories=CATEGORY_ORDER, ordered=True
    )
    unique_scenarios = mean_df["scenario"].unique()

    def get_sort_key(scenario_name):
        category = assign_category(scenario_name)
        if category == "LMCache":
            category_sort_value = 1
        elif category == "CacheBlend":
            category_sort_value = 2
        else:
            category_sort_value = 3

        return (category_sort_value, scenario_name)

    sorted_scenarios = sorted(unique_scenarios, key=get_sort_key)

    sns.barplot(
        x="scenario",
        y=metric_col,
        hue="category",
        data=mean_df,
        ax=ax,
        palette=CATEGORY_COLORS,
        order=sorted_scenarios,
        hue_order=CATEGORY_ORDER,
    )

    ax.set_title(f"Average {metric_title}")
    ax.set_ylabel(f"Average {y_label}")
    ax.set_xlabel("Scenario")
    ax.tick_params(axis="x", rotation=10)
    ax.legend(title="Category")

    if "Time" in y_label or "Throughput" in y_label:
        ax.set_ylim(bottom=0)


def analyze_benchmark_results(
    csv_paths: List[str], results_dir: str, scenario_map: Dict[str, str]
):
    file_paths: Dict[str, str] = {}

    FINAL_QUALITY_COL = "quality"
    FINAL_LATENCY_COL = "end_to_end_time"
    FINAL_THROUGHPUT_COL = "throughput"

    LATENCY_CANDIDATES = ["ttft", "end_to_end_latency", "generation_time"]

    THROUGHPUT_CANDIDATES = ["throughput", "tpot"]

    print("\n--- Mapping Files to Scenarios ---")

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: File {csv_path} does not exist!")
            continue

        filename = os.path.basename(csv_path)
        scenario = "Unknown Scenario"

        for identifier, display_name in scenario_map.items():
            if identifier in filename:
                scenario = display_name
                break

        if scenario in file_paths:
            print(
                f"Warning: Scenario '{scenario}' already mapped! "
                f"Skipping file {filename}."
            )
            continue

        file_paths[scenario] = csv_path
        print(f"  Mapped: {filename} -> {scenario}")

    if not file_paths:
        print("No valid CSV files to analyze!")
        return

    df_list = []

    for scenario, path in file_paths.items():
        try:
            df_full = pd.read_csv(path)

            latency_col_found = next(
                (col for col in LATENCY_CANDIDATES if col in df_full.columns), None
            )
            throughput_col_found = next(
                (col for col in THROUGHPUT_CANDIDATES if col in df_full.columns), None
            )

            required_cols = []
            rename_map = {}

            if FINAL_QUALITY_COL in df_full.columns:
                required_cols.append(FINAL_QUALITY_COL)

            if latency_col_found:
                required_cols.append(latency_col_found)
                rename_map[latency_col_found] = FINAL_LATENCY_COL

            if throughput_col_found and throughput_col_found != latency_col_found:
                required_cols.append(throughput_col_found)
                rename_map[throughput_col_found] = FINAL_THROUGHPUT_COL

            if not required_cols:
                print(
                    f"Error loading {path}: No required columns "
                    "(quality, latency, or throughput metrics) found."
                )
                continue

            df = df_full[required_cols].copy()
            df["scenario"] = scenario

            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            df_list.append(df)

        except Exception as e:
            print(f"Error loading {path}: {e}")

    if not df_list:
        print("Error: Failed to load any valid dataframes.")
        return

    df = pd.concat(df_list, ignore_index=True)
    print(f"\nSuccessfully loaded {len(df_list)} dataframes.")

    df["category"] = df["scenario"].apply(assign_category)

    analysis_cols = [FINAL_QUALITY_COL, FINAL_LATENCY_COL, FINAL_THROUGHPUT_COL]
    analysis_cols = [col for col in analysis_cols if col in df.columns]

    print("\n=== Core Metrics Statistics by Category and Scenario ===")
    print(
        df.groupby(["category", "scenario"])[analysis_cols]
        .agg(["mean", "median", "std", "min", "max"])
        .round(4)
    )

    sns.set_style("whitegrid")

    metrics_to_plot = []
    if FINAL_QUALITY_COL in df.columns:
        metrics_to_plot.append((FINAL_QUALITY_COL, "Quality Score", "Score"))
    if FINAL_LATENCY_COL in df.columns:
        metrics_to_plot.append((FINAL_LATENCY_COL, "Latency (Time)", "Time (s)"))
    if FINAL_THROUGHPUT_COL in df.columns:
        metrics_to_plot.append(
            (FINAL_THROUGHPUT_COL, "Throughput", "Throughput (req/s)")
        )

    num_plots = len(metrics_to_plot)
    if num_plots == 0:
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    fig.suptitle(
        "Benchmark Core Metrics Analysis: Average Performance Comparison", fontsize=16
    )

    for i, (col, title, y_label) in enumerate(metrics_to_plot):
        plot_bar_metric(df, col, title, y_label, axes[i])

    core_plot_path = os.path.join(results_dir, "metrics_average_analysis.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(core_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n=== Plots saved to {results_dir} ===")
    print(f"1. {os.path.basename(core_plot_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results-dir", required=True, help="Directory to save plots")
    parser.add_argument(
        "--scenario-map",
        nargs="+",
        default=[
            "lmcache_offline:LMCache (No Blend) Offline",
            "cacheblend_offline:CacheBlend Offline",
            "lmcache_online:LMCache (No Blend) Online",
            "cacheblend_online:CacheBlend Online",
        ],
        help="Map filenames identifiers to scenario display names "
        '(e.g., "lmcache_offline:LMCache (No Blend) Offline")',
    )
    parser.add_argument("csv_paths", nargs="+", help="Paths to benchmark CSV files")
    args = parser.parse_args()

    scenario_map_dict = {}
    for item in args.scenario_map:
        try:
            identifier, display_name = item.split(":", 1)
            scenario_map_dict[identifier.strip()] = display_name.strip()
        except ValueError:
            print(
                f"Error: Invalid scenario map format '{item}'. "
                "Use 'identifier:DisplayName'."
            )
            exit(1)

    os.makedirs(args.results_dir, exist_ok=True)

    analyze_benchmark_results(args.csv_paths, args.results_dir, scenario_map_dict)
