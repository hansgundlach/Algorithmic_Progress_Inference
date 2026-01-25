"""
GPQA-D Pareto Frontier with Model Names (ICML Style)

Publication-quality visualization showing:
- Pareto frontier with step function
- Model names on the left, colored by open/closed weight
- Non-overlapping labels with arrows for disambiguation
- Large fonts suitable for ICML dual-column format
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


def calculate_pareto_frontier(
    df: pd.DataFrame, price_col: str, error_col: str
) -> pd.DataFrame:
    """
    Calculate the Pareto frontier - points where no other point is both cheaper AND better.

    A point is on the Pareto frontier if it is NOT dominated by any other point.
    Point B dominates point A if:
    - B has lower or equal price AND lower or equal error
    - AND at least one is strictly better
    """
    if len(df) == 0:
        return df

    df_work = df.reset_index(drop=True).copy()
    pareto_indices = []

    for i in range(len(df_work)):
        is_dominated = False
        row_i = df_work.iloc[i]

        for j in range(len(df_work)):
            if i == j:
                continue

            row_j = df_work.iloc[j]

            # Point i is dominated if j is both cheaper-or-equal AND better-or-equal
            if (
                row_j[price_col] <= row_i[price_col]
                and row_j[error_col] <= row_i[error_col]
                and (
                    row_j[price_col] < row_i[price_col]
                    or row_j[error_col] < row_i[error_col]
                )
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_indices.append(i)

    # Return sorted by price
    pareto_df = df_work.loc[pareto_indices].sort_values(price_col).copy()
    return pareto_df


def clean_model_name(name: str) -> str:
    """Remove dates and clean up model names."""
    if pd.isna(name):
        return ""
    name = str(name)
    # Remove common date patterns
    name = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "", name)
    name = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", "", name)
    name = re.sub(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(
        r"\b\d{4}\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"\s+20\d{2}\s*$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[-_/]\s*$", "", name).strip()
    return name


def place_labels_no_overlap(
    labels_data: list, ax, fig, fontsize: float, min_gap_points: float = 18
) -> list:
    """
    Place labels on the left side of points, avoiding overlap.

    Uses a greedy algorithm to stack labels vertically when they would overlap.
    Returns list of (name, point_xy, label_xy, needs_arrow) tuples.
    """
    if not labels_data:
        return []

    # Sort by y position (error rate) - top to bottom
    labels_data = sorted(labels_data, key=lambda d: d["y"], reverse=True)

    # Get axis transform to convert between data and display coordinates
    transform = ax.transData
    inv_transform = ax.transData.inverted()

    # Calculate label positions in display coordinates
    placed = []

    for i, label_info in enumerate(labels_data):
        x_data = label_info["x"]
        y_data = label_info["y"]
        name = label_info["name"]
        color = label_info["color"]

        # Convert point to display coordinates
        point_display = transform.transform((x_data, y_data))

        # Initial label position: to the left of the point
        # Start with label at same y as point
        label_y_display = point_display[1]

        # Check for overlap with already placed labels
        # Labels need minimum vertical gap
        for placed_label in placed:
            placed_y = placed_label["label_y_display"]
            if abs(label_y_display - placed_y) < min_gap_points:
                # Overlap detected - move this label down
                if label_y_display >= placed_y:
                    label_y_display = placed_y - min_gap_points
                else:
                    # Already below, but too close - push further down
                    label_y_display = placed_y - min_gap_points

        # Check if label position is significantly different from point position
        needs_arrow = abs(label_y_display - point_display[1]) > 5

        placed.append(
            {
                "name": name,
                "x_data": x_data,
                "y_data": y_data,
                "color": color,
                "label_y_display": label_y_display,
                "point_display": point_display,
                "needs_arrow": needs_arrow,
            }
        )

    return placed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/price_reduction_models.csv")
    parser.add_argument("--out", default="../figures/gpqa_pareto_with_names.png")
    args = parser.parse_args()

    # =========================================================================
    # STYLE PARAMETERS (easily adjustable)
    # =========================================================================

    # Figure size for ICML dual-column
    FIG_WIDTH = 7.0
    FIG_HEIGHT = 6.0

    # Font sizes - large for ICML readability
    TICK_LABELSIZE = 12
    AXIS_LABELSIZE = 14
    TITLE_FONTSIZE = 16
    LEGEND_FONTSIZE = 11
    MODEL_NAME_FONTSIZE = 9  # Font size for model names

    # Point sizes
    POINT_SIZE_SCATTER = 80
    PARETO_POINT_SIZE = 100

    # Colors
    COLOR_OPEN = "#27AE60"  # Green for open weight
    COLOR_CLOSED = "#8E44AD"  # Purple for closed weight
    COLOR_FRONTIER = "#2C3E50"  # Dark blue for frontier line

    # =========================================================================
    # AXIS LIMITS (easily adjustable)
    # =========================================================================
    XMIN = 0.001 * 0.5
    XMAX = 100
    YMIN = 0.08
    YMAX = 0.90

    # Label placement parameters
    LABEL_X_OFFSET = -15  # Points to the left of data point
    MIN_LABEL_GAP = 22  # Minimum vertical gap between labels (in points)
    ARROW_COLOR_ALPHA = 0.5
    LABEL_FONTWEIGHT = 'heavy'  # Options: 'normal', 'bold', 'heavy', or numeric (400, 700, 900)

    # =========================================================================
    # LOAD AND PREPARE DATA
    # =========================================================================

    df = pd.read_csv(args.data)

    price_col = "Benchmark Cost USD"
    benchmark_col = "epoch_gpqa"

    # Parse benchmark scores
    df[benchmark_col] = df[benchmark_col].astype(str).str.replace("%", "").str.strip()
    df[benchmark_col] = pd.to_numeric(df[benchmark_col], errors="coerce")
    df["error_rate"] = 1 - (df[benchmark_col] / 100)

    # Parse prices
    df[price_col] = df[price_col].astype(str).str.replace("[$,]", "", regex=True)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Determine open weight status
    df["is_open_weight"] = df["License"].notna() & df["License"].str.contains(
        "open", case=False, na=False
    )

    # Drop rows with missing data
    df = df.dropna(subset=[price_col, "error_rate"])
    df = df[df[price_col] > 0].copy()

    print(f"Loaded {len(df)} models")

    # =========================================================================
    # SETUP FIGURE
    # =========================================================================

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],  # DejaVu has good bold support
            "font.size": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # =========================================================================
    # PLOT ALL SCATTER POINTS
    # =========================================================================

    # Open weight points
    df_open = df[df["is_open_weight"]]
    df_closed = df[~df["is_open_weight"]]

    if len(df_open) > 0:
        ax.scatter(
            df_open[price_col],
            df_open["error_rate"],
            c=COLOR_OPEN,
            s=POINT_SIZE_SCATTER,
            alpha=0.5,
            edgecolors="white",
            linewidth=0.8,
            label="Open Weight",
            zorder=3,
        )

    if len(df_closed) > 0:
        ax.scatter(
            df_closed[price_col],
            df_closed["error_rate"],
            c=COLOR_CLOSED,
            s=POINT_SIZE_SCATTER,
            alpha=0.5,
            edgecolors="white",
            linewidth=0.8,
            label="Closed Weight",
            zorder=3,
        )

    # =========================================================================
    # CALCULATE AND PLOT PARETO FRONTIER
    # =========================================================================

    pareto_df = calculate_pareto_frontier(df, price_col, "error_rate")
    print(f"Pareto frontier: {len(pareto_df)} models")

    if len(pareto_df) > 0:
        prices = pareto_df[price_col].values
        errors = pareto_df["error_rate"].values

        # Build step function with extensions
        step_x = []
        step_y = []

        # Vertical extension upward from leftmost point
        if errors[0] < YMAX:
            step_x.append(prices[0])
            step_y.append(YMAX)

        for j in range(len(prices)):
            if j == 0:
                step_x.append(prices[j])
                step_y.append(errors[j])
            else:
                # Horizontal then vertical
                step_x.append(prices[j])
                step_y.append(errors[j - 1])
                step_x.append(prices[j])
                step_y.append(errors[j])

        # Horizontal extension to the right
        if prices[-1] < XMAX:
            step_x.append(XMAX)
            step_y.append(errors[-1])

        # Plot frontier line
        ax.plot(
            step_x,
            step_y,
            color=COLOR_FRONTIER,
            linestyle="-",
            linewidth=2.5,
            label="Pareto Frontier",
            zorder=5,
            alpha=0.9,
        )

        # Plot Pareto points with color by open/closed
        for _, row in pareto_df.iterrows():
            point_color = COLOR_OPEN if row["is_open_weight"] else COLOR_CLOSED
            ax.scatter(
                [row[price_col]],
                [row["error_rate"]],
                c=point_color,
                s=PARETO_POINT_SIZE,
                marker="o",
                edgecolors="white",
                linewidth=1.5,
                zorder=6,
                alpha=1.0,
            )

    # =========================================================================
    # SET AXIS LIMITS AND SCALE (before label placement)
    # =========================================================================

    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)

    # Need to draw figure to get accurate transforms
    fig.canvas.draw()

    # =========================================================================
    # PLACE MODEL NAME LABELS WITH PROPER OVERLAP DETECTION
    # =========================================================================

    if len(pareto_df) > 0:
        # Collect label data
        labels_data = []
        for _, row in pareto_df.iterrows():
            clean_name = clean_model_name(row["Model"])
            if clean_name:
                labels_data.append(
                    {
                        "name": clean_name,
                        "x": row[price_col],
                        "y": row["error_rate"],
                        "color": COLOR_OPEN if row["is_open_weight"] else COLOR_CLOSED,
                    }
                )

        # Sort by error rate (y position) from top (high error) to bottom (low error)
        labels_data = sorted(labels_data, key=lambda d: d["y"], reverse=True)

        # STEP 1: Create temporary text objects to measure actual bounding boxes
        temp_texts = []
        for label_info in labels_data:
            # Create text at the data point position initially
            txt = ax.text(
                label_info["x"],
                label_info["y"],
                label_info["name"],
                fontsize=MODEL_NAME_FONTSIZE,
                fontweight=LABEL_FONTWEIGHT,
                ha="right",
                va="center",
                color=label_info["color"],
                transform=ax.transData,
            )
            temp_texts.append(txt)
            label_info["text_obj"] = txt

        # Force render to get accurate bounding boxes
        fig.canvas.draw()

        # STEP 2: Get bounding boxes in display coordinates
        renderer = fig.canvas.get_renderer()
        for i, label_info in enumerate(labels_data):
            bbox = label_info["text_obj"].get_window_extent(renderer=renderer)
            label_info["bbox_height"] = bbox.height
            label_info["bbox_width"] = bbox.width
            label_info["bbox"] = bbox
            # Initial label position in display coords (y center of text)
            point_display = ax.transData.transform((label_info["x"], label_info["y"]))
            label_info["label_y_display"] = point_display[1]
            label_info["point_x_display"] = point_display[0]
            label_info["point_y_display"] = point_display[1]

        # Remove temporary text objects
        for txt in temp_texts:
            txt.remove()

        # STEP 3: Place labels using a simple greedy stacking algorithm
        # Process labels from top to bottom, ensuring each is placed below all previous ones
        PADDING = 6  # Extra padding between labels in display units (pixels)

        def get_label_bottom(label_info):
            """Get bottom y position of label in display coords."""
            center_y = label_info["label_y_display"]
            half_height = label_info["bbox_height"] / 2
            return center_y - half_height

        # Process labels from top to bottom (already sorted by y descending)
        for i, label_info in enumerate(labels_data):
            if i == 0:
                # First label stays at its point position
                continue

            # Find the lowest bottom edge of all labels above this one
            lowest_bottom = float("inf")
            for j in range(i):
                bottom_j = get_label_bottom(labels_data[j])
                if bottom_j < lowest_bottom:
                    lowest_bottom = bottom_j

            # This label's top must be below that lowest bottom (with padding)
            half_height = label_info["bbox_height"] / 2
            required_center_y = lowest_bottom - PADDING - half_height

            # Only move label down if its current position would overlap
            if label_info["label_y_display"] + half_height > lowest_bottom - PADDING:
                label_info["label_y_display"] = required_center_y

        # Verify no overlaps remain
        def labels_overlap(a, b):
            top_a = a["label_y_display"] + a["bbox_height"] / 2
            bottom_a = a["label_y_display"] - a["bbox_height"] / 2
            top_b = b["label_y_display"] + b["bbox_height"] / 2
            bottom_b = b["label_y_display"] - b["bbox_height"] / 2
            return not (bottom_a >= top_b or bottom_b >= top_a)

        overlap_count = 0
        for i in range(len(labels_data)):
            for j in range(i + 1, len(labels_data)):
                if labels_overlap(labels_data[i], labels_data[j]):
                    overlap_count += 1

        if overlap_count > 0:
            print(f"  Warning: {overlap_count} label pairs still overlap")
        else:
            print(f"  All {len(labels_data)} labels placed without overlap")

        # STEP 4: Place the labels with arrows where displaced
        for label_info in labels_data:
            x_data = label_info["x"]
            y_data = label_info["y"]
            name = label_info["name"]
            color = label_info["color"]

            # Calculate displacement in display coordinates
            y_displacement_display = (
                label_info["label_y_display"] - label_info["point_y_display"]
            )

            # Convert label position back to data coordinates for proper placement
            label_display_pos = (
                label_info["point_x_display"] + LABEL_X_OFFSET,
                label_info["label_y_display"],
            )
            label_data_pos = ax.transData.inverted().transform(label_display_pos)
            label_y_data = label_data_pos[1]

            # Determine if we need an arrow (label displaced significantly from point)
            needs_arrow = abs(y_displacement_display) > 8

            if needs_arrow:
                # Use annotation with arrow connecting label to point
                ax.annotate(
                    name,
                    xy=(x_data, y_data),  # Arrow points to data point
                    xytext=(
                        label_data_pos[0],
                        label_y_data,
                    ),  # Label position in data coords
                    textcoords="data",
                    fontsize=MODEL_NAME_FONTSIZE,
                    fontweight=LABEL_FONTWEIGHT,
                    color=color,
                    ha="right",
                    va="center",
                    zorder=7,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=color,
                        alpha=ARROW_COLOR_ALPHA,
                        lw=0.8,
                        shrinkA=0,
                        shrinkB=3,
                        connectionstyle="arc3,rad=0.0",
                    ),
                )
            else:
                # Simple text annotation without arrow
                ax.text(
                    label_data_pos[0],
                    label_y_data,
                    name,
                    fontsize=MODEL_NAME_FONTSIZE,
                    fontweight=LABEL_FONTWEIGHT,
                    color=color,
                    ha="right",
                    va="center",
                    zorder=7,
                )

    # =========================================================================
    # FORMATTING
    # =========================================================================

    # Axis formatters
    def dollar_formatter(x, pos):
        if x >= 1:
            return f"${x:.0f}"
        elif x >= 0.01:
            return f"${x:.2f}"
        elif x >= 0.001:
            return f"${x:.3f}"
        else:
            return f"${x:.4f}"

    ax.xaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Labels
    ax.set_xlabel("Benchmark Cost (USD)", fontsize=AXIS_LABELSIZE, fontweight="bold")
    ax.set_ylabel("Error Rate", fontsize=AXIS_LABELSIZE, fontweight="bold")
    ax.set_title(
        "GPQA-D: Pareto Frontier with Model Names",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=12,
    )

    # Grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3, zorder=0)

    # Tick sizes
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("gray")
        spine.set_linewidth(0.5)

    # Legend
    ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="gray",
    )

    plt.tight_layout()

    # =========================================================================
    # SAVE
    # =========================================================================

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        args.out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Saved: {args.out}")

    pdf_out = args.out.replace(".png", ".pdf")
    plt.savefig(pdf_out, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved: {pdf_out}")

    plt.show()

    # Print pareto models
    print(f"\nPareto frontier models ({len(pareto_df)}):")
    for _, row in pareto_df.iterrows():
        status = "Open" if row["is_open_weight"] else "Closed"
        print(
            f"  {clean_model_name(row['Model'])}: ${row[price_col]:.3f}, {row['error_rate']*100:.1f}% error [{status}]"
        )


if __name__ == "__main__":
    main()
