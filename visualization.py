"""
Visualization Module for Telco Customer Churn Prediction.
Modular helper functions for generating training curves, confusion matrix heatmaps,
and data distribution plots with theme-adaptive styling and crisp typography.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional


def plot_training_curves(history: Any) -> plt.Figure:
    """
    Plot Training vs Validation Accuracy and Training vs Validation Loss
    across epochs, replicating the Jupyter Notebook visualizations.
    """
    hist_dict = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist_dict.get("loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)

    # Accuracy Plot
    if "accuracy" in hist_dict:
        ax1.plot(epochs, hist_dict["accuracy"], label="Training Accuracy", color="#3b82f6", linewidth=2.2)
    if "val_accuracy" in hist_dict:
        ax1.plot(epochs, hist_dict["val_accuracy"], label="Validation Accuracy", color="#22c55e", linewidth=2.2, linestyle="--")

    ax1.set_title("Training vs Validation Accuracy", fontsize=12, fontweight="bold", pad=10, color="#2563eb")
    ax1.set_xlabel("Epoch", fontsize=10, color="#64748b")
    ax1.set_ylabel("Accuracy", fontsize=10, color="#64748b")
    ax1.tick_params(colors="#64748b", labelsize=9)
    ax1.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.35, color="#94a3b8")
    ax1.set_ylim(bottom=0.55, top=1.0)
    for spine in ax1.spines.values():
        spine.set_color("#cbd5e1")

    # Loss Plot
    if "loss" in hist_dict:
        ax2.plot(epochs, hist_dict["loss"], label="Training Loss", color="#f97316", linewidth=2.2)
    if "val_loss" in hist_dict:
        ax2.plot(epochs, hist_dict["val_loss"], label="Validation Loss", color="#ef4444", linewidth=2.2, linestyle="--")

    ax2.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold", pad=10, color="#ea580c")
    ax2.set_xlabel("Epoch", fontsize=10, color="#64748b")
    ax2.set_ylabel("Binary Cross-Entropy Loss", fontsize=10, color="#64748b")
    ax2.tick_params(colors="#64748b", labelsize=9)
    ax2.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.35, color="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#cbd5e1")

    plt.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: Tuple[str, str] = ("Retained (0)", "Churned (1)")
) -> plt.Figure:
    """
    Plot a compact, styled confusion matrix with clear labels and counts.
    Sized appropriately to avoid taking excessive space.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Matrix colormap
    cax = ax.matshow(cm, cmap="Blues", alpha=0.88)
    fig.colorbar(cax, fraction=0.046, pad=0.04, shrink=0.85)

    total = np.sum(cm)
    labels_grid = [
        ["TN", "FP"],
        ["FN", "TP"]
    ]

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100
            # Choose white text for dark cells, dark text for light cells
            text_color = "white" if count > np.max(cm) * 0.45 else "#0f172a"
            ax.text(
                j, i,
                f"{count:,}\n({pct:.1f}%)\n{labels_grid[i][j]}",
                ha="center", va="center",
                color=text_color, fontsize=9.5, fontweight="bold",
                linespacing=1.2
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=9.5, fontweight="bold", color="#1e293b")
    ax.set_yticklabels(class_names, fontsize=9.5, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold", labelpad=8, color="#334155")
    ax.set_ylabel("Actual Label", fontsize=10, fontweight="bold", labelpad=8, color="#334155")
    ax.set_title("Confusion Matrix (N=1,407)", fontsize=11, fontweight="bold", pad=12, color="#0f172a")

    for spine in ax.spines.values():
        spine.set_color("#94a3b8")

    plt.tight_layout()
    return fig


def plot_feature_distributions(df: pd.DataFrame) -> plt.Figure:
    """
    Plot distributions of input features (Tenure, MonthlyCharges, TotalCharges)
    broken down by Churn status.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    fig.patch.set_alpha(0.0)

    features = [
        ("tenure", "Tenure (Months)"),
        ("MonthlyCharges", "Monthly Charges ($)"),
        ("TotalCharges", "Total Charges ($)")
    ]

    churn_col = "Churn"
    has_churn = churn_col in df.columns

    for idx, (col, label) in enumerate(features):
        ax = axes[idx]
        ax.patch.set_alpha(0.0)
        if col in df.columns:
            if has_churn:
                no_churn = df[df[churn_col].isin(["No", 0, "0"])][col].dropna()
                yes_churn = df[df[churn_col].isin(["Yes", 1, "1"])][col].dropna()

                ax.hist(no_churn, bins=20, alpha=0.65, label="Retained", color="#2563eb", density=True)
                ax.hist(yes_churn, bins=20, alpha=0.65, label="Churned", color="#ef4444", density=True)
                ax.legend(fontsize=8.5, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
            else:
                ax.hist(df[col].dropna(), bins=20, color="#475569", alpha=0.7)

            ax.set_title(f"{label}", fontsize=10.5, fontweight="bold", color="#1e293b")
            ax.set_xlabel(label, fontsize=9, color="#64748b")
            ax.set_ylabel("Density", fontsize=9, color="#64748b")
            ax.tick_params(colors="#64748b", labelsize=8)
            ax.grid(True, linestyle=":", alpha=0.35, color="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1")

    plt.tight_layout()
    return fig


def plot_churn_donut(df: pd.DataFrame) -> plt.Figure:
    """
    Plot a clean, highly legible donut chart with high-contrast text and explicit legend.
    Displays counts and percentages clearly in both light and dark themes.
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    if "Churn" in df.columns:
        counts = df["Churn"].map({"Yes": "Churned", "No": "Retained", 1: "Churned", 0: "Retained"}).value_counts()
    else:
        counts = pd.Series({"Retained": 5163, "Churned": 1869})

    # Order: Retained first, Churned second
    retained_count = counts.get("Retained", 5163)
    churned_count = counts.get("Churned", 1869)
    total = retained_count + churned_count

    data_values = [retained_count, churned_count]
    labels = ["Retained (No)", "Churned (Yes)"]
    colors = ["#2563eb", "#ef4444"]

    # Donut pie slices
    wedges, texts, autotexts = ax.pie(
        data_values,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.72,
        wedgeprops=dict(width=0.45, edgecolor="#ffffff", linewidth=2.5)
    )

    # Style percentage text inside slices (crisp bold white)
    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontsize(11)
        autotext.set_weight("bold")

    # Center text inside the donut hole
    ax.text(
        0, 0,
        f"Total\n{total:,}\nRecords",
        ha="center", va="center",
        fontsize=7.5, fontweight="bold",
        color="#334155",
        linespacing=1.2
    )

    # Clean legend showing exact counts and percentages
    ret_pct = (retained_count / total) * 100
    chu_pct = (churned_count / total) * 100
    legend_labels = [
        f"Retained: {retained_count:,} ({ret_pct:.1f}%)",
        f"Churned: {churned_count:,} ({chu_pct:.1f}%)"
    ]
    ax.legend(
        wedges, legend_labels,
        title="Customer Status",
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=9,
        title_fontsize=9.5
    )

    ax.set_title("Telco Churn vs Retention Ratio", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
    plt.tight_layout()
    return fig
