"""Shared calculations and figures for the public calibration report."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MODEL_ORDER = (
    "GPT-5.6 Luna",
    "GLM 5.3 Flash",
    "Kimi K3",
    "Grok 4.3",
    "TypeSafe Jev 1.13",
)
DIFFICULTY_ORDER = ("easy", "medium", "hard")
LABEL_ORDER = ("non_match", "match")
MODEL_COLORS = {
    "GPT-5.6 Luna": "#0072B2",
    "GLM 5.3 Flash": "#009E73",
    "Kimi K3": "#CC79A7",
    "Grok 4.3": "#D55E00",
    "TypeSafe Jev 1.13": "#E69F00",
}
LOG_LOSS_FLOOR = np.finfo(float).eps

PUBLIC_COLUMNS = {
    "case_number",
    "model",
    "order",
    "label",
    "difficulty",
    "probability_match",
    "probability_correct",
    "predicted_label",
    "correct",
    "extraction_method",
}


def load_predictions(path: Path | str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if set(data.columns) != PUBLIC_COLUMNS:
        raise ValueError("public prediction columns differ from the frozen report schema")
    if len(data) != 1_000:
        raise ValueError(f"expected 1,000 prediction rows, found {len(data)}")
    if set(data["model"]) != set(MODEL_ORDER):
        raise ValueError("public predictions do not contain the five expected models")
    if set(data["order"]) != {"published", "swapped"}:
        raise ValueError("public predictions must contain both record orders")
    if set(data["difficulty"]) != set(DIFFICULTY_ORDER):
        raise ValueError("public predictions contain unexpected difficulty values")
    if set(data["label"]) != set(LABEL_ORDER):
        raise ValueError("public predictions contain unexpected labels")
    if data["case_number"].nunique() != 100:
        raise ValueError("public predictions must contain 100 anonymous cases")
    if data.duplicated(["case_number", "model", "order"]).any():
        raise ValueError("public predictions contain duplicate model-case-order coordinates")
    for column in ("probability_match", "probability_correct"):
        if not data[column].between(0, 1).all():
            raise ValueError(f"{column} contains a value outside [0, 1]")
    if data["correct"].dtype != bool:
        mapped = data["correct"].astype(str).str.lower().map({"true": True, "false": False})
        if mapped.isna().any():
            raise ValueError("correct contains a non-boolean value")
        data["correct"] = mapped
    expected_probability_correct = np.where(
        data["label"] == "match",
        data["probability_match"],
        1 - data["probability_match"],
    )
    if not np.allclose(
        data["probability_correct"], expected_probability_correct, rtol=0, atol=1e-12
    ):
        raise ValueError("probability_correct differs from label and probability_match")
    expected_prediction = np.where(data["probability_match"] >= 0.5, "match", "non_match")
    if not np.array_equal(data["predicted_label"].to_numpy(), expected_prediction):
        raise ValueError("predicted_label differs from probability_match")
    expected_correct = expected_prediction == data["label"].to_numpy()
    if not np.array_equal(data["correct"].to_numpy(), expected_correct):
        raise ValueError("correct differs from predicted_label and label")
    metadata_counts = data.groupby(["case_number", "order"])[["label", "difficulty"]].nunique()
    if not (metadata_counts == 1).all().all():
        raise ValueError("case metadata differs across models")
    data["model"] = pd.Categorical(data["model"], MODEL_ORDER, ordered=True)
    data["difficulty"] = pd.Categorical(data["difficulty"], DIFFICULTY_ORDER, ordered=True)
    return data.sort_values(["case_number", "model", "order"]).reset_index(drop=True)


def published_predictions(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[data["order"] == "published"].copy()


def dataset_breakdown(data: pd.DataFrame) -> pd.DataFrame:
    cases = published_predictions(data).drop_duplicates("case_number")
    result = (
        cases.groupby(["difficulty", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=DIFFICULTY_ORDER, columns=LABEL_ORDER)
    )
    result.index.name = "Difficulty"
    return result.rename(columns={"non_match": "Non-match", "match": "Match"})


def fixed_width_reliability(values: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Reliability summary that keeps identical probabilities in the same bin."""

    if bins < 1:
        raise ValueError("bins must be at least one")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = values.assign(
        _bin=pd.cut(
            values["probability_match"],
            bins=edges,
            labels=False,
            include_lowest=True,
        )
    )
    output: list[dict[str, float | int]] = []
    z = 1.959963984540054
    for index, members in assigned.groupby("_bin", observed=True, sort=True):
        count = len(members)
        mean_probability = float(members["probability_match"].mean())
        observed_rate = float((members["label"] == "match").mean())
        denominator = 1 + z**2 / count
        center = (observed_rate + z**2 / (2 * count)) / denominator
        radius = (
            z
            * math.sqrt(observed_rate * (1 - observed_rate) / count + z**2 / (4 * count**2))
            / denominator
        )
        output.append(
            {
                "bin": int(index) + 1,
                "lower_edge": float(edges[int(index)]),
                "upper_edge": float(edges[int(index) + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "observed_match_rate": observed_rate,
                "ci_lower": max(0.0, center - radius),
                "ci_upper": min(1.0, center + radius),
                "gap": observed_rate - mean_probability,
            }
        )
    return pd.DataFrame(output)


def _log_loss_scores(correct_probability: np.ndarray) -> np.ndarray:
    """Natural-log loss with a machine-epsilon floor for exact zeroes."""

    return -np.log(np.clip(correct_probability, LOG_LOSS_FLOOR, 1.0))


def metric_table(
    data: pd.DataFrame,
    *,
    difficulty: str | None = None,
    order: str = "published",
) -> pd.DataFrame:
    selected = data.loc[data["order"] == order]
    if difficulty is not None:
        selected = selected.loc[selected["difficulty"] == difficulty]
    rows: list[dict[str, float | int | str]] = []
    for model in MODEL_ORDER:
        values = selected.loc[selected["model"] == model].sort_values("case_number")
        probability = values["probability_match"].to_numpy(dtype=float)
        target = (values["label"] == "match").to_numpy(dtype=int)
        correct_probability = np.where(target == 1, probability, 1 - probability)
        losses = _log_loss_scores(correct_probability)
        reliability = fixed_width_reliability(values)
        rows.append(
            {
                "Model": model,
                "N": len(values),
                "Accuracy": float(values["correct"].mean()),
                "Brier": float(np.mean((probability - target) ** 2)),
                "Log loss": float(np.mean(losses)),
                "ECE": float(np.average(reliability["gap"].abs(), weights=reliability["count"])),
            }
        )
    return pd.DataFrame(rows)


def difficulty_metric_table(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            metric_table(data, difficulty=value).assign(Difficulty=value)
            for value in DIFFICULTY_ORDER
        ],
        ignore_index=True,
    )[["Difficulty", "Model", "N", "Accuracy", "Brier", "Log loss", "ECE"]]


def difficulty_score_intervals(
    data: pd.DataFrame,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 20260919,
) -> pd.DataFrame:
    """Brier and log-loss means with deterministic case-bootstrap intervals."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least one")
    published = published_predictions(data)
    rng = np.random.default_rng(seed)
    output: list[dict[str, float | int | str]] = []
    for difficulty in DIFFICULTY_ORDER:
        for model in MODEL_ORDER:
            values = published.loc[
                (published["difficulty"] == difficulty) & (published["model"] == model)
            ].sort_values("case_number")
            probability = values["probability_match"].to_numpy(dtype=float)
            target = (values["label"] == "match").to_numpy(dtype=int)
            correct_probability = np.where(target == 1, probability, 1 - probability)
            per_case = {
                "Brier": (probability - target) ** 2,
                "Log loss": _log_loss_scores(correct_probability),
            }
            indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
            for metric, scores in per_case.items():
                bootstrap_means = scores[indices].mean(axis=1)
                lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
                output.append(
                    {
                        "Difficulty": difficulty,
                        "Model": model,
                        "Metric": metric,
                        "N": len(values),
                        "Score": float(scores.mean()),
                        "Lower": float(lower),
                        "Upper": float(upper),
                    }
                )
    return pd.DataFrame(output)


def difficulty_confidence_sd_table(data: pd.DataFrame) -> pd.DataFrame:
    """Sample SD of correct-label probability by endpoint and difficulty."""

    published = published_predictions(data)
    result = (
        published.groupby(["model", "difficulty"], observed=True)["probability_correct"]
        .std(ddof=1)
        .unstack()
        .reindex(index=MODEL_ORDER, columns=DIFFICULTY_ORDER)
    )
    result.index.name = "Model"
    return result.rename(columns=str.title).reset_index()


def hard_confidence_bands(data: pd.DataFrame) -> pd.DataFrame:
    hard = published_predictions(data)
    hard = hard.loc[hard["difficulty"] == "hard"]
    labels = ("<10%", "10–25%", "25–50%", "50–75%", "75–90%", "≥90%")

    def band(probability: float) -> str:
        if probability < 0.10:
            return labels[0]
        if probability < 0.25:
            return labels[1]
        if probability < 0.50:
            return labels[2]
        if probability < 0.75:
            return labels[3]
        if probability < 0.90:
            return labels[4]
        return labels[5]

    output = []
    for model in MODEL_ORDER:
        values = hard.loc[hard["model"] == model, "probability_correct"]
        counts = values.map(band).value_counts()
        output.append({"Model": model, **{label: int(counts.get(label, 0)) for label in labels}})
    return pd.DataFrame(output)


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#243447",
            "text.color": "#1F2933",
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, images_dir: Path, stem: str, *, square: bool = False) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    if square:
        fig.savefig(images_dir / f"{stem}.png", dpi=150)
    else:
        fig.savefig(images_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    if not square:
        fig.savefig(images_dir / f"{stem}.svg", bbox_inches="tight")


def plot_dataset_composition(data: pd.DataFrame, images_dir: Path) -> plt.Figure:
    _style()
    counts = dataset_breakdown(data)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(counts))
    width = 0.34
    colors = ("#56B4E9", "#E69F00")
    for offset, column, color in zip((-width / 2, width / 2), counts.columns, colors, strict=True):
        bars = ax.bar(x + offset, counts[column], width, label=column, color=color)
        ax.bar_label(bars, padding=3, fontsize=10, fontweight="bold")
    ax.set_xticks(x, [value.title() for value in counts.index])
    ax.set_ylim(0, 20)
    ax.set_ylabel("Number of restaurant pairs")
    ax.set_title("The benchmark is balanced across labels and difficulty")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, images_dir, "01_dataset_composition")
    return fig


def plot_overall_metrics(data: pd.DataFrame, images_dir: Path) -> plt.Figure:
    _style()
    scores = metric_table(data).set_index("Model").loc[list(MODEL_ORDER)]
    definitions = (
        ("Accuracy", "Accuracy ↑", (0.70, 0.95)),
        ("Brier", "Brier score ↓", (0.07, 0.25)),
        ("Log loss", "Natural-log loss ↓", (0.25, 1.40)),
        ("ECE", "5-band ECE ↓", (0.06, 0.26)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    y = np.arange(len(MODEL_ORDER))
    for ax, (metric, title, limits) in zip(axes.flat, definitions, strict=True):
        values = scores[metric].to_numpy()
        colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
        ax.hlines(y, limits[0], values, color="#D5DCE3", linewidth=2, zorder=1)
        ax.scatter(values, y, s=90, c=colors, edgecolor="white", linewidth=0.8, zorder=2)
        for index, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (value, index), xytext=(6, 0), textcoords="offset points")
        ax.set_xlim(*limits)
        ax.set_title(title)
        ax.set_yticks(y, MODEL_ORDER)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("No single endpoint wins every calibration metric", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, images_dir, "02_overall_metrics")
    return fig


def _strip_summary(
    ax: plt.Axes,
    values: pd.DataFrame,
    *,
    probability_column: str,
    seed: int,
    highlight_jev: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    for index, model in enumerate(MODEL_ORDER):
        probabilities = values.loc[values["model"] == model, probability_column].to_numpy()
        jitter = rng.uniform(-0.17, 0.17, len(probabilities))
        color = MODEL_COLORS[model]
        alpha = 0.92 if not highlight_jev or model == "TypeSafe Jev 1.13" else 0.34
        ax.scatter(
            probabilities,
            index + jitter,
            s=24 if not highlight_jev else 34,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=2,
        )
        lower, median, upper = np.quantile(probabilities, [0.25, 0.5, 0.75])
        ax.hlines(index, lower, upper, color="#172B3A", linewidth=4, zorder=3)
        ax.scatter([median], [index], marker="|", s=125, color="white", linewidth=2, zorder=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)


def plot_probability_distributions(data: pd.DataFrame, images_dir: Path) -> plt.Figure:
    _style()
    published = published_predictions(data)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6), sharex=True, sharey=True)
    for index, (ax, difficulty) in enumerate(zip(axes, DIFFICULTY_ORDER, strict=True)):
        values = published.loc[published["difficulty"] == difficulty]
        _strip_summary(ax, values, probability_column="probability_correct", seed=913 + index)
        ax.axvline(0.5, color="#52606D", linestyle="--", linewidth=1.2)
        ax.set_title(f"{difficulty.title()} (n={values['case_number'].nunique()})")
        ax.set_xlabel("Probability assigned to the correct label")
    axes[0].set_ylabel("")
    fig.suptitle(
        "Confidence becomes polarized as cases become harder",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Dots are cases; thick segments show the interquartile range and white ticks the median.",
        ha="center",
        fontsize=10,
        color="#52606D",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, images_dir, "03_probability_distributions_by_difficulty")
    return fig


def plot_hard_case_focus(data: pd.DataFrame, images_dir: Path) -> plt.Figure:
    _style()
    hard = published_predictions(data)
    hard = hard.loc[hard["difficulty"] == "hard"]
    fig, ax = plt.subplots(figsize=(8, 8))
    _strip_summary(
        ax,
        hard,
        probability_column="probability_correct",
        seed=20260919,
        highlight_jev=True,
    )
    jev_probability = hard.loc[hard["model"] == "TypeSafe Jev 1.13", "probability_correct"]
    jev_low = round(float(jev_probability.min()) * 100)
    jev_high = round(float(jev_probability.max()) * 100)
    ax.axvspan(0, 0.1, color="#D55E00", alpha=0.07)
    ax.axvspan(0.9, 1, color="#009E73", alpha=0.07)
    ax.axvline(0.5, color="#52606D", linestyle="--", linewidth=1.4)
    ax.set_xlabel("Probability assigned to the correct label", fontsize=12)
    fig.suptitle(
        "Hard cases expose brittle confidence",
        fontsize=20,
        fontweight="bold",
        x=0.12,
        y=0.96,
        ha="left",
    )
    fig.text(
        0.12,
        0.905,
        f"Jev stays between {jev_low}% and {jev_high}%; "
        "token-derived scores cluster near 0% and 100%.",
        fontsize=11,
        color="#52606D",
    )
    bands = hard_confidence_bands(data).set_index("Model")
    for index, model in enumerate(MODEL_ORDER):
        wrong = int(bands.loc[model, "<10%"])
        confident = int(bands.loc[model, "≥90%"])
        ax.text(
            1.015,
            index,
            f"{wrong} <10%  ·  {confident} ≥90%",
            va="center",
            fontsize=9,
            color="#36454F",
            transform=ax.get_yaxis_transform(),
        )
    ax.text(
        0,
        -0.13,
        "33 reviewed hard cases · 16 matches / 17 non-matches · published record order",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#52606D",
    )
    fig.tight_layout(rect=(0.04, 0.08, 0.86, 0.86))
    _save(fig, images_dir, "04_hard_case_focus", square=True)
    return fig


def plot_raw_match_probabilities(data: pd.DataFrame, images_dir: Path) -> plt.Figure:
    _style()
    published = published_predictions(data)
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13), sharex=True, sharey=True)
    for row, difficulty in enumerate(DIFFICULTY_ORDER):
        for column, label in enumerate(("match", "non_match")):
            ax = axes[row, column]
            values = published.loc[
                (published["difficulty"] == difficulty) & (published["label"] == label)
            ]
            _strip_summary(
                ax,
                values,
                probability_column="probability_match",
                seed=700 + row * 10 + column,
            )
            ax.axvline(0.5, color="#52606D", linestyle="--", linewidth=1.1)
            ax.set_title(f"{difficulty.title()} · true {label.replace('_', '-')}")
            ax.set_xlabel("Predicted p(match)")
    fig.suptitle(
        "Raw match probabilities reveal class-specific behavior",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, images_dir, "05_raw_match_probability_by_label")
    return fig


def _reliability_yerr(reliability: pd.DataFrame) -> np.ndarray:
    observed = reliability["observed_match_rate"].to_numpy(dtype=float)
    lower = reliability["ci_lower"].to_numpy(dtype=float)
    upper = reliability["ci_upper"].to_numpy(dtype=float)
    return np.maximum(0.0, np.vstack((observed - lower, upper - observed)))


def _annotate_reliability_counts(ax: plt.Axes, reliability: pd.DataFrame) -> None:
    for point in reliability.itertuples(index=False):
        near_right = point.mean_probability > 0.85
        near_top = point.observed_match_rate > 0.9
        ax.annotate(
            f"{point.count}",
            (point.mean_probability, point.observed_match_rate),
            xytext=(-4 if near_right else 4, -7 if near_top else 4),
            textcoords="offset points",
            fontsize=7,
            color="#36454F",
            ha="right" if near_right else "left",
            va="top" if near_top else "bottom",
        )


def _plot_reliability_panel(
    ax: plt.Axes,
    reliability: pd.DataFrame,
    *,
    color: str,
    marker_size: float,
    line_width: float,
) -> None:
    ax.plot([0, 1], [0, 1], color="#A0AEC0", linestyle="--", linewidth=1)
    ax.errorbar(
        reliability["mean_probability"].to_numpy(dtype=float),
        reliability["observed_match_rate"].to_numpy(dtype=float),
        yerr=_reliability_yerr(reliability),
        color=color,
        fmt="o",
        markersize=marker_size,
        capsize=2.5,
        linewidth=line_width,
    )
    _annotate_reliability_counts(ax, reliability)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")


def plot_reliability(data: pd.DataFrame, images_dir: Path) -> plt.Figure:
    _style()
    published = published_predictions(data)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.5), sharex=True, sharey=True)
    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        values = published.loc[published["model"] == model]
        reliability = fixed_width_reliability(values)
        _plot_reliability_panel(
            ax,
            reliability,
            color=MODEL_COLORS[model],
            marker_size=5.5,
            line_width=1.2,
        )
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Mean p(match)", fontsize=8)
    axes[0].set_ylabel("Observed match rate", fontsize=9)
    fig.suptitle(
        "Reliability by endpoint (five fixed probability bands)",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Occupied bands only · labels are case counts · vertical bars are 95% Wilson intervals",
        ha="center",
        fontsize=9,
        color="#52606D",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    _save(fig, images_dir, "06_reliability_by_model")
    return fig


def plot_difficulty_scores(data: pd.DataFrame, images_dir: Path) -> plt.Figure:
    _style()
    intervals = difficulty_score_intervals(data)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex="col", sharey=True)
    metric_limits = {
        metric: float(intervals.loc[intervals["Metric"] == metric, "Upper"].max()) * 1.22
        for metric in ("Brier", "Log loss")
    }
    y = np.arange(len(MODEL_ORDER))
    for row, difficulty in enumerate(DIFFICULTY_ORDER):
        for column, metric in enumerate(("Brier", "Log loss")):
            ax = axes[row, column]
            values = (
                intervals.loc[
                    (intervals["Difficulty"] == difficulty) & (intervals["Metric"] == metric)
                ]
                .set_index("Model")
                .loc[list(MODEL_ORDER)]
            )
            score = values["Score"].to_numpy()
            ax.errorbar(
                score,
                y,
                xerr=np.vstack((score - values["Lower"], values["Upper"] - score)),
                fmt="none",
                ecolor="#8996A3",
                elinewidth=1.7,
                capsize=3,
                zorder=1,
            )
            ax.scatter(
                score,
                y,
                s=70,
                c=[MODEL_COLORS[model] for model in MODEL_ORDER],
                edgecolor="white",
                linewidth=0.7,
                zorder=2,
            )
            for index, value in enumerate(score):
                ax.annotate(
                    f"{value:.3f}",
                    (value, index),
                    xytext=(5, -8),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xlim(0, metric_limits[metric])
            ax.set_yticks(y, MODEL_ORDER)
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            ax.spines[["top", "right", "left"]].set_visible(False)
            if row == 0:
                ax.set_title(f"{metric} ↓")
            if column == 0:
                case_count = int(values["N"].iloc[0])
                ax.set_ylabel(f"{difficulty.title()}\n(n={case_count})", fontweight="bold")
            if row == len(DIFFICULTY_ORDER) - 1:
                ax.set_xlabel("Mean score (lower is better)")
    fig.suptitle(
        "Probabilistic performance by difficulty",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Points are observed means; horizontal bars are 95% case-bootstrap intervals.",
        ha="center",
        fontsize=10,
        color="#52606D",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, images_dir, "07_probabilistic_scores_by_difficulty")
    return fig


def plot_reliability_by_difficulty(data: pd.DataFrame, images_dir: Path) -> plt.Figure:
    _style()
    published = published_predictions(data)
    fig, axes = plt.subplots(3, 5, figsize=(16, 10), sharex=True, sharey=True)
    for row, difficulty in enumerate(DIFFICULTY_ORDER):
        difficulty_values = published.loc[published["difficulty"] == difficulty]
        for column, model in enumerate(MODEL_ORDER):
            ax = axes[row, column]
            values = difficulty_values.loc[difficulty_values["model"] == model]
            reliability = fixed_width_reliability(values)
            _plot_reliability_panel(
                ax,
                reliability,
                color=MODEL_COLORS[model],
                marker_size=5,
                line_width=1.1,
            )
            if row == 0:
                ax.set_title(model, fontsize=10)
            if column == 0:
                ax.set_ylabel(
                    f"{difficulty.title()} (n={len(values)})\nObserved match rate",
                    fontsize=9,
                )
            if row == len(DIFFICULTY_ORDER) - 1:
                ax.set_xlabel("Mean predicted p(match)", fontsize=8)
    fig.suptitle(
        "Reliability by endpoint and difficulty",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Five fixed probability bands; ties stay together. "
        "Labels are counts; bars are 95% Wilson intervals.",
        ha="center",
        fontsize=10,
        color="#52606D",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(fig, images_dir, "08_reliability_by_difficulty")
    return fig


def generate_all_figures(data: pd.DataFrame, images_dir: Path) -> tuple[Path, ...]:
    figures = (
        (plot_dataset_composition, "01_dataset_composition", True),
        (plot_overall_metrics, "02_overall_metrics", True),
        (plot_probability_distributions, "03_probability_distributions_by_difficulty", True),
        (plot_hard_case_focus, "04_hard_case_focus", False),
        (plot_raw_match_probabilities, "05_raw_match_probability_by_label", True),
        (plot_reliability, "06_reliability_by_model", True),
        (plot_difficulty_scores, "07_probabilistic_scores_by_difficulty", True),
        (plot_reliability_by_difficulty, "08_reliability_by_difficulty", True),
    )
    for function, _, _ in figures:
        figure = function(data, images_dir)
        plt.close(figure)
    return tuple(
        images_dir / f"{stem}.{extension}"
        for _, stem, has_svg in figures
        for extension in (("png", "svg") if has_svg else ("png",))
    )
