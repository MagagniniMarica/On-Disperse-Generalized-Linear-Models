"""Reusable plotting utilities for P-CD experiment results.

The module reads the Excel files produced by the experiment runners and creates
the main plot families used in the project:

* accuracy/dispersion Pareto fronts;
* comparisons between two result sets;
* coefficient heatmaps;
* objective histories;
* relative-gap histories.


Expected result schemas
-----------------------
Accuracy maximization (problem ``A``):
    ``tau``, ``theta``, ``gamma``, ``SP_obj``

Dispersion maximization (problem ``D``):
    ``tau``, ``theta``, ``gamma_max``

Objective histories:
    ``time``, ``obj``; the filename must contain ``tau..._theta...`` and, for
    problem A, ``_gamma...``.

Coefficient files:
    files produced by ``save_results_coeff_xlsx``. Their names must contain
    ``tau..._theta..._gamma...`` and their first header should contain the
    number of new models as ``P1:<number>``. Alternatively, pass ``p_models``.

Quick start
-----------
A possible input layout is::

    results/
    |-- D_Obj_Lin_dsa.xlsx
    |-- histories/
    |   `-- D_objtime_HEUR_Lin_dsa_tau1.2_theta7.xlsx
    `-- coefficients/
        `-- D_Coeff_HEUR_Lin_dsa_tau1.2_theta7_gamma4.xlsx

From a terminal, create a Pareto-front plot with::

    python plot_results.py pareto --input results/D_Obj_Lin_dsa.xlsx --problem D --model Lin --dispersion dsa --output-dir plots

The equivalent Python call is::

    from plot_results import plot_pareto_front

    plot_pareto_front(
        "results/D_Obj_Lin_dsa.xlsx",
        problem="D",
        model="Lin",
        dispersion="dsa",
        output_dir="plots",
    )

Run ``python plot_results.py --help`` to see all five available commands.
"""

from __future__ import annotations

import argparse
import itertools
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}

NUMBER_PATTERN = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
INSTANCE_PATTERN = re.compile(
    rf"tau(?P<tau>{NUMBER_PATTERN})_theta(?P<theta>{NUMBER_PATTERN})"
    rf"(?:_gamma(?P<gamma>{NUMBER_PATTERN}))?",
    re.IGNORECASE,
)
P_COUNT_PATTERN = re.compile(r"P1:(?P<p>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CoefficientRecord:
    """One coefficient file and the instance metadata encoded in its name."""

    path: Path
    tau: float
    theta: int
    gamma: Optional[float]
    p_count: int
    frame: pd.DataFrame


def _problem_code(problem: str) -> str:
    code = problem.upper()
    if code not in {"A", "D"}:
        raise ValueError("problem must be 'A' or 'D'")
    return code


def _sense_name(sense: str) -> str:
    value = sense.lower()
    if value not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")
    return value


def _ensure_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _read_excel(path: Union[str, Path], required: Iterable[str] = ()) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_excel(source)
    _ensure_columns(frame, required, source)
    return frame


def _infer_problem(frame: pd.DataFrame) -> str:
    if {"tau", "theta", "gamma", "SP_obj"}.issubset(frame.columns):
        return "A"
    if {"tau", "theta", "gamma_max"}.issubset(frame.columns):
        return "D"
    raise ValueError(
        "Cannot infer the problem type. Expected either "
        "['tau', 'theta', 'gamma', 'SP_obj'] or "
        "['tau', 'theta', 'gamma_max']."
    )


def _metadata_from_name(path: Path, problem: str) -> tuple[float, int, Optional[float]]:
    match = INSTANCE_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot read tau/theta metadata from filename: {path.name}")

    tau = round(float(match.group("tau")), 6)
    theta = int(round(float(match.group("theta"))))
    gamma_text = match.group("gamma")
    gamma = round(float(gamma_text), 6) if gamma_text is not None else None

    if problem == "A" and gamma is None:
        raise ValueError(f"Problem A requires gamma in filename: {path.name}")
    return tau, theta, gamma


def _context_title(model: Optional[str], dispersion: Optional[str]) -> str:
    parts = [part for part in (model, dispersion) if part]
    return " | ".join(parts)


def _filter_accuracy_results(
    frame: pd.DataFrame, model: Optional[str]
) -> pd.DataFrame:
    """Apply the sentinel-value filter used by the original plotting code."""

    if model is None or model.lower() in {"lin", "linear", "linear regression"}:
        return frame
    return frame[frame["SP_obj"] < 1e9].copy()


def _dispersion_ticks(values: pd.Series, dispersion: Optional[str]) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if dispersion and dispersion.lower() == "dsa":
        start = int(np.ceil(minimum))
        stop = int(np.floor(maximum))
        return np.arange(start, stop + 1, dtype=int)
    return np.linspace(minimum, maximum, 5)


def _save_figure(
    fig: plt.Figure,
    output_dir: Optional[Union[str, Path]],
    filename: str,
    *,
    show: bool,
    close: bool,
) -> Optional[Path]:
    saved_path = None
    if output_dir is not None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_path = target_dir / filename
        fig.savefig(saved_path, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return saved_path


def plot_pareto_front(
    results_file: Union[str, Path],
    *,
    problem: Optional[str] = None,
    model: Optional[str] = None,
    dispersion: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = False,
) -> dict[Union[float, str], dict[str, Any]]:
    """Plot Pareto results stored in one objective-summary workbook.

    For problem A, one plot is created for each tau, with gamma on the x-axis
    and SP_obj on the y-axis. For problem D, a single plot has tau on the
    x-axis and gamma_max on the y-axis.
    """

    source = Path(results_file)
    frame = _read_excel(source)
    code = _problem_code(problem) if problem else _infer_problem(frame)
    required = (
        {"tau", "theta", "gamma", "SP_obj"}
        if code == "A"
        else {"tau", "theta", "gamma_max"}
    )
    _ensure_columns(frame, required, source)
    context = _context_title(model, dispersion)
    plots: dict[Union[float, str], dict[str, Any]] = {}

    if code == "A":
        groups = ((float(tau), data.copy()) for tau, data in frame.groupby("tau"))
    else:
        groups = (("all", frame.copy()),)

    for key, data in groups:
        if code == "A":
            data = _filter_accuracy_results(data, model)
        with plt.rc_context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=(10, 6))
            x_col, y_col = ("gamma", "SP_obj") if code == "A" else ("tau", "gamma_max")
            theta_values = sorted(data["theta"].unique())
            palette = sns.color_palette("tab10", n_colors=len(theta_values))
            colors = dict(zip(theta_values, palette))

            for theta, theta_data in data.groupby("theta", sort=True):
                theta_data = theta_data.sort_values(x_col)
                ax.plot(
                    theta_data[x_col],
                    theta_data[y_col],
                    marker="o",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    color=colors[theta],
                    linestyle="--",
                    linewidth=2,
                    label=rf"$\theta={theta}$",
                )

            ax.set_xlabel(r"$\gamma$" if code == "A" else r"$\tau$")
            ax.set_ylabel("Objective value" if code == "A" else r"$\gamma_{\max}$")
            title_parts = ["Pareto front"]
            if context:
                title_parts.append(context)
            if code == "A":
                title_parts.append(rf"$\tau={key:g}$")
            ax.set_title(" | ".join(title_parts))
            tick_values = data[x_col] if code == "A" else data[y_col]
            ticks = _dispersion_ticks(tick_values, dispersion)
            if code == "A":
                ax.set_xticks(ticks)
            else:
                ax.set_yticks(ticks)
            ax.grid(False)
            ax.legend(title=r"$\theta$", loc="best")
            fig.tight_layout()

        suffix = f"_tau{key:g}" if code == "A" else ""
        saved = _save_figure(
            fig,
            output_dir,
            f"pareto_{code}{suffix}.pdf",
            show=show,
            close=close,
        )
        plots[key] = {"fig": fig, "ax": ax, "data": data, "path": saved}

    return plots


def plot_pareto_comparison(
    first_file: Union[str, Path],
    second_file: Union[str, Path],
    *,
    labels: Sequence[str] = ("Method 1", "Method 2"),
    problem: Optional[str] = None,
    model: Optional[str] = None,
    dispersion: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = False,
) -> dict[Union[float, str], dict[str, Any]]:
    """Compare two objective-summary workbooks using a common plot."""

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two entries")

    paths = (Path(first_file), Path(second_file))
    frames = (_read_excel(paths[0]), _read_excel(paths[1]))
    code = _problem_code(problem) if problem else _infer_problem(frames[0])
    if _infer_problem(frames[1]) != code:
        raise ValueError("The two files describe different problem types")

    required = (
        {"tau", "theta", "gamma", "SP_obj"}
        if code == "A"
        else {"tau", "theta", "gamma_max"}
    )
    for frame, path in zip(frames, paths):
        _ensure_columns(frame, required, path)

    if code == "A":
        tau_values = sorted(set(frames[0]["tau"]).union(frames[1]["tau"]))
    else:
        tau_values = ["all"]

    context = _context_title(model, dispersion)
    linestyles = ("-", "--")
    markers = ("o", "s")
    plots: dict[Union[float, str], dict[str, Any]] = {}

    for tau in tau_values:
        with plt.rc_context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=(10, 6))
            plotted_data: dict[str, pd.DataFrame] = {}

            for index, (label, frame) in enumerate(zip(labels, frames)):
                data = frame[frame["tau"] == tau].copy() if code == "A" else frame.copy()
                if code == "A":
                    data = _filter_accuracy_results(data, model)
                plotted_data[label] = data
                x_col, y_col = ("gamma", "SP_obj") if code == "A" else ("tau", "gamma_max")
                theta_values = sorted(set(frames[0]["theta"]).union(frames[1]["theta"]))
                palette = sns.color_palette("tab10", n_colors=len(theta_values))
                colors = dict(zip(theta_values, palette))

                for theta, theta_data in data.groupby("theta", sort=True):
                    theta_data = theta_data.sort_values(x_col)
                    ax.plot(
                        theta_data[x_col],
                        theta_data[y_col],
                        linestyle=linestyles[index],
                        marker=markers[index],
                        markersize=9,
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        color=colors[theta],
                        linewidth=2,
                        label=rf"{label}, $\theta={theta}$",
                    )

            ax.set_xlabel(r"$\gamma$" if code == "A" else r"$\tau$")
            ax.set_ylabel("Objective value" if code == "A" else r"$\gamma_{\max}$")
            title_parts = ["Pareto comparison"]
            if context:
                title_parts.append(context)
            if code == "A":
                title_parts.append(rf"$\tau={float(tau):g}$")
            ax.set_title(" | ".join(title_parts))
            ax.grid(False)
            ax.legend(loc="best")
            fig.tight_layout()

        suffix = f"_tau{float(tau):g}" if code == "A" else ""
        saved = _save_figure(
            fig,
            output_dir,
            f"pareto_comparison_{code}{suffix}.pdf",
            show=show,
            close=close,
        )
        plots[tau] = {"fig": fig, "ax": ax, "data": plotted_data, "path": saved}

    return plots


def _load_histories(
    history_dir: Union[str, Path],
    problem: str,
) -> dict[tuple[float, ...], pd.DataFrame]:
    directory = Path(history_dir)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    histories: dict[tuple[float, ...], pd.DataFrame] = {}
    for path in sorted(directory.glob("*.xlsx")):
        try:
            tau, theta, gamma = _metadata_from_name(path, problem)
        except ValueError:
            continue

        frame = pd.read_excel(path)
        if not {"time", "obj"}.issubset(frame.columns):
            continue
        frame = frame[["time", "obj"]].dropna().sort_values("time").copy()
        if frame.empty:
            continue

        key = (tau, float(theta), float(gamma)) if problem == "A" else (tau, float(theta))
        if key in histories:
            warnings.warn(f"Duplicate history for {key}; using {path.name}")
        histories[key] = frame

    if not histories:
        raise FileNotFoundError(f"No compatible objective histories found in {directory}")
    return histories


def _extend_step_data(frame: pd.DataFrame, value_column: str, x_max: float) -> pd.DataFrame:
    extended = frame.copy()
    if float(extended["time"].iloc[-1]) < x_max:
        tail = pd.DataFrame(
            {"time": [x_max], value_column: [extended[value_column].iloc[-1]]}
        )
        extended = pd.concat([extended, tail], ignore_index=True)
    return extended


def plot_objective_histories(
    history_dir: Union[str, Path],
    *,
    problem: str,
    sense: Optional[str] = None,
    use_incumbent: bool = True,
    time_limit: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = False,
) -> dict[float, dict[str, Any]]:
    """Plot objective histories, grouped into one figure for each tau."""

    code = _problem_code(problem)
    objective_sense = _sense_name(sense or ("min" if code == "A" else "max"))
    histories = _load_histories(history_dir, code)
    by_tau: dict[float, dict[tuple[float, ...], pd.DataFrame]] = {}

    for key, frame in histories.items():
        data = frame.copy()
        if use_incumbent:
            data["objective"] = (
                data["obj"].cummin() if objective_sense == "min" else data["obj"].cummax()
            )
        else:
            data["objective"] = data["obj"]
        by_tau.setdefault(key[0], {})[key] = data[["time", "objective"]]

    plots: dict[float, dict[str, Any]] = {}
    for tau, tau_data in sorted(by_tau.items()):
        observed_max = max(float(frame["time"].max()) for frame in tau_data.values())
        base_time = float(time_limit) if time_limit is not None else observed_max
        x_max = base_time * 1.1 if base_time > 0 else 1.0
        extended = {
            key: _extend_step_data(frame, "objective", x_max)
            for key, frame in tau_data.items()
        }

        with plt.rc_context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=(12, 8))
            for key in sorted(extended, key=lambda item: item[1:]):
                data = extended[key]
                theta = int(key[1])
                label = rf"$\theta={theta}$"
                if code == "A":
                    label += rf", $\gamma={key[2]:g}$"
                if use_incumbent:
                    ax.step(data["time"], data["objective"], where="post", linewidth=2.5, label=label)
                else:
                    ax.plot(data["time"], data["objective"], linewidth=2, label=label)

            ax.set_xlabel("Time (s)")
            objective_label = (
                r"$-\mathcal{A}(\mathcal{B})$"
                if code == "A"
                else r"$\mathcal{D}(\mathcal{B})$"
            )
            ax.set_ylabel(objective_label)
            ax.set_title(rf"Objective evolution | $\tau={tau:g}$")
            ax.set_xlim(0, x_max)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            y_values = [
                float(value)
                for data in extended.values()
                for value in data["objective"].tolist()
            ]
            if y_values:
                y_min, y_max = min(y_values), max(y_values)
                spread = y_max - y_min
                if abs(spread) < 1e-12:
                    margin = 1.0 if abs(y_max) < 1e-12 else 0.05 * abs(y_max)
                else:
                    margin = 0.05 * spread
                lower = y_min if y_min == 0 else y_min - margin
                ax.set_ylim(lower, y_max + margin)
            fig.tight_layout()

        saved = _save_figure(
            fig,
            output_dir,
            f"objective_tau{tau:g}.pdf",
            show=show,
            close=close,
        )
        plots[tau] = {"fig": fig, "ax": ax, "data": extended, "path": saved}

    return plots


def plot_relative_gap_histories(
    reference_file: Union[str, Path],
    history_dir: Union[str, Path],
    *,
    problem: str,
    sense: Optional[str] = None,
    time_limit: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = False,
) -> dict[float, dict[str, Any]]:
    """Plot relative gaps between histories and reference objective values."""

    code = _problem_code(problem)
    objective_sense = _sense_name(sense or ("min" if code == "A" else "max"))
    required = (
        {"tau", "theta", "gamma", "SP_obj"}
        if code == "A"
        else {"tau", "theta", "gamma_max"}
    )
    reference = _read_excel(reference_file, required)
    histories = _load_histories(history_dir, code)

    reference_values: dict[tuple[float, ...], float] = {}
    for _, row in reference.iterrows():
        tau = round(float(row["tau"]), 6)
        theta = float(int(round(float(row["theta"]))))
        if code == "A":
            gamma = round(float(row["gamma"]), 6)
            reference_values[(tau, theta, gamma)] = float(row["SP_obj"])
        else:
            reference_values[(tau, theta)] = float(row["gamma_max"])

    by_tau: dict[float, dict[tuple[float, ...], pd.DataFrame]] = {}
    for key, history in histories.items():
        if key not in reference_values:
            continue
        optimum = reference_values[key]

        data = history.copy()
        if objective_sense == "min":
            data["relative_gap"] = (data["obj"] - optimum) / abs(optimum)
        elif abs(optimum) <= 1e-12:
            data["relative_gap"] = 0.0
        else:
            data["relative_gap"] = (optimum - data["obj"]) / abs(optimum)
        data["relative_gap"] = data["relative_gap"].clip(lower=0) * 100
        data_key = key if code == "A" else (key[0], key[1], optimum)
        by_tau.setdefault(key[0], {})[data_key] = data[["time", "relative_gap"]]

    if not by_tau:
        raise FileNotFoundError("No history matches the reference table")

    plots: dict[float, dict[str, Any]] = {}
    for tau, tau_data in sorted(by_tau.items()):
        observed_max = max(float(frame["time"].max()) for frame in tau_data.values())
        base_time = float(time_limit) if time_limit is not None else observed_max
        x_max = base_time * 1.1 if base_time > 0 else 1.0
        extended = {
            key: _extend_step_data(frame, "relative_gap", x_max)
            for key, frame in tau_data.items()
        }

        with plt.rc_context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=(12, 8))
            for key in sorted(extended, key=lambda item: item[1:]):
                data = extended[key]
                theta = int(key[1])
                label = rf"$\theta={theta}$"
                if code == "A":
                    label += rf", $\gamma={key[2]:g}$"
                ax.step(
                    data["time"],
                    data["relative_gap"],
                    where="post",
                    linewidth=2.5,
                    label=label,
                )

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Relative gap (%)")
            ax.set_title(rf"Relative gap evolution | $\tau={tau:g}$")
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, 101)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()

        saved = _save_figure(
            fig,
            output_dir,
            f"relative_gap_tau{tau:g}.pdf",
            show=show,
            close=close,
        )
        plots[tau] = {"fig": fig, "ax": ax, "data": extended, "path": saved}

    return plots


def _infer_p_count(frame: pd.DataFrame, path: Path, p_models: Optional[int]) -> int:
    if p_models is not None:
        if p_models <= 0:
            raise ValueError("p_models must be positive")
        return p_models
    first_header = str(frame.columns[0])
    match = P_COUNT_PATTERN.search(first_header)
    if match is None:
        raise ValueError(
            f"Cannot infer P from {path.name}; pass p_models explicitly"
        )
    return int(match.group("p"))


def _coefficient_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {str(frame.columns[0]), "Obj", "SP_obj", "gamma_max", "bias"}
    columns = [str(column) for column in frame.columns if str(column) not in excluded]
    if not columns:
        raise ValueError("No coefficient columns found")
    return columns


def _load_coefficient_records(
    coefficients_dir: Union[str, Path],
    problem: str,
    p_models: Optional[int],
) -> list[CoefficientRecord]:
    directory = Path(coefficients_dir)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    records: list[CoefficientRecord] = []
    for path in sorted(directory.glob("*.xlsx")):
        if "coeff" not in path.stem.lower():
            continue
        try:
            tau, theta, gamma = _metadata_from_name(path, problem)
        except ValueError as error:
            warnings.warn(str(error))
            continue
        frame = pd.read_excel(path)
        count = _infer_p_count(frame, path, p_models)
        if len(frame) < count:
            raise ValueError(f"{path.name} has fewer than P={count} model rows")
        records.append(CoefficientRecord(path, tau, theta, gamma, count, frame))

    if not records:
        raise FileNotFoundError(f"No compatible coefficient files found in {directory}")
    return records


def _align_block(block: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reorder a small block so its rows best match a reference block."""

    if block.shape != reference.shape or len(block) <= 1:
        return block
    if len(block) > 8:
        warnings.warn("Skipping row alignment because P > 8")
        return block

    best_order = tuple(range(len(block)))
    best_distance = float("inf")
    for order in itertools.permutations(range(len(block))):
        candidate = block[list(order)]
        distance = float(np.linalg.norm(reference - candidate, axis=1).sum())
        if distance < best_distance:
            best_distance = distance
            best_order = order
    return block[list(best_order)]


def _coefficient_groups(
    records: Sequence[CoefficientRecord],
    problem: str,
    gamma_order: str,
) -> dict[tuple[float, Optional[int]], list[CoefficientRecord]]:
    if gamma_order not in {"ascending", "descending"}:
        raise ValueError("gamma_order must be 'ascending' or 'descending'")

    if problem == "D":
        groups: dict[tuple[float, Optional[int]], list[CoefficientRecord]] = {}
        for record in records:
            groups.setdefault((record.tau, None), []).append(record)
        return groups

    by_tau_theta: dict[tuple[float, int], list[CoefficientRecord]] = {}
    for record in records:
        by_tau_theta.setdefault((record.tau, record.theta), []).append(record)

    reverse = gamma_order == "descending"
    groups = {}
    tau_values = sorted({record.tau for record in records})
    for tau in tau_values:
        theta_values = sorted({record.theta for record in records if record.tau == tau})
        ranked = {
            theta: sorted(
                by_tau_theta[(tau, theta)],
                key=lambda record: float(record.gamma),
                reverse=reverse,
            )
            for theta in theta_values
        }
        level_count = max(len(items) for items in ranked.values())
        for level in range(level_count):
            level_records = [items[level] for items in ranked.values() if level < len(items)]
            groups[(tau, level + 1)] = level_records
    return groups


def _infer_dispersion(records: Sequence[CoefficientRecord]) -> Optional[str]:
    known = ("dsa", "l1", "l2", "o1")
    found = {
        dispersion
        for record in records
        for dispersion in known
        if f"_{dispersion}_" in f"_{record.path.stem.lower()}_"
    }
    return next(iter(found)) if len(found) == 1 else None


def plot_coefficient_heatmaps(
    coefficients_dir: Union[str, Path],
    *,
    problem: str,
    p_models: Optional[int] = None,
    model: Optional[str] = None,
    dispersion: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = False,
) -> dict[tuple[float, Optional[int]], dict[str, Any]]:
    """Create coefficient heatmaps using only runner-produced workbooks.

    Problem D files are grouped by tau. Problem A may contain a different
    numerical gamma for every theta; therefore gamma values are ranked within
    each (tau, theta) group and heatmaps are created by rank. As in the old
    script, dsa gamma values are ordered from largest to smallest and the other
    dispersion notions from smallest to largest.
    """

    code = _problem_code(problem)
    records = _load_coefficient_records(coefficients_dir, code, p_models)
    resolved_dispersion = dispersion or _infer_dispersion(records)
    gamma_order = "descending" if resolved_dispersion == "dsa" else "ascending"
    groups = _coefficient_groups(records, code, gamma_order)
    plots: dict[tuple[float, Optional[int]], dict[str, Any]] = {}

    for key, group_records in sorted(groups.items()):
        group_records = sorted(group_records, key=lambda record: record.theta)
        feature_columns = _coefficient_columns(group_records[0].frame)
        blocks: list[np.ndarray] = []
        labels: list[str] = []
        centers: list[float] = []
        separators: list[int] = []
        current_position = 0

        for record in group_records:
            if _coefficient_columns(record.frame) != feature_columns:
                raise ValueError(f"Coefficient columns differ in {record.path.name}")
            block = (
                record.frame.loc[:, feature_columns]
                .iloc[: record.p_count]
                .apply(pd.to_numeric, errors="raise")
                .to_numpy(dtype=float)
            )
            if blocks:
                block = _align_block(block, blocks[0])
            blocks.append(block)
            centers.append(current_position + len(block) / 2)
            label = rf"$\theta={record.theta}$"
            if record.gamma is not None:
                label += "\n" + rf"$\gamma={record.gamma:g}$"
            labels.append(label)
            current_position += len(block)
            separators.append(current_position)

        first = group_records[0]
        if len(first.frame) > first.p_count:
            reference = (
                first.frame.loc[:, feature_columns]
                .iloc[first.p_count :]
                .apply(pd.to_numeric, errors="raise")
                .to_numpy(dtype=float)
            )
            if len(reference):
                blocks.append(reference)
                centers.append(current_position + len(reference) / 2)
                labels.append(r"$\mathcal{B}_0$")
                current_position += len(reference)
                separators.append(current_position)

        heatmap_data = np.vstack(blocks)
        vertical_layout = len(feature_columns) > 50

        with plt.rc_context(DEFAULT_STYLE):
            if vertical_layout:
                height = max(10, min(40, len(feature_columns) * 0.35))
                fig, ax = plt.subplots(figsize=(10, height))
                sns.heatmap(
                    heatmap_data.T,
                    cmap="PiYG",
                    center=0,
                    xticklabels=False,
                    yticklabels=feature_columns,
                    linewidths=0.5,
                    linecolor="gray",
                    ax=ax,
                )
                for separator in separators[:-1]:
                    ax.axvline(separator, color="black", linewidth=2)
                ax.set_xticks(centers)
                ax.set_xticklabels(labels, rotation=0)
                ax.set_xlabel("Models")
                ax.set_ylabel("Coefficients")
            else:
                width = max(10, min(24, len(feature_columns) * 0.7))
                fig, ax = plt.subplots(figsize=(width, 8))
                sns.heatmap(
                    heatmap_data,
                    cmap="PiYG",
                    center=0,
                    xticklabels=feature_columns,
                    yticklabels=False,
                    linewidths=0.5,
                    linecolor="gray",
                    ax=ax,
                )
                for separator in separators[:-1]:
                    ax.axhline(separator, color="black", linewidth=2)
                ax.set_yticks(centers)
                ax.set_yticklabels(labels, rotation=0)
                ax.tick_params(axis="x", rotation=90)
                ax.set_xlabel("Coefficients")
                ax.set_ylabel("Models")

            tau, level = key
            title = rf"Coefficient heatmap | $\tau={tau:g}$"
            context = _context_title(model, resolved_dispersion)
            if context:
                title += f" | {context}"
            if code == "A":
                title += f" | gamma rank {level} ({gamma_order})"
            ax.set_title(title)
            fig.tight_layout()

        level_suffix = f"_gamma_rank{level}" if level is not None else ""
        saved = _save_figure(
            fig,
            output_dir,
            f"coefficient_heatmap_tau{key[0]:g}{level_suffix}.pdf",
            show=show,
            close=close,
        )
        plots[key] = {
            "fig": fig,
            "ax": ax,
            "data": heatmap_data,
            "records": group_records,
            "path": saved,
        }

    return plots


def _add_common_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for generated plots (default: ./plots)",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without executing it."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python plot_results.py pareto --input results.xlsx --problem D
  python plot_results.py compare --first exact.xlsx --second heuristic.xlsx --problem A
  python plot_results.py heatmap --input-dir coefficients --problem D
  python plot_results.py objective --history-dir histories --problem A
  python plot_results.py gap --reference optimum.xlsx --history-dir histories --problem D
""",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pareto = commands.add_parser("pareto", help="Plot one objective-summary workbook")
    pareto.add_argument("--input", required=True, type=Path)
    pareto.add_argument("--problem", choices=("A", "D"))
    pareto.add_argument("--model")
    pareto.add_argument("--dispersion")
    _add_common_output_arguments(pareto)

    compare = commands.add_parser("compare", help="Compare two objective-summary workbooks")
    compare.add_argument("--first", required=True, type=Path)
    compare.add_argument("--second", required=True, type=Path)
    compare.add_argument("--labels", nargs=2, default=("Method 1", "Method 2"))
    compare.add_argument("--problem", choices=("A", "D"))
    compare.add_argument("--model")
    compare.add_argument("--dispersion")
    _add_common_output_arguments(compare)

    heatmap = commands.add_parser("heatmap", help="Plot coefficient workbooks")
    heatmap.add_argument("--input-dir", required=True, type=Path)
    heatmap.add_argument("--problem", required=True, choices=("A", "D"))
    heatmap.add_argument("--p-models", type=int)
    heatmap.add_argument("--model")
    heatmap.add_argument("--dispersion")
    _add_common_output_arguments(heatmap)

    objective = commands.add_parser("objective", help="Plot objective histories")
    objective.add_argument("--history-dir", required=True, type=Path)
    objective.add_argument("--problem", required=True, choices=("A", "D"))
    objective.add_argument("--sense", choices=("min", "max"))
    objective.add_argument("--raw", action="store_true", help="Do not compute incumbents")
    objective.add_argument("--time-limit", type=float)
    _add_common_output_arguments(objective)

    gap = commands.add_parser("gap", help="Plot relative-gap histories")
    gap.add_argument("--reference", required=True, type=Path)
    gap.add_argument("--history-dir", required=True, type=Path)
    gap.add_argument("--problem", required=True, choices=("A", "D"))
    gap.add_argument("--sense", choices=("min", "max"))
    gap.add_argument("--time-limit", type=float)
    _add_common_output_arguments(gap)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    common = {
        "output_dir": args.output_dir,
        "show": args.show,
        "close": not args.show,
    }

    if args.command == "pareto":
        plot_pareto_front(
            args.input,
            problem=args.problem,
            model=args.model,
            dispersion=args.dispersion,
            **common,
        )
    elif args.command == "compare":
        plot_pareto_comparison(
            args.first,
            args.second,
            labels=args.labels,
            problem=args.problem,
            model=args.model,
            dispersion=args.dispersion,
            **common,
        )
    elif args.command == "heatmap":
        plot_coefficient_heatmaps(
            args.input_dir,
            problem=args.problem,
            p_models=args.p_models,
            model=args.model,
            dispersion=args.dispersion,
            **common,
        )
    elif args.command == "objective":
        plot_objective_histories(
            args.history_dir,
            problem=args.problem,
            sense=args.sense,
            use_incumbent=not args.raw,
            time_limit=args.time_limit,
            **common,
        )
    elif args.command == "gap":
        plot_relative_gap_histories(
            args.reference,
            args.history_dir,
            problem=args.problem,
            sense=args.sense,
            time_limit=args.time_limit,
            **common,
        )
    else:  # pragma: no cover - argparse enforces a known command
        raise AssertionError(args.command)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
