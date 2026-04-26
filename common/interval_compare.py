from __future__ import annotations

import pandas as pd


LOWER_IS_BETTER_SUFFIXES = (
    "_to_vein_rate",
    "_to_artery_rate",
    "_error",
    "_loss",
)


def _higher_is_better(metric_name: str) -> bool:
    return not metric_name.endswith(LOWER_IS_BETTER_SUFFIXES)


def compare_interval_metrics(
    baseline_interval_csv: str,
    candidate_interval_csv: str,
) -> pd.DataFrame:
    """
    Сравнение по правилу из сообщения:
    - интервалы пересекаются => H0 не отвергаем;
    - интервалы не пересекаются => статистически значимое отличие;
    - направление improvement/degradation зависит от метрики.
    """
    base = pd.read_csv(baseline_interval_csv)
    cand = pd.read_csv(candidate_interval_csv)

    required = {"metric_name", "lower_bound", "average", "upper_bound"}
    if not required.issubset(base.columns) or not required.issubset(cand.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    merged = base.merge(cand, on="metric_name", suffixes=("_baseline", "_candidate"))
    rows = []

    for row in merged.to_dict("records"):
        name = row["metric_name"]
        b_low = row["lower_bound_baseline"]
        b_up = row["upper_bound_baseline"]
        c_low = row["lower_bound_candidate"]
        c_up = row["upper_bound_candidate"]

        overlap = not (c_low > b_up or b_low > c_up)
        higher_is_better = _higher_is_better(name)

        if overlap:
            verdict = "H0_not_rejected"
        elif higher_is_better and c_low > b_up:
            verdict = "significant_improvement"
        elif higher_is_better and c_up < b_low:
            verdict = "significant_degradation"
        elif (not higher_is_better) and c_up < b_low:
            verdict = "significant_improvement"
        elif (not higher_is_better) and c_low > b_up:
            verdict = "significant_degradation"
        else:
            verdict = "significant_difference"

        rows.append({
            "metric_name": name,
            "baseline_interval": f"({b_low:.6f}; {b_up:.6f})",
            "candidate_interval": f"({c_low:.6f}; {c_up:.6f})",
            "baseline_average": row["average_baseline"],
            "candidate_average": row["average_candidate"],
            "overlap": overlap,
            "verdict": verdict,
        })

    return pd.DataFrame(rows)
