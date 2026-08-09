from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def fit_linear_regression(training_path: Path) -> dict[str, float | str]:
    """최소제곱법으로 `점수 = 절편 + 계수 × 게시글 수` 모델을 학습한다."""
    post_counts: list[float] = []
    target_scores: list[float] = []

    with training_path.open(encoding="utf-8", newline="") as training_file:
        for row in csv.DictReader(training_file):
            post_counts.append(float(row["post_count"]))
            target_scores.append(float(row["activity_score"]))

    if len(post_counts) < 2:
        raise ValueError("At least two training rows are required")

    # 외부 ML 라이브러리 없이 평균과 공분산을 직접 계산하는 교육용 구현이다.
    mean_x = sum(post_counts) / len(post_counts)
    mean_y = sum(target_scores) / len(target_scores)
    denominator = sum((value - mean_x) ** 2 for value in post_counts)
    if denominator == 0:
        raise ValueError("Training post_count values must not all be equal")

    coefficient = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(post_counts, target_scores, strict=True)
    ) / denominator
    intercept = mean_y - coefficient * mean_x
    return {
        "model_type": "ordinary_least_squares",
        "feature": "media_count",
        "target": "demo_activity_score",
        "coefficient": round(coefficient, 8),
        "intercept": round(intercept, 8),
    }


def predict_activity_scores(
    records: Iterable[dict[str, str | int]],
    model: dict[str, float | str],
) -> list[dict[str, str | int | float]]:
    """게시글 수를 활동 점수로 변환하고 결과를 0~100 범위로 제한한다."""
    coefficient = float(model["coefficient"])
    intercept = float(model["intercept"])
    predictions: list[dict[str, str | int | float]] = []

    for record in records:
        media_count = int(record["media_count"])
        raw_score = intercept + coefficient * media_count
        predictions.append(
            {
                "username": str(record["username"]),
                "media_count": media_count,
                "demo_activity_score": round(
                    max(0.0, min(100.0, raw_score)), 2
                ),
            }
        )
    return predictions
