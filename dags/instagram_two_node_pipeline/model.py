from __future__ import annotations

from typing import Iterable


def train_two_cluster_model(
    records: Iterable[dict[str, str | int | float]],
    *,
    batch_id: str,
    max_iterations: int = 50,
) -> dict[str, object]:
    """노드 B에서 활동량을 낮음·높음 두 군집으로 나누는 간이 모델을 학습한다."""
    rows = list(records)
    values = [float(record["log_media_count"]) for record in rows]
    if len(values) < 2 or min(values) == max(values):
        raise ValueError(
            "At least two distinct feature values are required for training"
        )

    # 최솟값과 최댓값을 초기 중심점으로 사용해 결과를 항상 재현 가능하게 한다.
    centroids = [min(values), max(values)]
    for _ in range(max_iterations):
        groups: list[list[float]] = [[], []]
        for value in values:
            index = min(
                range(2),
                key=lambda candidate: abs(value - centroids[candidate]),
            )
            groups[index].append(value)
        if not all(groups):
            raise ValueError("A cluster became empty during training")

        updated = [sum(group) / len(group) for group in groups]
        if max(
            abs(updated[index] - centroids[index]) for index in range(2)
        ) < 1e-9:
            centroids = updated
            break
        centroids = updated

    centroids = sorted(round(value, 8) for value in centroids)
    return {
        "model_type": "one_dimensional_kmeans",
        "feature": "log_media_count",
        "centroids": centroids,
        "decision_threshold": round(sum(centroids) / 2, 8),
        "training_batch_id": batch_id,
        "training_record_count": len(rows),
    }


def predict_activity_clusters(
    records: Iterable[dict[str, str | int | float]],
    model: dict[str, object],
) -> list[dict[str, str | int | float]]:
    """각 사용자를 가까운 중심점에 배정하고 읽기 쉬운 데모 등급을 붙인다."""
    centroids = [float(value) for value in model["centroids"]]
    if len(centroids) != 2:
        raise ValueError("The demo model must contain exactly two centroids")

    results: list[dict[str, str | int | float]] = []
    for record in records:
        value = float(record["log_media_count"])
        cluster = min(
            range(2),
            key=lambda candidate: abs(value - centroids[candidate]),
        )
        results.append(
            {
                "username": str(record["username"]),
                "media_count": int(record["media_count"]),
                "activity_cluster": cluster,
                "demo_grade": "LOW_ACTIVITY" if cluster == 0 else "HIGH_ACTIVITY",
            }
        )
    return results
