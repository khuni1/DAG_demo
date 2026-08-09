from __future__ import annotations

import csv
import json
import os
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pendulum
from airflow.sdk import dag, task

from instagram_two_node_pipeline.model import (
    predict_activity_clusters,
    train_two_cluster_model,
)
from instagram_two_node_pipeline.collector import collect_fixture
from instagram_two_node_pipeline.feature_store import (
    load_feature_batch,
    preprocess_records,
    store_feature_batch,
)


# Git Bundle이 checkout되는 실제 경로를 기준으로 fixture 위치를 계산한다.
DAG_DIRECTORY = Path(__file__).resolve().parent
FIXTURE_DIRECTORY = DAG_DIRECTORY / "fixtures"
NODE_A_DATA_ROOT = Path(
    os.getenv(
        "INSTAGRAM_DEMO_DATA_DIR",
        "/opt/airflow/data/instagram_two_node_demo",
    )
)
NODE_B_MODEL_ROOT = Path(
    os.getenv(
        "INSTAGRAM_DEMO_MODEL_DIR",
        "/opt/airflow/model-store/instagram_two_node_demo",
    )
)


def _write_json(path: Path, payload: object) -> None:
    """임시 파일을 완성한 뒤 교체하여 부분 저장된 모델 파일을 방지한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


@dag(
    dag_id="instagram_two_node_feature_model_demo",
    schedule="0 3 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=["demo", "instagram", "two-node", "feature-store", "model-store"],
)
def instagram_two_node_feature_model_demo() -> None:
    # queue 값이 Task를 실행할 물리 노드를 결정한다.
    # node_a: 수집·전처리, node_b: 모델 학습·추론
    @task(
        task_id="node_a_collect_name_and_post_count",
        queue="node_a",
        retries=2,
        retry_delay=timedelta(minutes=1),
    )
    def collect_on_node_a() -> str:
        # 노드 A에서 mock Instagram 이름과 게시글 수를 수집한다.
        run_directory = NODE_A_DATA_ROOT / uuid4().hex
        raw_path = run_directory / "raw_influencers.json"
        records = collect_fixture(FIXTURE_DIRECTORY / "influencers.json")
        _write_json(raw_path, records)
        return str(raw_path)

    @task(task_id="node_a_preprocess_and_store_features", queue="node_a")
    def preprocess_and_store_on_node_a(raw_path_value: str) -> str:
        # 게시글 수를 로그 변환하고 VM1 MySQL Feature Store에 저장한다.
        raw_path = Path(raw_path_value)
        with raw_path.open(encoding="utf-8") as raw_file:
            records = json.load(raw_file)

        features = preprocess_records(records)
        batch_id = raw_path.parent.name
        store_feature_batch(
            batch_id=batch_id,
            source_mode="fixture",
            records=features,
        )
        # 노드 사이에는 파일 대신 32자리 batch_id만 전달한다.
        return batch_id

    @task(task_id="node_b_train_and_store_model", queue="node_b")
    def train_and_store_on_node_b(batch_id: str) -> str:
        # 노드 B가 batch_id로 Feature Store를 조회하여 2개 군집을 학습한다.
        features = load_feature_batch(batch_id)
        model = train_two_cluster_model(features, batch_id=batch_id)
        model_path = NODE_B_MODEL_ROOT / batch_id / "cluster_model.json"
        _write_json(model_path, model)
        return str(model_path)

    @task(task_id="node_b_load_model_and_run_inference", queue="node_b")
    def infer_on_node_b(batch_id: str, model_path_value: str) -> str:
        # 저장한 모델을 다시 읽어 실제 운영의 모델 로딩 과정을 축소 재현한다.
        features = load_feature_batch(batch_id)
        model_path = Path(model_path_value)
        with model_path.open(encoding="utf-8") as model_file:
            model = json.load(model_file)

        predictions = predict_activity_clusters(features, model)
        result_path = model_path.parent / "inference_results.csv"
        with result_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "username",
                    "media_count",
                    "activity_cluster",
                    "demo_grade",
                ],
            )
            writer.writeheader()
            writer.writerows(predictions)
        return str(result_path)

    # max_active_tasks=1과 반환값 연결로 네 Task를 반드시 순차 실행한다.
    raw_path = collect_on_node_a()
    batch_id = preprocess_and_store_on_node_a(raw_path)
    model_path = train_and_store_on_node_b(batch_id)
    infer_on_node_b(batch_id, model_path)


instagram_two_node_feature_model_demo()
