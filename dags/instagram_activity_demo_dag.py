from __future__ import annotations

import csv
import json
import os
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pendulum
from airflow.sdk import dag, task

from instagram_activity_demo.collector import (
    collect_from_environment,
    validate_records,
)
from instagram_activity_demo.linear_model import (
    fit_linear_regression,
    predict_activity_scores,
)


# Git Bundle이 어느 임시 경로에 checkout되더라도 fixture를 찾을 수 있도록
# 현재 DAG 파일 위치를 기준으로 상대 경로를 계산한다.
DAG_DIRECTORY = Path(__file__).resolve().parent
FIXTURE_DIRECTORY = DAG_DIRECTORY / "instagram_activity_demo" / "fixtures"
DATA_ROOT = Path(
    os.getenv(
        "INSTAGRAM_DEMO_DATA_DIR",
        "/opt/airflow/data/instagram_activity_demo",
    )
)


def _write_json(path: Path, payload: object) -> None:
    """중간 파일을 임시 파일에 쓴 뒤 교체하여 불완전한 파일 생성을 막는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


@dag(
    dag_id="instagram_activity_linear_demo",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=["demo", "instagram", "sequential", "linear-model"],
)
def instagram_activity_linear_demo() -> None:
    # 네 작업은 모두 node_a Queue에서 한 번에 하나씩 순서대로 실행된다.
    @task(
        task_id="collect_name_and_post_count",
        queue="node_a",
        retries=2,
        retry_delay=timedelta(minutes=1),
    )
    def collect_name_and_post_count() -> str:
        # 실행마다 별도 디렉터리를 만들어 이전 실행 결과와 섞이지 않게 한다.
        run_directory = DATA_ROOT / uuid4().hex
        raw_path = run_directory / "raw_influencers.json"
        records = collect_from_environment(
            FIXTURE_DIRECTORY / "influencers.json"
        )
        _write_json(raw_path, records)
        return str(raw_path)

    @task(task_id="validate_and_store", queue="node_a")
    def validate_and_store(raw_path_value: str) -> str:
        # username 중복, 게시글 수 형식과 음수 여부를 검사한다.
        raw_path = Path(raw_path_value)
        with raw_path.open(encoding="utf-8") as raw_file:
            records = json.load(raw_file)
        validated_records = validate_records(records)
        validated_path = raw_path.parent / "validated_influencers.json"
        _write_json(validated_path, validated_records)
        return str(validated_path)

    @task(task_id="train_linear_model", queue="node_a")
    def train_linear_model(validated_path_value: str) -> str:
        # 학습 데이터는 fixture를 사용하지만 validated_path_value를 입력으로 받아
        # 수집 → 검증 → 학습 순서가 Airflow 의존성에 명확히 드러나게 한다.
        run_directory = Path(validated_path_value).parent
        model = fit_linear_regression(
            FIXTURE_DIRECTORY / "training.csv"
        )
        model_path = run_directory / "linear_model.json"
        _write_json(model_path, model)
        return str(model_path)

    @task(task_id="predict_activity_score", queue="node_a")
    def predict_activity_score(
        validated_path_value: str,
        model_path_value: str,
    ) -> str:
        # 검증된 사용자 데이터와 방금 저장한 모델을 다시 읽어 점수를 계산한다.
        validated_path = Path(validated_path_value)
        with validated_path.open(encoding="utf-8") as input_file:
            records = json.load(input_file)
        with Path(model_path_value).open(encoding="utf-8") as model_file:
            model = json.load(model_file)

        predictions = predict_activity_scores(records, model)
        result_path = validated_path.parent / "activity_scores.csv"
        with result_path.open("w", encoding="utf-8", newline="") as result_file:
            writer = csv.DictWriter(
                result_file,
                fieldnames=[
                    "username",
                    "media_count",
                    "demo_activity_score",
                ],
            )
            writer.writeheader()
            writer.writerows(predictions)
        return str(result_path)

    # TaskFlow API의 반환값 연결이 곧 네 작업의 순차 실행 그래프가 된다.
    raw_path = collect_name_and_post_count()
    validated_path = validate_and_store(raw_path)
    model_path = train_linear_model(validated_path)
    predict_activity_score(validated_path, model_path)


instagram_activity_linear_demo()
