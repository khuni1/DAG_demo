from __future__ import annotations

import math
import os
from typing import Any

from instagram_two_node_pipeline.collector import validate_records


FEATURE_STORE_DATABASE = "instagram_feature_store"


def preprocess_records(
    records: list[dict[str, Any]],
) -> list[dict[str, str | int | float]]:
    """수집값을 검증하고 모델 입력인 log_media_count를 생성한다."""
    validated = validate_records(records)
    return [
        {
            "username": str(record["username"]),
            "media_count": int(record["media_count"]),
            "log_media_count": round(
                math.log1p(int(record["media_count"])),
                8,
            ),
        }
        for record in validated
    ]


def _connect():
    """노드 A의 별도 MySQL 논리 DB인 Feature Store에 연결한다."""
    try:
        import MySQLdb
    except ImportError as error:
        raise RuntimeError(
            "mysqlclient is required inside the Airflow worker image"
        ) from error

    host = os.getenv("FEATURE_STORE_HOST", os.getenv("VM1_IP", "mysql"))
    password = os.getenv("MYSQL_AIRFLOW_PASSWORD", "")
    if not password:
        raise RuntimeError("MYSQL_AIRFLOW_PASSWORD is required")

    return MySQLdb.connect(
        host=host,
        port=int(os.getenv("FEATURE_STORE_PORT", "3306")),
        user=os.getenv("FEATURE_STORE_USER", "airflow"),
        passwd=password,
        db=FEATURE_STORE_DATABASE,
        charset="utf8mb4",
    )


def ensure_schema() -> None:
    """재실행해도 안전한 방식으로 Batch·Feature 테이블을 준비한다."""
    connection = _connect()
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_batches (
                batch_id CHAR(32) PRIMARY KEY,
                source_mode VARCHAR(20) NOT NULL,
                status VARCHAR(20) NOT NULL,
                record_count INT UNSIGNED NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ready_at TIMESTAMP NULL
            ) ENGINE=InnoDB
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS influencer_features (
                batch_id CHAR(32) NOT NULL,
                username VARCHAR(64) NOT NULL,
                media_count BIGINT UNSIGNED NOT NULL,
                log_media_count DOUBLE NOT NULL,
                PRIMARY KEY (batch_id, username),
                CONSTRAINT fk_feature_batch
                    FOREIGN KEY (batch_id)
                    REFERENCES feature_batches(batch_id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB
            """
        )
        connection.commit()
    finally:
        connection.close()

def store_feature_batch(
    *,
    batch_id: str,
    source_mode: str,
    records: list[dict[str, str | int | float]],
) -> int:
    """Feature 묶음을 하나의 트랜잭션으로 저장한 뒤 READY로 전환한다."""
    if not records:
        raise ValueError("A feature batch cannot be empty")
    if len(batch_id) != 32 or any(
        character not in "0123456789abcdef" for character in batch_id
    ):
        raise ValueError("batch_id must be a 32-character lowercase hex value")

    ensure_schema()
    connection = _connect()
    try:
        cursor = connection.cursor()
        # 모든 행이 저장된 뒤에만 READY로 바꿔 노드 B의 부분 읽기를 막는다.
        cursor.execute(
            """
            INSERT INTO feature_batches
                (batch_id, source_mode, status, record_count, ready_at)
            VALUES (%s, %s, 'WRITING', %s, NULL)
            ON DUPLICATE KEY UPDATE
                source_mode = VALUES(source_mode),
                status = 'WRITING',
                record_count = VALUES(record_count),
                ready_at = NULL
            """,
            (batch_id, source_mode, len(records)),
        )
        cursor.execute(
            "DELETE FROM influencer_features WHERE batch_id = %s",
            (batch_id,),
        )
        cursor.executemany(
            """
            INSERT INTO influencer_features
                (batch_id, username, media_count, log_media_count)
            VALUES (%s, %s, %s, %s)
            """,
            [
                (
                    batch_id,
                    str(record["username"]),
                    int(record["media_count"]),
                    float(record["log_media_count"]),
                )
                for record in records
            ],
        )
        cursor.execute(
            """
            UPDATE feature_batches
            SET status = 'READY', ready_at = CURRENT_TIMESTAMP
            WHERE batch_id = %s
            """,
            (batch_id,),
        )
        connection.commit()
        return len(records)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_feature_batch(
    batch_id: str,
) -> list[dict[str, str | int | float]]:
    """READY 상태인 Batch만 읽고 선언된 행 수와 실제 행 수를 검증한다."""
    connection = _connect()
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT status, record_count
            FROM feature_batches
            WHERE batch_id = %s
            """,
            (batch_id,),
        )
        batch = cursor.fetchone()
        if batch is None:
            raise ValueError(f"Unknown feature batch: {batch_id}")
        if batch[0] != "READY":
            raise ValueError(f"Feature batch is not READY: {batch_id}")

        cursor.execute(
            """
            SELECT username, media_count, log_media_count
            FROM influencer_features
            WHERE batch_id = %s
            ORDER BY username
            """,
            (batch_id,),
        )
        records = [
            {
                "username": str(username),
                "media_count": int(media_count),
                "log_media_count": float(log_media_count),
            }
            for username, media_count, log_media_count in cursor.fetchall()
        ]
        if len(records) != int(batch[1]):
            raise RuntimeError(
                f"Feature batch row count mismatch for {batch_id}"
            )
        return records
    finally:
        connection.close()
