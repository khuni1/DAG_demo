from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_record(record: dict[str, Any]) -> dict[str, str | int]:
    """mock 레코드를 파이프라인이 사용하는 username과 media_count로 정규화한다."""
    username = str(record.get("username", "")).strip()
    if not username:
        raise ValueError("username은 비어 있을 수 없습니다")

    media_count = record.get("media_count")
    if isinstance(media_count, bool):
        raise ValueError(f"media_count는 정수여야 합니다: {username}")
    try:
        normalized_count = int(media_count)
    except (TypeError, ValueError) as error:
        raise ValueError(f"media_count는 정수여야 합니다: {username}") from error
    if normalized_count < 0:
        raise ValueError(f"media_count는 음수일 수 없습니다: {username}")

    return {"username": username, "media_count": normalized_count}


def collect_fixture(fixture_path: Path) -> list[dict[str, str | int]]:
    """외부 인증정보 없이 재현 가능한 Instagram mock 데이터를 수집한다."""
    with fixture_path.open(encoding="utf-8") as fixture_file:
        payload = json.load(fixture_file)
    if not isinstance(payload, list) or not payload:
        raise ValueError("fixture는 비어 있지 않은 JSON 배열이어야 합니다")
    return validate_records(payload)


def validate_records(
    records: list[dict[str, Any]],
) -> list[dict[str, str | int]]:
    """빈 결과와 중복 username을 차단하고 레코드 형식을 통일한다."""
    if not records:
        raise ValueError("수집된 Instagram 레코드가 없습니다")
    normalized = [_normalize_record(record) for record in records]
    usernames = [str(record["username"]) for record in normalized]
    if len(usernames) != len(set(usernames)):
        raise ValueError("중복 username이 수집되었습니다")
    return normalized
