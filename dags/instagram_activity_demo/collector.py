from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests


USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._]+$")
GRAPH_API_VERSION_PATTERN = re.compile(r"^v\d+\.\d+$")
DEFAULT_GRAPH_API_VERSION = "v23.0"


class CollectionConfigurationError(ValueError):
    """Raised when the live Graph API collector is not configured safely."""


def _graph_get_json(
    session: requests.Session,
    *,
    endpoint: str,
    access_token: str,
    params: dict[str, str | int],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call Graph API without placing the access token in the URL or errors."""
    try:
        response = session.get(
            endpoint,
            params=params,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "Meta Graph API request failed before receiving a response"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Meta Graph API returned invalid JSON (HTTP {response.status_code})"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError("Meta Graph API returned an unexpected response type")
    if not 200 <= response.status_code < 300:
        error_payload = payload.get("error")
        error_code = None
        error_type = None
        if isinstance(error_payload, dict):
            error_code = error_payload.get("code")
            error_type = error_payload.get("type")
        raise RuntimeError(
            "Meta Graph API request failed "
            f"(HTTP {response.status_code}, code={error_code!r}, "
            f"type={error_type!r})"
        )
    return payload


def _normalize_record(record: dict[str, Any]) -> dict[str, str | int]:
    """입력 출처와 관계없이 모델이 사용할 두 필드를 동일한 형식으로 맞춘다."""
    username = str(record.get("username", "")).strip()
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError(f"Invalid Instagram username: {username!r}")

    media_count = record.get("media_count")
    if isinstance(media_count, bool):
        raise ValueError(f"media_count must be an integer for {username}")

    try:
        normalized_count = int(media_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"media_count must be an integer for {username}"
        ) from error

    if normalized_count < 0:
        raise ValueError(f"media_count must be non-negative for {username}")

    # Deliberately retain only the two fields required by this demo.
    return {
        "username": username,
        "media_count": normalized_count,
    }


def load_fixture(fixture_path: Path) -> list[dict[str, str | int]]:
    """인증정보 없이 재현 가능한 mock Instagram 데이터를 읽는다."""
    with fixture_path.open(encoding="utf-8") as fixture_file:
        payload = json.load(fixture_file)

    if not isinstance(payload, list) or not payload:
        raise ValueError("The fixture must contain a non-empty JSON list")

    return [_normalize_record(record) for record in payload]


def collect_business_discovery(
    *,
    access_token: str,
    business_account_id: str,
    target_usernames: list[str],
    graph_api_version: str,
    timeout_seconds: int = 20,
) -> list[dict[str, str | int]]:
    """Collect username and media_count one target at a time.

    This function intentionally uses a regular for-loop. It does not create
    threads, processes, dynamic Airflow tasks, or concurrent HTTP requests.
    """
    if not access_token:
        raise CollectionConfigurationError("META_ACCESS_TOKEN is required")
    if not business_account_id.isdigit():
        raise CollectionConfigurationError(
            "META_IG_BUSINESS_ACCOUNT_ID must contain digits only"
        )
    if not GRAPH_API_VERSION_PATTERN.fullmatch(graph_api_version):
        raise CollectionConfigurationError(
            "META_GRAPH_API_VERSION must use the form 'v<major>.<minor>'"
        )
    if not target_usernames:
        raise CollectionConfigurationError(
            "META_TARGET_USERNAMES must contain at least one username"
        )

    endpoint = (
        f"https://graph.facebook.com/{graph_api_version}/"
        f"{business_account_id}"
    )
    collected: list[dict[str, str | int]] = []

    with requests.Session() as session:
        for username in target_usernames:
            normalized_username = username.strip()
            if not USERNAME_PATTERN.fullmatch(normalized_username):
                raise CollectionConfigurationError(
                    f"Invalid target username: {normalized_username!r}"
                )

            fields = (
                f"business_discovery.username({normalized_username})"
                "{username,media_count}"
            )
            payload = _graph_get_json(
                session,
                endpoint=endpoint,
                access_token=access_token,
                params={"fields": fields},
                timeout_seconds=timeout_seconds,
            )
            discovery = payload.get("business_discovery")
            if not isinstance(discovery, dict):
                error_payload = payload.get("error")
                raise RuntimeError(
                    f"Business Discovery response is missing for "
                    f"{normalized_username}; error_present="
                    f"{isinstance(error_payload, dict)}"
                )
            collected.append(_normalize_record(discovery))

    return collected


def collect_managed_professional_accounts(
    *,
    access_token: str,
    graph_api_version: str = DEFAULT_GRAPH_API_VERSION,
    timeout_seconds: int = 20,
    max_pages: int = 20,
    max_accounts: int = 100,
) -> list[dict[str, str | int]]:
    """Collect professional accounts connected to Pages managed by the token.

    The token must be a Facebook User Access Token with ``pages_show_list``
    and ``instagram_basic`` permissions. Only username and media_count leave
    this function; Page IDs and Instagram IDs remain in memory.
    """
    if not access_token:
        raise CollectionConfigurationError("META_ACCESS_TOKEN is required")
    if not GRAPH_API_VERSION_PATTERN.fullmatch(graph_api_version):
        raise CollectionConfigurationError(
            "META_GRAPH_API_VERSION must use the form 'v<major>.<minor>'"
        )

    graph_root = f"https://graph.facebook.com/{graph_api_version}"
    page_ids: list[str] = []
    seen_page_ids: set[str] = set()
    seen_cursors: set[str] = set()
    after_cursor: str | None = None

    with requests.Session() as session:
        for _ in range(max_pages):
            params: dict[str, str | int] = {
                "fields": "id",
                "limit": 100,
            }
            if after_cursor:
                params["after"] = after_cursor
            payload = _graph_get_json(
                session,
                endpoint=f"{graph_root}/me/accounts",
                access_token=access_token,
                params=params,
                timeout_seconds=timeout_seconds,
            )
            data = payload.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Meta /me/accounts response is missing data")
            for page in data:
                if not isinstance(page, dict):
                    continue
                page_id = str(page.get("id", ""))
                if page_id.isdigit() and page_id not in seen_page_ids:
                    page_ids.append(page_id)
                    seen_page_ids.add(page_id)

            paging = payload.get("paging")
            cursors = paging.get("cursors") if isinstance(paging, dict) else None
            next_cursor = (
                str(cursors.get("after", ""))
                if isinstance(cursors, dict)
                else ""
            )
            if not next_cursor:
                break
            if next_cursor in seen_cursors:
                raise RuntimeError("Meta pagination cursor repeated")
            seen_cursors.add(next_cursor)
            after_cursor = next_cursor
        else:
            raise RuntimeError("Meta Page pagination exceeded the safety limit")

        if not page_ids:
            raise CollectionConfigurationError(
                "The token does not expose a manageable Facebook Page"
            )

        instagram_ids: list[str] = []
        seen_instagram_ids: set[str] = set()
        for page_id in page_ids:
            payload = _graph_get_json(
                session,
                endpoint=f"{graph_root}/{page_id}",
                access_token=access_token,
                params={"fields": "instagram_business_account"},
                timeout_seconds=timeout_seconds,
            )
            account = payload.get("instagram_business_account")
            if not isinstance(account, dict):
                continue
            instagram_id = str(account.get("id", ""))
            if instagram_id.isdigit() and instagram_id not in seen_instagram_ids:
                instagram_ids.append(instagram_id)
                seen_instagram_ids.add(instagram_id)
            if len(instagram_ids) > max_accounts:
                raise RuntimeError(
                    "Connected Instagram accounts exceeded the safety limit"
                )

        if not instagram_ids:
            raise CollectionConfigurationError(
                "No Instagram Business or Creator Account is connected to "
                "the Pages visible to this token"
            )

        collected = []
        for instagram_id in instagram_ids:
            payload = _graph_get_json(
                session,
                endpoint=f"{graph_root}/{instagram_id}",
                access_token=access_token,
                params={"fields": "username,media_count"},
                timeout_seconds=timeout_seconds,
            )
            collected.append(_normalize_record(payload))
        return validate_records(collected)


def collect_from_environment(
    fixture_path: Path,
) -> list[dict[str, str | int]]:
    """환경변수에 따라 mock 또는 실제 Graph API 수집기를 선택한다."""
    mode = os.getenv("INSTAGRAM_DEMO_MODE", "fixture").strip().lower()
    if mode == "fixture":
        # 현재 데모의 기본값: 외부 API 호출 없이 저장소의 고정 데이터를 사용한다.
        return load_fixture(fixture_path)
    graph_api_version = os.getenv(
        "META_GRAPH_API_VERSION",
        DEFAULT_GRAPH_API_VERSION,
    )
    access_token = os.getenv("META_ACCESS_TOKEN", "")
    if mode == "graph_token_only":
        return collect_managed_professional_accounts(
            access_token=access_token,
            graph_api_version=graph_api_version,
        )
    if mode != "graph_api":
        raise CollectionConfigurationError(
            "INSTAGRAM_DEMO_MODE must be 'fixture', 'graph_token_only', "
            "or 'graph_api'"
        )

    usernames = [
        username.strip()
        for username in os.getenv("META_TARGET_USERNAMES", "").split(",")
        if username.strip()
    ]
    return collect_business_discovery(
        access_token=access_token,
        business_account_id=os.getenv("META_IG_BUSINESS_ACCOUNT_ID", ""),
        target_usernames=usernames,
        graph_api_version=graph_api_version,
    )


def validate_records(
    records: list[dict[str, Any]],
) -> list[dict[str, str | int]]:
    """빈 결과와 중복 사용자를 차단하고 모든 레코드를 정규화한다."""
    if not records:
        raise ValueError("No influencer records were collected")

    normalized = [_normalize_record(record) for record in records]
    usernames = [str(record["username"]) for record in normalized]
    if len(usernames) != len(set(usernames)):
        raise ValueError("Duplicate usernames were collected")
    return normalized
