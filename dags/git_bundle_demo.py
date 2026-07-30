from __future__ import annotations

import pendulum
from airflow.sdk import dag, task


@dag(
    dag_id="git_bundle_demo",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
    catchup=False,
    tags=["git-bundle-demo"],
)
def git_bundle_demo() -> None:
    @task(queue="node_a")
    def print_message() -> str:
        message = "GitDagBundle version 1"
        print(message)
        return message

    print_message()


git_bundle_demo()