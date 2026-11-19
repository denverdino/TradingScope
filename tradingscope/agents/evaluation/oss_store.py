"""Write-once storage of evaluation records in OSS or a local directory.

Records live at ``tradingscope-evaluations/<analysis_date>/<ticker>/<run_id>/h<horizon>.json``.
A record's existence marks that run and horizon as evaluated.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Iterable, Protocol
from uuid import uuid4

from tradingscope.default_config import DEFAULT_CONFIG

from .models import EvaluationRecord

logger = logging.getLogger(__name__)

EVALUATION_KEY_PREFIX = "tradingscope-evaluations"


def evaluation_key(analysis_date: str, ticker: str, run_id: str, horizon: int) -> str:
    from .market_outcome import validate_horizon

    date.fromisoformat(analysis_date)
    EvaluationRecord.safe_path_component(ticker)
    EvaluationRecord.safe_path_component(run_id)
    validate_horizon(horizon)
    return f"{EVALUATION_KEY_PREFIX}/{analysis_date}/{ticker}/{run_id}/h{horizon}.json"


def record_key(record: EvaluationRecord) -> str:
    return evaluation_key(record.analysis_date.isoformat(), record.ticker, record.run_id, record.horizon)


class EvaluationStore(Protocol):
    def exists(self, key: str) -> bool: ...

    def write(self, record: EvaluationRecord) -> None: ...

    def list_records(self, tickers: Iterable[str] | None = None) -> list[EvaluationRecord]: ...


class LocalEvaluationStore:
    """Filesystem store used when OSS is not configured and in offline tests."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root.resolve()):
            raise ValueError("evaluation key must remain inside the store directory")
        return path

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def write(self, record: EvaluationRecord) -> None:
        path = self._path(record_key(record))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(record.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def list_records(self, tickers: Iterable[str] | None = None) -> list[EvaluationRecord]:
        wanted = set(tickers) if tickers is not None else None
        records = []
        for path in sorted((self._root / EVALUATION_KEY_PREFIX).glob("*/*/*/h*.json")):
            if wanted is not None and path.parent.parent.name not in wanted:
                continue
            records.append(EvaluationRecord.model_validate_json(path.read_text(encoding="utf-8")))
        return records


class OSSEvaluationStore:
    """OSS store; writes forbid overwriting an existing record."""

    def __init__(self, client, bucket_name: str) -> None:
        self._client = client
        self._bucket = bucket_name

    def exists(self, key: str) -> bool:
        return bool(self._client.is_object_exist(self._bucket, key))

    def write(self, record: EvaluationRecord) -> None:
        import alibabacloud_oss_v2 as oss

        self._client.put_object(
            oss.PutObjectRequest(
                bucket=self._bucket,
                key=record_key(record),
                body=record.model_dump_json(indent=2).encode("utf-8"),
                content_type="application/json; charset=utf-8",
                forbid_overwrite=True,
            )
        )

    def list_records(self, tickers: Iterable[str] | None = None) -> list[EvaluationRecord]:
        import alibabacloud_oss_v2 as oss

        wanted = set(tickers) if tickers is not None else None
        request = oss.ListObjectsV2Request(bucket=self._bucket, prefix=f"{EVALUATION_KEY_PREFIX}/")
        records = []
        for page in self._client.list_objects_v2_paginator().iter_page(request):
            for item in page.contents or []:
                parts = item.key.split("/")
                if len(parts) != 5 or (wanted is not None and parts[2] not in wanted):
                    continue
                response = self._client.get_object(oss.GetObjectRequest(bucket=self._bucket, key=item.key))
                records.append(EvaluationRecord.model_validate_json(response.body.content))
        return records


def default_evaluation_store(results_dir: str | None = None) -> EvaluationStore:
    """Use OSS when configured, otherwise ``<results_dir>``."""
    from tradingscope.utils.oss_structured_output_uploader import _get_bucket_name, _get_client

    client = _get_client()
    if client is not None:
        return OSSEvaluationStore(client, _get_bucket_name())
    root = results_dir or DEFAULT_CONFIG["results_dir"]
    logger.info("[EvaluationStore] OSS not configured; using local evaluation records under %s", root)
    return LocalEvaluationStore(root)
