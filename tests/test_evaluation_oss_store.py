"""Offline tests for evaluation record stores and completed-run ingestion."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tests.support.evaluation import EVALUATED_AT, completed_run, evaluation_record, flat_future, market_bars
from tests.test_main import _analysis_result
from tests.test_output_models import _all_outputs
from tradingscope.agents.evaluation.evaluator import build_evaluation_record
from tradingscope.agents.evaluation.oss_store import LocalEvaluationStore, OSSEvaluationStore, evaluation_key, record_key
from tradingscope.utils.oss_structured_output_reader import fetch_completed_run, latest_completed_session


def test_local_store_round_trip_and_write_once(tmp_path):
    store = LocalEvaluationStore(tmp_path)
    record = evaluation_record()
    assert not store.exists(record_key(record))
    store.write(record)
    assert store.exists(record_key(record))
    assert store.list_records() == [record]
    assert store.list_records(["MSFT"]) == []
    with pytest.raises(FileExistsError):
        store.write(record)


@pytest.mark.asyncio
async def test_local_completed_run_can_be_evaluated_end_to_end(tmp_path):
    from datetime import datetime, timezone

    from tests.support.evaluation import ANALYSIS_DATE, FakeMarketData
    from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator
    from tradingscope.utils import oss_structured_output_uploader as uploader

    fixed = datetime(2026, 7, 14, 0, 5, tzinfo=timezone.utc)
    with (
        patch.dict(uploader.DEFAULT_CONFIG, {"results_dir": str(tmp_path)}),
        patch.object(uploader, "_get_client", return_value=None),
        patch.object(uploader, "datetime", wraps=datetime) as clock,
    ):
        clock.now.return_value = fixed
        await uploader.persist_analysis_result(_analysis_result())
        run = fetch_completed_run(ANALYSIS_DATE.isoformat(), "AAPL", results_dir=str(tmp_path))
        assert len(run.outputs) == 7
        data = FakeMarketData(market_bars(flat_future([105.0] * 5)))
        evaluator = AnalysisEvaluator(LocalEvaluationStore(tmp_path), results_dir=str(tmp_path), market_data=data, clock=lambda: EVALUATED_AT)
        records = evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
        assert [record.horizon for record in records] == [1]
        assert evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat()) == []


def test_mutable_aliases_cannot_contaminate_the_run_bundle():
    manifest = completed_run().manifest.model_dump(mode="json")
    bundle = _analysis_result().model_dump(mode="json")
    alias = _all_outputs()[0].model_dump(mode="json")
    alias["decision"]["summary"] = "different process"
    data = {"manifest": manifest, "runs/run-1/full_report": bundle, "market_analyst": alias}
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", side_effect=lambda _date, _ticker, agent: data.get(agent)):
        run = fetch_completed_run("2026-07-14", "AAPL")
    assert run.outputs["market_analyst"].decision.summary != "different process"


def test_key_includes_run_id_and_horizon():
    assert evaluation_key("2026-07-14", "AAPL", "uuid", 1) == "tradingscope-evaluations/2026-07-14/AAPL/uuid/h1.json"
    for horizon in (3, 5):
        with pytest.raises(ValueError, match="allowed horizons are 1"):
            evaluation_key("2026-07-14", "AAPL", "uuid", horizon)


def test_oss_store_forbids_overwrite_and_reads_records():
    record = evaluation_record()
    client = Mock()
    client.is_object_exist.return_value = True
    store = OSSEvaluationStore(client, "bucket")
    store.write(record)
    request = client.put_object.call_args.args[0]
    assert request.forbid_overwrite is True
    assert request.key == record_key(record)
    assert store.exists(request.key)
    client.list_objects_v2_paginator.return_value.iter_page.return_value = [SimpleNamespace(contents=[SimpleNamespace(key=request.key)])]
    client.get_object.return_value = SimpleNamespace(body=SimpleNamespace(content=record.model_dump_json().encode()))
    assert store.list_records(["AAPL"]) == [record]
    assert store.list_records(["MSFT"]) == []


def test_completed_run_reads_all_available_valid_stages():
    manifest = completed_run().manifest.model_dump(mode="json")
    outputs = _all_outputs()
    data = {"manifest": manifest, "runs/run-1/full_report": _analysis_result().model_dump(mode="json")}
    data.update({output.agent_name: output.model_dump(mode="json") for output in outputs})
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", side_effect=lambda _date, _ticker, agent: data.get(agent)):
        run = fetch_completed_run("2026-07-14", "AAPL")
    assert len(run.outputs) == 7
    assert run.run_id == "run-1"


def test_changed_manifest_during_read_rejects_mixed_run():
    initial = completed_run().manifest.model_dump(mode="json")
    replacement = completed_run(run_id="replacement").manifest.model_dump(mode="json")
    stages = {"runs/run-1/full_report": _analysis_result().model_dump(mode="json")}
    manifests = iter([initial, replacement])
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=lambda _date, _ticker, agent: next(manifests) if agent == "manifest" else stages.get(agent),
    ):
        assert fetch_completed_run("2026-07-14", "AAPL") is None


def test_current_manifest_requires_all_seven_valid_stages():
    initial = completed_run().manifest.model_dump(mode="json")
    stages = {"runs/run-1/full_report": _analysis_result().model_dump(mode="json")}
    stages["runs/run-1/full_report"]["analysts"].pop("news")
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=lambda _date, _ticker, agent: initial if agent == "manifest" else stages.get(agent),
    ):
        assert fetch_completed_run("2026-07-14", "AAPL") is None


def test_wrong_ticker_stage_is_not_associated_with_another_run():
    initial = completed_run().manifest.model_dump(mode="json")
    stages = {"runs/run-1/full_report": _analysis_result().model_dump(mode="json")}
    stages["runs/run-1/full_report"]["analysts"]["news"]["ticker"] = "MSFT"
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=lambda _date, _ticker, agent: initial if agent == "manifest" else stages.get(agent),
    ):
        assert fetch_completed_run("2026-07-14", "AAPL") is None


def test_completed_run_requires_manifest_and_portfolio():
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", return_value=None):
        assert fetch_completed_run("2026-07-14", "AAPL") is None
    manifest = completed_run().manifest.model_dump(mode="json")
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=lambda _date, _ticker, agent: manifest if agent == "manifest" else None,
    ):
        assert fetch_completed_run("2026-07-14", "AAPL") is None


def test_latest_session_ignores_old_schema_manifest():
    manifest = {"schema_version": "3.0", "status": "complete", "ticker": "AAPL", "analysis_date": "2026-07-14"}
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", return_value=manifest):
        assert latest_completed_session("AAPL", date(2026, 7, 20), 7) is None


def test_latest_session_checks_same_date_and_prior_days():
    manifest = completed_run().manifest
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_completed_manifest", side_effect=lambda day, _ticker: manifest if day == "2026-07-14" else None
    ) as fetch:
        assert latest_completed_session("AAPL", date(2026, 7, 20), 7) == date(2026, 7, 13)
    assert fetch.call_count == 8
