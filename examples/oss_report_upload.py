#!/usr/bin/env python3
"""Test OSS report upload: upload, verify, overwrite, and cleanup.

Requires environment variables:
    OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_REGION, OSS_BUCKET
"""

import asyncio
import sys

import alibabacloud_oss_v2 as oss

from tradingscope.utils.oss_report_uploader import (
    _get_client,
    _upload_report,
    upload_report,
    upload_reports,
)

TRADE_DATE = "2026-01-01"
TICKER = "_TEST_"
PREFIX = f"tradingscope/{TRADE_DATE}/{TICKER}/"


def _read_object(client: oss.Client, bucket: str, key: str) -> str:
    """Read an object from OSS and return its content as string."""
    result = client.get_object(oss.GetObjectRequest(bucket=bucket, key=key))
    return result.body.content.decode("utf-8")


def _delete_object(client: oss.Client, bucket: str, key: str) -> None:
    """Delete an object from OSS."""
    client.delete_object(oss.DeleteObjectRequest(bucket=bucket, key=key))


def _object_exists(client: oss.Client, bucket: str, key: str) -> bool:
    """Check if an object exists in OSS."""
    try:
        client.head_object(oss.HeadObjectRequest(bucket=bucket, key=key))
        return True
    except Exception:
        return False


async def main():
    # Step 0: Verify OSS client can be initialized
    import tradingscope.utils.oss_report_uploader as m

    m._client = None  # reset cached client
    client = _get_client()
    if client is None:
        print("FAIL: OSS not configured. Set OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_REGION, OSS_BUCKET.")
        sys.exit(1)

    bucket = m._bucket_name
    print(f"OSS bucket: {bucket}")
    print(f"Test prefix: {PREFIX}")
    print()

    uploaded_keys = []

    try:
        # Test 1: Single upload (sync)
        print("=== Test 1: Single upload (sync) ===")
        content_v1 = "# Test Report V1\n\nThis is the first version."
        key = f"{PREFIX}test_agent.md"
        ok = _upload_report(TRADE_DATE, TICKER, "test_agent", content_v1)
        assert ok, "Upload should succeed"
        uploaded_keys.append(key)

        actual = _read_object(client, bucket, key)
        assert actual == content_v1, f"Content mismatch: {actual!r}"
        print(f"  PASS: uploaded and verified {key}")

        # Test 2: Overwrite existing file
        print("\n=== Test 2: Overwrite existing file ===")
        content_v2 = "# Test Report V2\n\nThis is the updated version."
        ok = _upload_report(TRADE_DATE, TICKER, "test_agent", content_v2)
        assert ok, "Overwrite should succeed"

        actual = _read_object(client, bucket, key)
        assert actual == content_v2, f"Content not overwritten: {actual!r}"
        print("  PASS: overwrite verified, content is V2")

        # Test 3: Async single upload
        print("\n=== Test 3: Async single upload ===")
        async_content = "# Async Report\n\nUploaded via async wrapper."
        ok = await upload_report(TRADE_DATE, TICKER, "async_agent", async_content)
        assert ok, "Async upload should succeed"
        async_key = f"{PREFIX}async_agent.md"
        uploaded_keys.append(async_key)

        actual = _read_object(client, bucket, async_key)
        assert actual == async_content
        print(f"  PASS: async upload verified {async_key}")

        # Test 4: Batch upload
        print("\n=== Test 4: Batch upload ===")
        reports = {
            "market_analyst": "# Market Analyst\n\nTechnical analysis.",
            "news_analyst": "# News Analyst\n\nNews analysis.",
            "fundamentals_analyst": "# Fundamentals\n\nFundamental analysis.",
        }
        results = await upload_reports(TRADE_DATE, TICKER, reports)
        assert all(results.values()), f"Some uploads failed: {results}"

        for name, content in reports.items():
            k = f"{PREFIX}{name}.md"
            uploaded_keys.append(k)
            actual = _read_object(client, bucket, k)
            assert actual == content, f"Content mismatch for {name}"
        print(f"  PASS: batch upload verified ({len(reports)} files)")

        # Test 5: Empty reports
        print("\n=== Test 5: Empty reports ===")
        results = await upload_reports(TRADE_DATE, TICKER, {})
        assert results == {}
        print("  PASS: empty reports returns empty dict")

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")

    finally:
        # Cleanup: delete all uploaded test objects
        print(f"\nCleaning up {len(uploaded_keys)} test objects...")
        for key in uploaded_keys:
            try:
                _delete_object(client, bucket, key)
                print(f"  Deleted: {key}")
            except Exception as e:
                print(f"  Failed to delete {key}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
