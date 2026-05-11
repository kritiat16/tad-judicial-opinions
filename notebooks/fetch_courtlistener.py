"""
notebooks/fetch_courtlistener.py

Pull opinion text for the 60 validation cases (30 splits × 2 sides) from
CourtListener's REST API.  Saves to data/raw/validation_opinions.json in the
same schema that preprocess.py consumes:

    id, date_filed, court, type, cluster_id, plain_text

Plus two carry-forward fields preserved for downstream joins:

    split_id, side

Opinion IDs are deterministic: op_split{NNN}_{A|B}  (e.g. op_split001_A).
Cluster IDs group each split's two opinions together: cl_split{NNN}.

Usage:
    python notebooks/fetch_courtlistener.py

Environment:
    COURTLISTENER_TOKEN  Required.  The CourtListener v4 REST API requires
                         authentication for all endpoints.  Get a free token
                         at courtlistener.com → Profile → API.
                         Run as: COURTLISTENER_TOKEN=xxx python notebooks/fetch_courtlistener.py

API note:
    The /opinions/{id}/ endpoint does not include date_filed or court directly;
    those fields live on the cluster object.  To keep the request count low,
    this script uses CSV metadata as the primary source:
        court      ← side_{a|b}_circuit mapped to the CL court slug
        date_filed ← side_{a|b}_year + "-01-01"
    If the opinion response happens to include these fields (some API versions
    do), they are used instead.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).parent.parent
SPLITS_CSV = REPO / "data" / "validation" / "circuit_splits.csv"
OUT_PATH = REPO / "data" / "raw" / "validation_opinions.json"
API_BASE = "https://www.courtlistener.com/api/rest/v4"

CIRCUIT_TO_CODE: dict[str, str] = {
    "First Circuit": "ca1",
    "Second Circuit": "ca2",
    "Third Circuit": "ca3",
    "Fourth Circuit": "ca4",
    "Fifth Circuit": "ca5",
    "Sixth Circuit": "ca6",
    "Seventh Circuit": "ca7",
    "Eighth Circuit": "ca8",
    "Ninth Circuit": "ca9",
    "Tenth Circuit": "ca10",
    "Eleventh Circuit": "ca11",
    "D.C. Circuit": "cadc",
    "Federal Circuit": "cafc",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_headers() -> dict:
    token = os.environ.get("COURTLISTENER_TOKEN", "").strip()
    if not token:
        print(
            "\n[ERROR] COURTLISTENER_TOKEN is not set.\n"
            "The CourtListener v4 API requires authentication for all endpoints.\n"
            "Get a free token at courtlistener.com → Profile → API, then run:\n"
            "  COURTLISTENER_TOKEN=your_token python notebooks/fetch_courtlistener.py\n"
        )
        sys.exit(1)
    return {"Authorization": f"Token {token}"}


def extract_court_slug(value: str) -> str:
    """'https://.../courts/ca4/' → 'ca4'"""
    m = re.search(r"/courts/([^/?]+)", value)
    return m.group(1) if m else ""


def html_to_text(html: str) -> str:
    """Strip HTML tags. Uses BeautifulSoup if available, else regex."""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser").get_text(separator="\n")
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"[ \t]{2,}", " ", text).strip()


def extract_text(data: dict) -> str:
    """Return the best available text from a CL opinion API response."""
    plain = data.get("plain_text", "").strip()
    if plain:
        return plain
    for field in ("html_with_citations", "html", "html_lawbox", "html_columbia"):
        raw = data.get(field, "").strip()
        if raw:
            return html_to_text(raw)
    return ""


def fetch_json(url: str, headers: dict) -> dict | None:
    """GET a JSON endpoint; retry once on 5xx, skip on 4xx."""
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if 400 <= resp.status_code < 500:
                print(f"HTTP {resp.status_code} — skipping (4xx, no retry)")
                return None
            if attempt == 0:
                print(f"HTTP {resp.status_code} — retrying in 10s")
                time.sleep(10)
            else:
                print(f"HTTP {resp.status_code} — giving up")
                return None
        except requests.RequestException as exc:
            if attempt == 0:
                print(f"request error: {exc} — retrying in 5s")
                time.sleep(5)
            else:
                print(f"request error: {exc} — giving up")
                return None
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    headers = get_headers()

    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    opinions: list[dict] = []
    n_attempted = n_ok = n_skipped = n_failed = 0
    fail_log: list[str] = []

    for row in rows:
        split_id = row["split_id"]
        # "split_001" → "001"
        num = split_id.split("_", 1)[1] if "_" in split_id else split_id

        sides = [
            (
                "A",
                row.get("side_a_cl_id", "").strip(),
                row.get("side_a_circuit", ""),
                row.get("side_a_year", ""),
            ),
            (
                "B",
                row.get("side_b_cl_id", "").strip(),
                row.get("side_b_circuit", ""),
                row.get("side_b_year", ""),
            ),
        ]

        for side, cl_id, circuit, year in sides:
            op_id = f"op_split{num}_{side}"

            if not cl_id:
                print(f"  [SKIP] {op_id}: no CL ID in CSV")
                n_skipped += 1
                continue

            n_attempted += 1
            url = f"{API_BASE}/opinions/{cl_id}/"
            print(f"  Fetching {op_id} (CL {cl_id})...", end=" ", flush=True)

            data = fetch_json(url, headers)
            time.sleep(0.5)

            if data is None:
                print("FAILED")
                n_failed += 1
                fail_log.append(f"{op_id}: HTTP fetch failed")
                continue

            text = extract_text(data)
            if not text:
                print("FAILED (no text in any field)")
                n_failed += 1
                fail_log.append(f"{op_id}: plain_text and all html fields empty")
                continue

            # court: from API if present, else map from CSV circuit name
            raw_court = data.get("court", "")
            if isinstance(raw_court, str) and "/courts/" in raw_court:
                court = extract_court_slug(raw_court)
            else:
                court = CIRCUIT_TO_CODE.get(circuit, "unknown")

            # date_filed: from API if present, else year-01-01 from CSV
            date_filed = data.get("date_filed") or f"{year}-01-01"

            opinions.append(
                {
                    "id": op_id,
                    "date_filed": date_filed,
                    "court": court,
                    "type": "majority",
                    "cluster_id": f"cl_split{num}",
                    "plain_text": text,
                    # carry-forward fields — ignored by preprocess.py but
                    # used by validate.py for joining back to the CSV
                    "split_id": split_id,
                    "side": side,
                }
            )
            print(f"OK  ({len(text):,} chars, court={court})")
            n_ok += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(opinions, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("Fetch summary")
    print("=" * 55)
    print(f"  Attempted  : {n_attempted}")
    print(f"  Successful : {n_ok}")
    print(f"  Skipped    : {n_skipped}  (no CL ID in CSV)")
    print(f"  Failed     : {n_failed}")
    if fail_log:
        print("  Failed list:")
        for entry in fail_log:
            print(f"    - {entry}")
    print(f"\n  Output     : {OUT_PATH}")
    print(f"  Records    : {n_ok} opinions written")
    print("=" * 55)


if __name__ == "__main__":
    main()
