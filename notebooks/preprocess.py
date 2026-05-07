"""
Preprocess judicial opinions from data/raw/dummy_opinions.json.
Cleans text, links cluster members, and saves to data/processed/opinions_clean.json.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "dummy_opinions.json"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "opinions_clean.json"


# ── Compiled patterns ─────────────────────────────────────────────────────────

# Lines that look like case headers rather than legal reasoning:
# docket numbers, court names, judge lists, standalone dates, etc.
_HEADER_LINE_RE = re.compile(
    r"""
    ^\s*(
        No\.\s+[\d\w]+-[\d\w]+              # docket: No. 19-1234
        | IN\s+THE\s+                        # "In the United States Court..."
        | UNITED\s+STATES\s+COURT           # court name
        | FOR\s+THE\s+\w+\s+CIRCUIT        # "For the Ninth Circuit"
        | Argued\s+and\s+Submitted          # scheduling line
        | Before:\s+                        # judge list
        | \d{1,2}/\d{1,2}/\d{4}\s*$       # bare date
        | Filed:\s+
        | ORDER\s*$
        | OPINION\s*$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Standalone closing phrases that appear as their own paragraph
_STANDALONE_BOILERPLATE_RE = re.compile(
    r"""
    ^\s*(
        AFFIRMED\.?
        | REVERSED\.?
        | REVERSED\s+AND\s+REMANDED\.?
        | AFFIRMED\s+IN\s+PART[^.]*\.?
        | SO\s+ORDERED\.?
        | PER\s+CURIAM\.?
        | IT\s+IS\s+SO\s+ORDERED\.?
        | IT\s+IS\s+HEREBY\s+ORDERED[^.]*\.?
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Prefixes that introduce closing disposition sentences
_CLOSING_PREFIX_RE = re.compile(
    r"""
    ^(
        For\s+(the\s+)?foregoing\s+reasons
        | For\s+these\s+reasons
        | For\s+the\s+reasons\s+(stated|set\s+forth)\s+above
        | For\s+the\s+reasons\s+discussed
        | Accordingly
        | Therefore
        | We\s+therefore
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# All-caps disposition verbs (e.g. AFFIRM, REVERSED, REMANDED)
_DISPOSITION_RE = re.compile(
    r"\b(AFFIRM(?:ED)?|REVERS(?:ED)?|REMAND(?:ED)?|VACAT(?:ED)?|"
    r"GRANT(?:ED)?|DENI(?:ED|ES?)?)\b"
)

# Inline footnote markers: [1]  *3  ^2  fn1  fn 12
_FOOTNOTE_RE = [
    re.compile(r"\[\d{1,3}\]"),
    re.compile(r"(?<!\w)\*\d{1,3}\b"),
    re.compile(r"\^\d{1,3}\b"),
    re.compile(r"\bfn\s*\d{1,3}\b", re.IGNORECASE),
]

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# ── Per-step cleaning functions ───────────────────────────────────────────────

def _is_header_paragraph(para: str) -> bool:
    """True if para looks like a case header (docket, court, judge list, etc.)."""
    s = para.strip()
    if not s:
        return True
    # Short with no sentence-ending punctuation → likely a label or caption
    if len(s) < 80 and not re.search(r"[.!?]", s):
        return True
    if _HEADER_LINE_RE.match(s):
        return True
    return False


def _is_closing_disposition(para: str) -> bool:
    """True if para is a closing disposition paragraph to be removed."""
    s = para.strip()
    # Bare boilerplate word ("AFFIRMED.", "SO ORDERED.")
    if _STANDALONE_BOILERPLATE_RE.match(s):
        return True
    # Starts with a conventional closing phrase AND contains a disposition verb
    if _CLOSING_PREFIX_RE.match(s) and _DISPOSITION_RE.search(s):
        return True
    # Short paragraph whose only function is stating the outcome
    # (e.g. "The order of the district court is REVERSED." or "We AFFIRM...")
    if len(s) < 250 and _DISPOSITION_RE.search(s):
        return True
    return False


def strip_headers(text: str) -> str:
    """Drop leading paragraphs that precede the first substantive legal paragraph."""
    paras = text.split("\n\n")
    for i, para in enumerate(paras):
        if not _is_header_paragraph(para):
            return "\n\n".join(paras[i:])
    return text  # all paragraphs look like headers; return unchanged


def remove_boilerplate(text: str) -> str:
    """
    Remove:
      - Any standalone boilerplate paragraph throughout the text.
      - The final paragraph if it is a closing disposition sentence.
    """
    paras = text.split("\n\n")

    # Pass 1: drop standalone boilerplate ("AFFIRMED.", "SO ORDERED.") anywhere
    paras = [p for p in paras if not _STANDALONE_BOILERPLATE_RE.match(p.strip())]

    # Pass 2: drop the last paragraph if it reads as a closing disposition
    if paras and _is_closing_disposition(paras[-1]):
        paras.pop()

    return "\n\n".join(paras)


def remove_footnote_markers(text: str) -> str:
    """Strip inline footnote reference markers from the text."""
    for pattern in _FOOTNOTE_RE:
        text = pattern.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def fix_encoding(text: str) -> str:
    """
    Repair mojibake caused by UTF-8 bytes being misread as Latin-1/CP1252.
    Example: em dash U+2014 (bytes E2 80 94) misread as latin-1 produces
    the three-character sequence U+00E2 U+20AC U+201D; this reverses that.
    Returns the original string unchanged if it is not valid latin-1 or the
    re-decoded result is not valid UTF-8.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def clean(text: str) -> str:
    text = fix_encoding(text)
    text = strip_headers(text)
    text = remove_boilerplate(text)
    text = remove_footnote_markers(text)
    text = normalize_whitespace(text)
    return text


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(RAW_PATH, encoding="utf-8") as f:
        raw_opinions: list[dict] = json.load(f)

    # Index opinions by cluster so we can record companions
    clusters: dict[str, list[dict]] = defaultdict(list)
    for op in raw_opinions:
        clusters[op["cluster_id"]].append(op)

    # For each cluster: map opinion type → opinion id
    cluster_type_to_id: dict[str, dict[str, str]] = {
        cid: {m["type"]: m["id"] for m in members}
        for cid, members in clusters.items()
    }

    cleaned_opinions: list[dict] = []
    for op in raw_opinions:
        cid = op["cluster_id"]
        companions = {
            t: oid
            for t, oid in cluster_type_to_id[cid].items()
            if oid != op["id"]
        }
        cleaned_opinions.append(
            {
                "id": op["id"],
                "date_filed": op["date_filed"],
                "court": op["court"],
                "type": op["type"],
                "cluster_id": cid,
                "companion_ids": companions,  # {"dissent": "op_003"} or {}
                "clean_text": clean(op["plain_text"]),
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_opinions, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_opinions = len(cleaned_opinions)
    n_clusters = len(clusters)

    has_majority_and_dissent = sum(
        1
        for members in clusters.values()
        if any(m["type"] == "majority" for m in members)
        and any(m["type"] == "dissent" for m in members)
    )

    type_counts: dict[str, int] = defaultdict(int)
    for op in cleaned_opinions:
        type_counts[op["type"]] += 1

    print(f"Opinions processed : {n_opinions}")
    print(f"Clusters           : {n_clusters}")
    print(f"  majority + dissent: {has_majority_and_dissent}")
    print(f"Type breakdown     : {dict(type_counts)}")
    print(f"Output             : {OUT_PATH}")


if __name__ == "__main__":
    main()
