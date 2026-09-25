import json
import re
from pathlib import Path

import pymupdf
import pytest

from inforce.config import Settings
from inforce.ingest import RAW_DIR
from inforce.vector import (
    connect,
    hybrid_search,
    index,
    lexical_search,
    load_chunks,
    search,
)

REPRESENTATIVE_QUERIES = [
    (
        "What is the minimum and maximum NFO period for an open ended scheme?",
        "minimum period of 3 working days",
    ),
    (
        "How are large cap, mid cap and small cap companies defined?",
        "full market capitalization",
    ),
    (
        "What is the minimum investment threshold for a Specialized Investment Fund?",
        "ten lakh rupees",
    ),
    (
        "When do the SEBI Mutual Funds Regulations 2026 come into force?",
        "come into force with effect from April 01, 2026",
    ),
    (
        "Conditions for inter scheme transfer of securities between schemes",
        "Inter Scheme Transfer",
    ),
    ("Stress testing requirements for open ended debt schemes", "Stress Testing"),
    ("Revised format of the Monthly Cumulative Report", "Monthly Cumulative Report"),
    (
        "Which circulars are rescinded by the Master Circular for Mutual Funds?",
        "rescinded",
    ),
    (
        "Can an asset management company become a proprietary trading member in the debt segment?",
        "proprietary trading member",
    ),
]

TERMINOLOGY_QUERIES = [
    (
        "AMC investment in Corporate Debt Market Development Fund",
        "invest such percentage of assets under management",
    ),
    (
        "How often is the risk-o-meter evaluated?",
        "Risk-o-meter shall be evaluated on a monthly basis",
    ),
    ("fineness of physical gold held by gold ETFs", "995 parts per thousand"),
    ("Regulation 25(16B) skin in the game", "25 (16B)"),
]

DOCUMENTS = [
    {
        "doc_id": "master-circular",
        "file": "master-circular.pdf",
        "source_url": "https://www.sebi.gov.in/master-circular.pdf",
        "doc_type": "master_circular",
        "title": "Master Circular for Mutual Funds",
        "circular_number": "HO/24/13/11(1)2026-IMD-POD-1/I/7602/2026",
        "date": "2026-03-20",
        "effective_date": "2026-04-01",
        "page_count": 12,
        "chunks": [
            {
                "chunk_id": "master-circular-0000",
                "section": "1.7. New Fund Offer (NFO) Period",
                "page_start": 12,
                "page_end": 12,
                "text": "1.7.1. The NFO shall be open for a minimum period of 3 working days[^17] "
                "and not more than 15 calendar days.",
                "footnotes": {"17": "SEBI Circular No. SEBI/HO/IMD/DF2/CIR/P/2016/42"},
            },
            {
                "chunk_id": "master-circular-0001",
                "section": "13.19. Inter Scheme Transfer of Securities",
                "page_start": 229,
                "page_end": 230,
                "text": "13.19.1. Transfers of securities from one scheme to another scheme "
                "shall be done at the prevailing market price.",
                "footnotes": {},
            },
        ],
    },
    {
        "doc_id": "regulations",
        "file": "regulations.pdf",
        "source_url": "https://www.sebi.gov.in/regulations.pdf",
        "doc_type": "regulations",
        "title": "SEBI (Mutual Funds) Regulations, 2026",
        "circular_number": None,
        "date": "2026-01-14",
        "effective_date": "2026-04-01",
        "page_count": 207,
        "chunks": [
            {
                "chunk_id": "regulations-0000",
                "section": "49. Conditions for Specialized Investment Fund",
                "page_start": 159,
                "page_end": 159,
                "text": "(1) A Specialized Investment Fund shall not accept an investment amount "
                "of less than ten lakh rupees from an investor.",
                "footnotes": {},
            }
        ],
    },
]


@pytest.fixture
def store(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    for doc in DOCUMENTS:
        (processed / f"{doc['doc_id']}.json").write_text(json.dumps(doc))
    settings = Settings(
        qdrant_url=None, qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="test"
    )
    client = connect(settings)
    yield client, settings, list(load_chunks(processed))
    client.close()


def test_search_returns_ranked_chunks_with_provenance(store):
    client, settings, chunks = store
    assert index(client, settings, chunks) == 3

    hits = search(
        client, settings, "How long must a new fund offer stay open?", top_k=3
    )

    assert hits[0]["chunk_id"] == "master-circular-0000"
    assert hits[0]["score"] > hits[-1]["score"]
    assert {k: v for k, v in hits[0].items() if k != "score"} == {
        **{k: v for k, v in DOCUMENTS[0].items() if k not in ("chunks", "page_count")},
        **DOCUMENTS[0]["chunks"][0],
    }


def test_lexical_and_hybrid_search_keep_provenance_and_ranks(store):
    client, settings, chunks = store
    index(client, settings, chunks)

    lexical = lexical_search(client, settings, "ten lakh rupees", top_k=3)
    hybrid = hybrid_search(
        client, settings, "minimum amount an investor must put into a SIF", top_k=2
    )

    assert lexical[0]["chunk_id"] == "regulations-0000"
    assert hybrid[0]["chunk_id"] == "regulations-0000"
    assert len(hybrid) == 2
    assert hybrid[0]["score"] >= hybrid[1]["score"]
    assert {
        "dense_rank",
        "lexical_rank",
        "rrf_score",
        "page_start",
        "source_url",
        "footnotes",
    } <= hybrid[0].keys()
    assert hybrid[0]["dense_rank"] or hybrid[0]["lexical_rank"]


def test_reindexing_replaces_the_collection(store):
    client, settings, chunks = store
    index(client, settings, chunks)
    index(client, settings, chunks[:1])

    assert client.count(settings.qdrant_collection).count == 1


@pytest.fixture(scope="module")
def corpus_index():
    settings = Settings()
    if settings.qdrant_url or not Path(settings.qdrant_path).exists():
        pytest.skip("corpus index not built; run `python -m inforce.vector index`")
    client = connect(settings)
    if not client.collection_exists(settings.qdrant_collection):
        client.close()
        pytest.skip("corpus index not built; run `python -m inforce.vector index`")
    yield client, settings
    client.close()


@pytest.mark.parametrize(
    ("retrieve", "query", "phrase"),
    [(search, q, p) for q, p in REPRESENTATIVE_QUERIES]
    + [(hybrid_search, q, p) for q, p in REPRESENTATIVE_QUERIES + TERMINOLOGY_QUERIES],
)
def test_representative_query_finds_passage_on_cited_pdf_pages(
    corpus_index, retrieve, query, phrase
):
    client, settings = corpus_index

    hits = retrieve(client, settings, query, top_k=5)

    hit = next((h for h in hits if phrase.lower() in h["text"].lower()), None)
    assert hit, f"no top-5 chunk contains {phrase!r}: {[h['chunk_id'] for h in hits]}"
    assert hit["source_url"].startswith("https://www.sebi.gov.in/")
    with pymupdf.open(RAW_DIR / hit["file"]) as pdf:
        cited = " ".join(
            pdf[p - 1].get_text() for p in range(hit["page_start"], hit["page_end"] + 1)
        )
    assert phrase.lower() in re.sub(r"\s+", " ", cited).lower()
