from datetime import date

import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from inforce.config import Settings
from inforce.graph import connect
from inforce.graphrag import (
    anchor_lookups,
    as_of_range,
    retrieve,
    search_terms,
    validity,
    without_document_mentions,
)

TODAY = date(2026, 9, 17)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Which rule applied on 1 January 2025?", ("2025-01-01", "2025-01-01")),
        ("Which regulations applied as of June 2026?", ("2026-06-01", "2026-06-30")),
        ("What was the definition in 2020?", ("2020-01-01", "2020-12-31")),
        ("Which master circular is currently in force?", ("2026-09-17", "2026-09-17")),
        ("What replaced the SEBI (Mutual Funds) Regulations, 1996?", None),
    ],
)
def test_as_of_range(query, expected):
    result = as_of_range(query, TODAY)
    assert (result and (result["start"], result["end"])) == expected


@pytest.mark.parametrize(
    ("node", "start", "end", "state"),
    [
        ({"effective_from": "2026-04-01"}, "2026-06-01", "2026-06-30", "in_force"),
        (
            {"effective_from": "2026-04-01"},
            "2025-01-01",
            "2025-01-01",
            "not_yet_in_force",
        ),
        ({"effective_to": "2026-04-01"}, "2026-04-01", "2026-04-01", "ended"),
        ({"effective_from": "2026-04-01"}, "2026-04-01", "2026-04-01", "in_force"),
        (
            {"effective_from": "2021-07-01"},
            "2021-01-01",
            "2021-12-31",
            "changed_during_period",
        ),
        (
            {"status": "rescinded", "issue_date": "2017-10-06"},
            "2020-01-01",
            "2020-01-01",
            "ended_unknown_date",
        ),
        ({"issue_date": "2026-05-19"}, "2026-01-01", "2026-01-01", "not_yet_issued"),
        ({"effective_to": "2026-04-01"}, "2025-01-01", "2025-01-01", "unknown_start"),
    ],
)
def test_validity_uses_half_open_intervals_and_never_guesses(node, start, end, state):
    result = validity(node, start, end)
    assert result["state"] == state
    if state == "unknown_start":
        assert "effective_to 2026-04-01" in result["reason"]


def test_anchor_lookups_and_search_terms():
    ids, master_circular_dates = anchor_lookups(
        "Relationship between SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/629, the Master Circular for "
        "Mutual Funds dated March 20, 2026 and the mutual fund regulations 1996"
    )
    assert ids == ["cir:HOIMDIMDIDOF5PCIR2021629", "reg:SEBIMUTUALFUNDSREGULATIONS1996"]
    assert master_circular_dates == ["2026-03-20"]

    query = "What was the definition of group under the SEBI (Mutual Funds) Regulations, 1996?"
    assert (
        search_terms(without_document_mentions(query))
        == '"group means"^3 OR definition OR group'
    )


@pytest.fixture(scope="module")
def session():
    settings = Settings()
    try:
        driver = connect(settings)
        driver.verify_connectivity()
    except (ServiceUnavailable, Neo4jError):
        pytest.skip("Neo4j is not reachable; run `docker compose up -d`")
    with driver.session(database=settings.neo4j_database) as graph_session:
        if (
            graph_session.run("MATCH (c:Circular) RETURN count(c) AS n").single()["n"]
            == 0
        ):
            driver.close()
            pytest.skip("graph not built; run `python -m inforce.graph build`")
        yield graph_session
    driver.close()


def find(relationships, **expected):
    return [r for r in relationships if all(r.get(k) == v for k, v in expected.items())]


MC_2026 = "cir:HO24131112026IMDPOD1I76022026"
MC_2024 = "doc:MASTERCIRCULARFORMUTUALFUNDS:2024-06-27"
REGS_1996 = "reg:SEBIMUTUALFUNDSREGULATIONS1996"
REGS_2026 = "reg:SEBIMUTUALFUNDSREGULATIONS2026"


def test_which_circular_superseded(session):
    result = retrieve(
        session,
        "Which circular superseded the Master Circular for Mutual Funds dated June 27, 2024?",
        TODAY,
    )

    [superseded] = find(
        result["relationships"], type="SUPERSEDES", start=MC_2026, end=MC_2024
    )
    assert (superseded["hop"], superseded["source_doc"], superseded["page_start"]) == (
        1,
        "mf-master-circular-2026",
        2,
    )
    assert superseded["effective_from"] == "2026-04-01"
    assert {"doc_id": "mf-master-circular-2026"}.items() <= next(
        s for s in result["sources"] if s["doc_id"] == "mf-master-circular-2026"
    ).items()
    assert find(
        result["relationships"],
        type="RESCINDS",
        start=MC_2026,
        hop=2,
        end="cir:HOIMDIMDPOD1PCIR202536",
    )


def test_which_circular_amended(session):
    result = retrieve(
        session, "Which circular amended SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/553?", TODAY
    )

    [amended] = find(
        result["relationships"],
        type="AMENDS",
        end="cir:HOIMDIMDIDOF5PCIR2021553",
        hop=1,
    )
    assert amended["start"] == "cir:HOIMDIMDIDOF5PCIR2021629"
    assert (amended["source_doc"], amended["page_start"], amended["scope"]) == (
        "cir-2021-09-alignment-clarifications",
        1,
        "partial",
    )


def test_relationship_between_two_circulars(session):
    result = retrieve(
        session,
        "What is the relationship between SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/629 and the "
        "Master Circular for Mutual Funds dated March 20, 2026?",
        TODAY,
    )

    [path] = result["paths"]
    assert path["kind"] == "citation"
    assert [r["type"] for r in path["relationships"]] == ["REFERENCES", "PART_OF"]
    assert all(
        r["page_start"] for r in path["relationships"] if r["type"] == "REFERENCES"
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "Which SEBI Mutual Funds Regulations applied as of June 2026?",
            {REGS_2026: "in_force", REGS_1996: "ended"},
        ),
        (
            "Which mutual fund regulations were in force on 1 January 2025?",
            {REGS_2026: "not_yet_in_force", REGS_1996: "unknown_start"},
        ),
    ],
)
def test_regulation_in_force_at_date(session, query, expected):
    result = retrieve(session, query, TODAY)

    states = {t["id"]: t["state"] for t in result["temporal"]}
    assert {k: states[k] for k in expected} == expected
    assert result["conflicts"] == []


def test_definition_as_of_date_exposes_later_amendment_with_prior_wording(session):
    result = retrieve(
        session,
        "What was the definition of group under the SEBI (Mutual Funds) Regulations, 1996 in 2020?",
        TODAY,
    )

    top = result["provisions"][0]
    assert (top["page_start"], top["circular"]) == (10, REGS_1996)
    [amendment] = [a for a in top["amendments"] if a["footnote"] == "21"]
    assert amendment["effective_from"] == "2021-03-05" > result["as_of"]["end"]
    assert "Prior to its substitution" in amendment["evidence"]
