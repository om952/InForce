import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from inforce.config import Settings
from inforce.graph import (
    CHECKS,
    connect,
    extract,
    number_key,
    obligated_entities,
    validate,
)

MC_NUMBER = "HO/24/13/11(1)2026-IMD-POD-1/I/7602/2026"


def document(doc_id, doc_type, number, title, issued, effective, chunks):
    return {
        "doc_id": doc_id,
        "file": f"{doc_id}.pdf",
        "source_url": f"https://www.sebi.gov.in/{doc_id}.pdf",
        "doc_type": doc_type,
        "title": title,
        "circular_number": number,
        "date": issued,
        "effective_date": effective,
        "page_count": 20,
        "chunks": [
            {
                "chunk_id": f"{doc_id}-{i:04d}",
                "section": section,
                "page_start": page,
                "page_end": page,
                "text": text,
                "footnotes": footnotes,
            }
            for i, (page, section, text, footnotes) in enumerate(chunks)
        ],
    }


def edges(graph, rel):
    return {
        (start[1], end[1]): props
        for (r, start, end, _), props in graph["edges"].items()
        if r == rel
    }


MASTER_CIRCULAR = document(
    "mc-2026",
    "master_circular",
    MC_NUMBER,
    "Master Circular for Mutual Funds",
    "2026-03-20",
    "2026-04-01",
    [
        (
            1,
            "Subject: Master Circular for Mutual Funds",
            (
                "4. In addition, with the issuance of this Master Circular, the guidelines/directions "
                "contained in the circulars listed out at Sr. Nos.\n1 to 2 in the Appendix to this "
                "Master Circular, to the extent they relate to the Mutual Funds industry, shall stand "
                "rescinded.\n8. This Master Circular shall come into force with effect from April 01, "
                "2026. This Master Circular for Mutual Funds shall replace the Master Circular for "
                "Mutual Funds dated June 27, 2024."
            ),
            {},
        ),
        (
            5,
            "CHAPTER 1: OFFER DOCUMENT",
            "1.7.1. The AMC shall keep the NFO open for a minimum period of 3 working days[^17].",
            {
                "17": "SEBI Circular No. SEBI/HO/IMD/DF2/CIR/P/2016/42 dated March 18, 2016"
            },
        ),
        (
            6,
            "1.8. Timelines",
            "1.8.1. The timelines at Paragraph 1.7.1 of this Master Circular apply to ELSS[^18].",
            {
                "18": "SEBI Circular No. SEBI/HO/IMD/PoD2/P/CIR/2025/92 dated June 26, 2025, "
                "Refer SEBI letter No. SEBI/HO/OW/IMD/P/2022/511371/1 dated October 04, 2022"
            },
        ),
        (
            9,
            "22.11. Updating contact details",
            (
                "APPENDIX:LISTOFRESCINDEDCIRCULARS\nSR. NO. CIRCULAR NO. DATE SUBJECT\n"
                "1. HO/(92)2026-IMD-POD-2/I/6961/2026 March 13, 2026 Borrowing by Mutual Funds\n"
                "2. SEBI/HO/IMD/PoD2/P/CIR/2 025/92 June 26, 2025 Timelines for rebalancing\n"
                "3. SEBI/HO/IMD/DF3/CIR/P/2017/114 October 06, 2017 Categorization and "
                "Rationalization of Mutual Fund Schemes\nANNEXURES"
            ),
            {},
        ),
    ],
)


def test_number_key_merges_formatting_variants_only():
    assert number_key("SEBI/HO/IMD/PoD2/P/CIR/2 025/92") == number_key(
        "HO/IMD/PoD2/P/CIR/2025/92"
    )
    assert number_key("MFD/CIR/ No.14/442/2002") == number_key("MFD/CIR/14/442/2002")
    assert number_key("SEBI/IMD/CIR No.10/178129/09") == number_key(
        "SEBI/IMD/CIR/10/178129/09"
    )
    assert number_key("SEBI/HO/IMD/IMD-II DOF3/P/CIR/2022/39") != number_key(
        "SEBI/HO/IMD/IMD-II DF3/P/CIR/2022/39"
    )


def test_master_circular_lifecycle_references_and_obligations():
    graph = extract([MASTER_CIRCULAR])
    circulars = graph["circulars"]
    mc = f"cir:{number_key(MC_NUMBER)}"
    row1, row2, row3 = (
        f"cir:{number_key(n)}"
        for n in (
            "HO/(92)2026-IMD-POD-2/I/6961/2026",
            "SEBI/HO/IMD/PoD2/P/CIR/2025/92",
            "SEBI/HO/IMD/DF3/CIR/P/2017/114",
        )
    )
    previous_mc = "doc:MASTERCIRCULARFORMUTUALFUNDS:2024-06-27"

    rescinds = edges(graph, "RESCINDS")
    assert set(rescinds) == {(mc, row1), (mc, row2)}
    assert rescinds[(mc, row2)]["page_start"] == 1
    assert (circulars[row1]["status"], circulars[row1]["effective_to"]) == (
        "rescinded",
        "2026-04-01",
    )
    assert circulars[row3]["status"] == "rescinded"
    assert circulars[row3].get("effective_to") is None

    assert set(edges(graph, "SUPERSEDES")) == {(mc, previous_mc)}
    assert circulars[previous_mc]["status"] == "superseded"
    assert circulars[previous_mc]["effective_to"] == "2026-04-01"

    references = edges(graph, "REFERENCES")
    footnote = references[
        ("mc-2026-0001", f"cir:{number_key('SEBI/HO/IMD/DF2/CIR/P/2016/42')}")
    ]
    assert (footnote["via"], footnote["page_start"]) == ("footnote 17", 5)
    assert (
        circulars[f"cir:{number_key('SEBI/HO/IMD/DF2/CIR/P/2016/42')}"]["issue_date"]
        == "2016-03-18"
    )
    assert ("mc-2026-0002", row2) in references
    assert ("mc-2026-0002", "mc-2026-0001") in references
    assert not any("OW" in target for _, target in references)

    assert ("mc-2026-0001", "Asset Management Company") in edges(graph, "APPLIES_TO")


def test_amendments_resolve_aliases_and_only_the_modified_document():
    alignment = document(
        "cir-553",
        "circular",
        "SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/553",
        "Alignment of interest of Key Employees",
        "2021-04-28",
        "2021-07-01",
        [(1, None, "1. Key Employees shall invest in units.", {})],
    )
    clarification = document(
        "cir-629",
        "circular",
        "SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/629",
        "Clarifications",
        "2021-09-20",
        None,
        [
            (
                1,
                None,
                (
                    "1. SEBI, vide Circular no. SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/553 dated April 28, "
                    "2021 (hereinafter referred to as “Alignment circular”), following are clarified.\n"
                    "2. The provision under para 2(i) of the Alignment Circular shall be read as below.\n"
                    "3. All other provisions of the Alignment circular shall remain unchanged."
                ),
                {},
            )
        ],
    )
    relaxation = document(
        "cir-2025-36",
        "circular",
        "SEBI/HO/IMD/IMD-PoD-1/P/CIR/2025/36",
        "Ease of doing business",
        "2025-03-21",
        None,
        [
            (
                1,
                None,
                (
                    "1. Amendments to SEBI (Mutual Funds) Regulations, 1996 (‘MF Regulations’) were "
                    "carried out.\n2. Accordingly, in terms of Regulation 25 (16B) of MF Regulations, "
                    "the Master Circular for Mutual Funds dated June 27, 2024 (‘Master Circular’) has "
                    "been modified as under:\nA Clause 6.10.1.1 modified as: “Deleted”"
                ),
                {},
            )
        ],
    )

    graph = extract([alignment, clarification, relaxation])
    amends = edges(graph, "AMENDS")

    key_553 = f"cir:{number_key('SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/553')}"
    key_629 = f"cir:{number_key('SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/629')}"
    key_36 = f"cir:{number_key('SEBI/HO/IMD/IMD-PoD-1/P/CIR/2025/36')}"
    assert amends[(key_629, key_553)]["scope"] == "partial"
    assert amends[(key_629, key_553)]["clauses"] == ["2(i)"]
    assert set(amends) == {
        (key_629, key_553),
        (key_36, "doc:MASTERCIRCULARFORMUTUALFUNDS:2024-06-27"),
    }
    assert amends[(key_36, "doc:MASTERCIRCULARFORMUTUALFUNDS:2024-06-27")][
        "clauses"
    ] == ["6.10.1.1"]
    assert graph["circulars"][key_553]["status"] == "amended"


def test_regulation_repeal_and_amendment_footnotes_with_ibid():
    new = document(
        "regs-2026",
        "regulations",
        None,
        "Securities and Exchange Board of India (Mutual Funds) Regulations, 2026",
        "2026-01-14",
        "2026-04-01",
        [
            (
                187,
                "85. Repeal and saving",
                (
                    "(1) The Securities & Exchange Board of India (Mutual Funds) Regulations, 1996 stand "
                    "repealed from the date of coming into force of these regulations."
                ),
                {},
            )
        ],
    )
    old = document(
        "regs-1996",
        "regulations",
        None,
        "Securities and Exchange Board of India (Mutual Funds) Regulations 1996 [Last amended on March 4, 2025.]",
        None,
        None,
        [
            (
                10,
                "Definitions.",
                (
                    "[^21] [(mm) “group” means a group as defined in the Competition Act, 2002.] "
                    "[^22] [(mn) “index fund scheme” means a scheme that tracks an index;] [^23]"
                ),
                {
                    "21": "Substituted by the SEBI (Mutual Funds) (Amendment) Regulations, 2021, "
                    "w.e.f. 5-3- 2021. Prior to its substitution, clause (mm) read as under;",
                    "22": "Inserted ibid.",
                    "23": "Substituted for “.” by the SEBI (Mutual Funds) (Second Amendment) "
                    "Regulations, 2018, w.e.f. 30.5.2018.",
                },
            )
        ],
    )

    graph = extract([new, old])
    regs_1996 = "reg:SEBIMUTUALFUNDSREGULATIONS1996"

    supersedes = edges(graph, "SUPERSEDES")
    assert (
        supersedes[("reg:SEBIMUTUALFUNDSREGULATIONS2026", regs_1996)]["effective_from"]
        == "2026-04-01"
    )
    assert graph["circulars"][regs_1996]["effective_to"] == "2026-04-01"
    assert graph["circulars"][regs_1996]["status"] == "superseded"

    footnote_amendments = {
        props["footnote"]: (start[1], props["action"], props["effective_from"])
        for (rel, start, end, _), props in graph["edges"].items()
        if rel == "AMENDS" and end[1] == regs_1996
    }
    assert footnote_amendments == {
        "21": (
            "reg:SEBIMUTUALFUNDSAMENDMENTREGULATIONS2021",
            "substituted",
            "2021-03-05",
        ),
        "22": ("reg:SEBIMUTUALFUNDSAMENDMENTREGULATIONS2021", "inserted", "2021-03-05"),
        "23": (
            "reg:SEBIMUTUALFUNDSSECONDAMENDMENTREGULATIONS2018",
            "substituted",
            "2018-05-30",
        ),
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The trustees of a mutual fund shall ensure compliance.", {"Trustee"}),
        (
            "AMCs and AMFI shall carry out investor education initiatives.",
            {"Asset Management Company", "AMFI"},
        ),
        (
            "No mutual fund lite scheme shall be launched by the mutual fund lite asset management company.",
            set(),
        ),
        ("The value shall be computed as the case may be.", set()),
    ],
)
def test_obligated_entities_use_the_subject_of_the_obligation(text, expected):
    assert set(obligated_entities(text)) == expected


@pytest.fixture(scope="module")
def built_graph():
    settings = Settings()
    try:
        driver = connect(settings)
        driver.verify_connectivity()
    except (ServiceUnavailable, Neo4jError):
        pytest.skip("Neo4j is not reachable; run `docker compose up -d`")
    with driver.session(database=settings.neo4j_database) as session:
        if session.run("MATCH (c:Circular) RETURN count(c) AS n").single()["n"] == 0:
            driver.close()
            pytest.skip("graph not built; run `python -m inforce.graph build`")
    yield driver, settings
    driver.close()


def test_built_graph_passes_validation(built_graph):
    driver, settings = built_graph
    _, issues = validate(driver, settings.neo4j_database)
    assert {
        name: items
        for (severity, name), items in issues.items()
        if severity == "error" and items
    } == {}
    assert len([1 for severity, _ in issues if severity == "error"]) == len(CHECKS)


def test_built_graph_contains_real_corpus_relationships(built_graph):
    driver, settings = built_graph
    with driver.session(database=settings.neo4j_database) as session:
        chain = session.run(
            """MATCH (new:Circular {doc_id: 'mf-regulations-2026'})-[:SUPERSEDES]->(old:Circular)
               RETURN old.doc_id AS old, old.effective_to AS ended, old.status AS status"""
        ).single()
        rescinded = session.run(
            """MATCH (:Circular {doc_id: 'mf-master-circular-2026'})-[r:RESCINDS]->(c:Circular)
               RETURN count(c) AS n, min(r.appendix_sr_no) AS first, max(r.appendix_sr_no) AS last"""
        ).single()
        amended = session.run(
            """MATCH (a:Circular {doc_id: 'cir-2021-09-alignment-clarifications'})-[r:AMENDS]->
                     (b:Circular {doc_id: 'cir-2021-04-alignment-key-employees'})
               RETURN r.source_doc AS doc, r.page_start AS page"""
        ).single()
    assert dict(chain) == {
        "old": "mf-regulations-1996-amended-2025",
        "ended": "2026-04-01",
        "status": "superseded",
    }
    assert dict(rescinded) == {"n": 34, "first": 1, "last": 34}
    assert dict(amended) == {"doc": "cir-2021-09-alignment-clarifications", "page": 1}
