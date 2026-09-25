import argparse
import calendar
import json
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from neo4j import Session

from inforce.config import Settings
from inforce.graph import (
    MASTER_CIRCULAR_RE,
    REGULATIONS_RE,
    TEXT_DATE,
    connect,
    number_key,
    parse_date,
    title_key,
)
from inforce.ingest import MONTH, MONTHS

NEIGHBOURS_PER_HOP = (12, 5, 3)
FRONTIER_LIMIT = 25
EXAMPLES = 3
ANCHOR_LIMIT = 3
REFERENCE_LIMIT = 8
PROVISION_LIMIT = 5

IDENTIFIER_RE = re.compile(r"[A-Za-z0-9()\-.]+(?:\s*/\s*[A-Za-z0-9()\-.]+){2,}")
MF_REGULATIONS_RE = re.compile(
    r"\b(?:SEBI\s+)?\(?(?:mutual\s+funds?|MF)\)?\s+regulations?,?\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
AS_OF_RE = re.compile(
    rf"\b(?:as\s+(?:of|on)|on|in|during|at)\s+({TEXT_DATE}|{MONTH},?\s+\d{{4}}|(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
DEFINITION_RE = re.compile(
    r"\b(?:definition|meaning)\s+of\s+(?:the\s+(?:term|expression)\s+)?[“\"']?"
    r"([A-Za-z][A-Za-z-]*(?:\s[A-Za-z-]+){0,3}?)[”\"']?(?=\s+(?:under|in|as)\b|\s*[?.,]|\s*$)",
    re.IGNORECASE,
)
CURRENT_RE = re.compile(
    r"\b(?:currently|today|now|at\s+present|presently)\b", re.IGNORECASE
)
STOPWORD_TEXT = """a an the and or of to in on at as by for with from into under over about what which who whom
    when where why how is are was were be been being do does did has have had this that these those
    it its any all there their them they circular circulars sebi supersede superseded supersedes
    replace replaced amend amended amends rescind rescinded rescinds repeal repealed relationship
    between applied apply applicable valid force effective currently today now present date rule
    rules requirement requirements later earlier issued"""
STOPWORDS = set(STOPWORD_TEXT.split())
ENDED_STATUSES = ("superseded", "rescinded")

CIRCULAR_BY_ID = "MATCH (c:Circular) WHERE c.id IN $ids RETURN c"

MASTER_CIRCULAR_BY_DATE = """
MATCH (c:Circular {doc_type: 'master_circular', issue_date: $date}) RETURN c"""

CIRCULAR_FULLTEXT = """
CALL db.index.fulltext.queryNodes('circular_text', $terms) YIELD node, score
WHERE coalesce(node.in_corpus, false) = $in_corpus
RETURN node AS c, score
LIMIT $limit"""

LIFECYCLE_NEIGHBOURS = """
MATCH (a:Circular {id: $id})-[r:SUPERSEDES|AMENDS|RESCINDS]-(b:Circular)
WITH b, type(r) AS type, r
ORDER BY r.page_start, r.footnote
WITH b, type, count(r) AS parallel_edges,
     collect(r {.*, type: type(r), start: startNode(r).id, end: endNode(r).id})[..$examples] AS rels
RETURN b, type, parallel_edges, rels
ORDER BY CASE type WHEN 'SUPERSEDES' THEN 0 WHEN 'RESCINDS' THEN 1 ELSE 2 END,
         b.in_corpus DESC, coalesce(b.issue_date, '') DESC
LIMIT $limit"""

CITED_BY_ANCHOR = """
MATCH (a:Circular {id: $id})<-[:PART_OF]-(p:Provision)-[r:REFERENCES]->(b:Circular)
WITH b, r, p ORDER BY p.page_start
WITH b, count(r) AS citations,
     collect(r {.*, type: 'REFERENCES', start: p.id, end: b.id, provision_section: p.section})[..$examples] AS rels
RETURN b, citations, rels
ORDER BY citations DESC
LIMIT $limit"""

CITING_ANCHOR = """
MATCH (b:Circular)<-[:PART_OF]-(p:Provision)-[r:REFERENCES]->(a:Circular {id: $id})
WITH a, b, r, p ORDER BY p.page_start
WITH a, b, count(r) AS citations,
     collect(r {.*, type: 'REFERENCES', start: p.id, end: a.id, provision_section: p.section})[..$examples] AS rels
RETURN b, citations, rels
ORDER BY citations DESC
LIMIT $limit"""

LIFECYCLE_PATH = """
MATCH (a:Circular {id: $a}), (b:Circular {id: $b})
MATCH p = shortestPath((a)-[:SUPERSEDES|AMENDS|RESCINDS*..3]-(b))
RETURN p"""

CITATION_PATH = """
MATCH (a:Circular {id: $a}), (b:Circular {id: $b})
MATCH p = shortestPath((a)-[:SUPERSEDES|AMENDS|RESCINDS|REFERENCES|PART_OF*..6]-(b))
RETURN p"""

PROVISION_FULLTEXT = """
CALL db.index.fulltext.queryNodes('provision_text', $terms) YIELD node AS p, score
MATCH (p)-[:PART_OF]->(c:Circular)
WHERE $scope IS NULL OR c.id IN $scope
WITH p, c, score ORDER BY score DESC LIMIT $limit
OPTIONAL MATCH (amender:Circular)-[am:AMENDS]->(c) WHERE am.provision = p.id
RETURN p, c, score,
       collect(am {.*, type: 'AMENDS', start: amender.id, end: c.id, amender_title: amender.title}) AS amendments
ORDER BY score DESC"""


def as_of_range(query: str, today: date) -> dict | None:
    if m := AS_OF_RE.search(query):
        when = m[1]
        if exact := parse_date(when):
            return {"text": m.group(), "start": exact, "end": exact}
        if when.isdigit():
            return {"text": m.group(), "start": f"{when}-01-01", "end": f"{when}-12-31"}
        month, year = re.match(
            rf"({MONTH}),?\s+(\d{{4}})", when, re.IGNORECASE
        ).groups()
        number = MONTHS.index(month.lower()) + 1
        last = calendar.monthrange(int(year), number)[1]
        return {
            "text": m.group(),
            "start": f"{year}-{number:02d}-01",
            "end": f"{year}-{number:02d}-{last:02d}",
        }
    if CURRENT_RE.search(query):
        return {
            "text": CURRENT_RE.search(query).group(),
            "start": today.isoformat(),
            "end": today.isoformat(),
        }
    return None


def validity(node: dict, start: str, end: str) -> dict:
    begins, ends, issued = (
        node.get("effective_from"),
        node.get("effective_to"),
        node.get("issue_date"),
    )
    if ends and ends <= start:
        return {
            "state": "ended",
            "reason": f"effective_to {ends} ({node.get('status_source')})",
        }
    if begins and begins > end:
        return {"state": "not_yet_in_force", "reason": f"effective_from {begins}"}
    if not begins and issued and issued > end:
        return {"state": "not_yet_issued", "reason": f"issued {issued}"}
    if (begins and begins > start) or (ends and ends <= end):
        return {
            "state": "changed_during_period",
            "reason": f"effective {begins} to {ends}",
        }
    if node.get("status") in ENDED_STATUSES and not ends:
        return {
            "state": "ended_unknown_date",
            "reason": f"{node['status']} ({node.get('status_source')})",
        }
    if not begins:
        detail = f"issued {issued}" if issued else "no issue date recorded"
        if ends:
            detail += f"; effective_to {ends} ({node.get('status_source')})"
        return {
            "state": "unknown_start",
            "reason": f"no effective_from recorded; {detail}",
        }
    return {
        "state": "in_force",
        "reason": f"effective_from {begins}; no end recorded in the graph",
    }


def without_document_mentions(query: str) -> str:
    for pattern in (
        IDENTIFIER_RE,
        REGULATIONS_RE,
        MF_REGULATIONS_RE,
        MASTER_CIRCULAR_RE,
    ):
        query = pattern.sub(" ", query)
    return query


def anchor_lookups(query: str) -> tuple[list[str], list[str]]:
    ids = [f"cir:{number_key(m.group())}" for m in IDENTIFIER_RE.finditer(query)]
    ids += [f"reg:{title_key(m.group())}" for m in REGULATIONS_RE.finditer(query)]
    ids += [
        f"reg:SEBIMUTUALFUNDSREGULATIONS{m[1]}"
        for m in MF_REGULATIONS_RE.finditer(query)
    ]
    master_circular_dates = [
        parse_date(m[2]) for m in MASTER_CIRCULAR_RE.finditer(query)
    ]
    return list(dict.fromkeys(ids)), master_circular_dates


def search_terms(query: str) -> str | None:
    words = [
        w
        for w in re.findall(r"[A-Za-z]{3,}|\d{4}", query)
        if w.lower() not in STOPWORDS
    ]
    phrases = [f'"{m[1].strip()} means"^3' for m in DEFINITION_RE.finditer(query)]
    return " OR ".join([*phrases, *dict.fromkeys(words)]) or None


def find_anchors(session: Session, query: str) -> list[dict]:
    ids, master_circular_dates = anchor_lookups(query)
    anchors = [
        {**dict(r["c"]), "matched_by": "identifier"}
        for r in session.run(CIRCULAR_BY_ID, ids=ids)
    ]
    for issued in master_circular_dates:
        anchors += [
            {**dict(r["c"]), "matched_by": "master_circular_date"}
            for r in session.run(MASTER_CIRCULAR_BY_DATE, date=issued)
        ]
    if anchors or not (terms := search_terms(query)):
        return list({a["id"]: a for a in anchors}.values())
    for in_corpus in (True, False):
        rows = list(
            session.run(
                CIRCULAR_FULLTEXT, terms=terms, in_corpus=in_corpus, limit=ANCHOR_LIMIT
            )
        )
        if rows:
            return [
                {
                    **dict(r["c"]),
                    "matched_by": "title_fulltext",
                    "score": round(r["score"], 3),
                }
                for r in rows
                if r["score"] >= 0.6 * rows[0]["score"]
            ]
    return []


def lifecycle(session: Session, anchor_ids: list[str]) -> tuple[dict, list[dict]]:
    nodes, relationships, seen = {}, {}, set(anchor_ids)
    frontier = list(anchor_ids)
    for hop, limit in enumerate(NEIGHBOURS_PER_HOP, start=1):
        next_frontier = []
        for node_id in frontier:
            rows = list(
                session.run(
                    LIFECYCLE_NEIGHBOURS,
                    id=node_id,
                    limit=limit,
                    examples=EXAMPLES,
                )
            )
            for row in rows:
                neighbour = dict(row["b"])
                nodes[neighbour["id"]] = neighbour
                for rel in row["rels"]:
                    key = (
                        rel["type"],
                        rel["start"],
                        rel["end"],
                        rel.get("source_chunk"),
                        rel.get("footnote"),
                    )
                    relationships.setdefault(
                        key,
                        {
                            **rel,
                            "hop": hop,
                            "parallel_edges": row["parallel_edges"],
                            "neighbours_truncated": len(rows) == limit,
                        },
                    )
                if neighbour["id"] not in seen:
                    seen.add(neighbour["id"])
                    next_frontier.append(neighbour["id"])
        frontier = next_frontier[:FRONTIER_LIMIT]
    return nodes, list(relationships.values())


def citations(session: Session, anchor_ids: list[str]) -> tuple[dict, list[dict]]:
    nodes, relationships = {}, []
    for node_id in anchor_ids:
        for query, direction in (
            (CITED_BY_ANCHOR, "cited_by_anchor"),
            (CITING_ANCHOR, "cites_anchor"),
        ):
            for row in session.run(
                query, id=node_id, limit=REFERENCE_LIMIT, examples=EXAMPLES
            ):
                other = dict(row["b"])
                nodes[other["id"]] = other
                relationships += [
                    {
                        **rel,
                        "hop": 1,
                        "direction": direction,
                        "citations": row["citations"],
                        "circular": other["id"],
                    }
                    for rel in row["rels"]
                ]
    return nodes, relationships


def path_between(session: Session, a: str, b: str) -> dict | None:
    for query, kind in ((LIFECYCLE_PATH, "lifecycle"), (CITATION_PATH, "citation")):
        record = session.run(query, a=a, b=b).single()
        if record:
            path = record["p"]
            return {
                "from": a,
                "to": b,
                "kind": kind,
                "nodes": [
                    {"label": next(iter(n.labels)), **dict(n)} for n in path.nodes
                ],
                "relationships": [
                    {
                        **dict(r),
                        "type": r.type,
                        "start": r.start_node["id"],
                        "end": r.end_node["id"],
                    }
                    for r in path.relationships
                ],
            }
    return None


def provisions(
    session: Session, query: str, scope: list[str] | None
) -> tuple[dict, list[dict]]:
    terms = search_terms(query)
    if not terms:
        return {}, []
    nodes, found = {}, {}
    for current_scope in [scope, None] if scope else [None]:
        for row in session.run(
            PROVISION_FULLTEXT, terms=terms, scope=current_scope, limit=PROVISION_LIMIT
        ):
            provision, circular = dict(row["p"]), dict(row["c"])
            nodes[circular["id"]] = circular
            found.setdefault(
                provision["id"],
                {
                    **provision,
                    "footnotes": json.loads(provision["footnotes"]),
                    "circular": circular["id"],
                    "score": round(row["score"], 3),
                    "in_anchor_scope": current_scope is not None,
                    "amendments": row["amendments"],
                },
            )
    return nodes, list(found.values())


def retrieve(session: Session, query: str, today: date | None = None) -> dict:
    as_of = as_of_range(query, today or datetime.now(ZoneInfo("Asia/Kolkata")).date())
    topic = query.replace(as_of["text"], " ") if as_of else query
    anchors = find_anchors(session, topic)
    anchor_ids = [a["id"] for a in anchors]
    nodes = {
        a["id"]: {k: v for k, v in a.items() if k not in ("matched_by", "score")}
        for a in anchors
    }

    lifecycle_nodes, relationships = lifecycle(session, anchor_ids)
    citation_nodes, citation_relationships = citations(session, anchor_ids)
    nodes |= lifecycle_nodes
    nodes |= citation_nodes

    paths = [
        path
        for i, a in enumerate(anchor_ids)
        for b in anchor_ids[i + 1 :]
        if (path := path_between(session, a, b))
    ]

    scope = [a["id"] for a in anchors if a.get("in_corpus")] or [
        n["id"] for n in lifecycle_nodes.values() if n.get("in_corpus")
    ]
    provision_nodes, matched_provisions = provisions(
        session, without_document_mentions(topic), scope or None
    )
    nodes |= provision_nodes

    temporal, conflicts = [], []
    if as_of:
        states = {}
        versioned = {
            r[end]
            for r in relationships
            if r["type"] in ("SUPERSEDES", "RESCINDS")
            for end in ("start", "end")
        }
        for node_id, node in nodes.items():
            if node.get("in_corpus") or node_id in anchor_ids or node_id in versioned:
                amendments = [
                    {
                        k: r.get(k)
                        for k in (
                            "start",
                            "effective_from",
                            "provision",
                            "footnote",
                            "clauses",
                            "source_doc",
                            "page_start",
                        )
                    }
                    for r in [
                        *relationships,
                        *(a for p in matched_provisions for a in p["amendments"]),
                    ]
                    if r["type"] == "AMENDS" and r["end"] == node_id
                ]
                states[node_id] = validity(node, as_of["start"], as_of["end"])
                temporal.append(
                    {
                        "id": node_id,
                        **states[node_id],
                        **{
                            k: node.get(k)
                            for k in (
                                "issue_date",
                                "effective_from",
                                "effective_to",
                                "status",
                                "status_source",
                            )
                        },
                        "amendments_in_effect": [
                            a
                            for a in amendments
                            if a["effective_from"]
                            and a["effective_from"] <= as_of["end"]
                        ],
                        "amendments_after": [
                            a
                            for a in amendments
                            if a["effective_from"]
                            and a["effective_from"] > as_of["end"]
                        ],
                        "amendments_undated": [
                            a for a in amendments if not a["effective_from"]
                        ],
                    }
                )
        for rel in relationships:
            if (
                rel["type"] in ("SUPERSEDES", "RESCINDS")
                and states.get(rel["start"], {}).get("state")
                == states.get(rel["end"], {}).get("state")
                == "in_force"
            ):
                conflicts.append(
                    {
                        "reason": "replacing and replaced documents both in force",
                        "relationship": rel,
                    }
                )

    pages = {}
    for item in [
        *relationships,
        *citation_relationships,
        *(r for p in paths for r in p["relationships"]),
    ]:
        if item.get("source_doc"):
            pages.setdefault(item["source_doc"], set()).update(
                range(item["page_start"], item["page_end"] + 1)
            )
    for provision in matched_provisions:
        doc_id = nodes[provision["circular"]].get("doc_id")
        pages.setdefault(doc_id, set()).update(
            range(provision["page_start"], provision["page_end"] + 1)
        )
    documents = {n["doc_id"]: n for n in nodes.values() if n.get("doc_id")}
    sources = [
        {
            "doc_id": doc_id,
            "title": documents.get(doc_id, {}).get("title"),
            "url": documents.get(doc_id, {}).get("url"),
            "pages": sorted(doc_pages),
        }
        for doc_id, doc_pages in sorted(pages.items())
    ]

    return {
        "query": query,
        "as_of": as_of,
        "anchors": [
            {"id": a["id"], "matched_by": a["matched_by"], "score": a.get("score")}
            for a in anchors
        ],
        "nodes": nodes,
        "relationships": relationships,
        "citations": citation_relationships,
        "paths": paths,
        "provisions": matched_provisions,
        "temporal": temporal,
        "conflicts": conflicts,
        "sources": sources,
    }


def describe(node: dict) -> str:
    return node.get("number") or node.get("title") or node["id"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve graph evidence for a regulatory question."
    )
    parser.add_argument("query")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    with (
        connect(settings) as driver,
        driver.session(database=settings.neo4j_database) as session,
    ):
        result = retrieve(session, args.query)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    nodes = result["nodes"]
    print(f"as_of: {result['as_of']}")
    print(
        "anchors:",
        [(describe(nodes[a["id"]]), a["matched_by"]) for a in result["anchors"]],
    )
    for rel in result["relationships"]:
        print(
            f"  hop {rel['hop']}: {describe(nodes.get(rel['start'], {'id': rel['start']}))} "
            f"-{rel['type']}-> {describe(nodes.get(rel['end'], {'id': rel['end']}))} "
            f"[{rel['source_doc']} p.{rel['page_start']}; effective_from={rel.get('effective_from')}; "
            f"parallel={rel['parallel_edges']}]"
        )
    for path in result["paths"]:
        steps = " / ".join(
            f"{r['start']} -{r['type']}-> {r['end']}" for r in path["relationships"]
        )
        print(f"  path ({path['kind']}): {steps}")
    for state in result["temporal"]:
        print(
            f"  temporal: {describe(nodes[state['id']])}: {state['state']} ({state['reason']})"
        )
    for conflict in result["conflicts"]:
        print(f"  CONFLICT: {conflict['reason']}")
    for provision in result["provisions"]:
        print(
            f"  provision {provision['id']} p.{provision['page_start']} [{provision.get('section')}] amendments={len(provision['amendments'])}"
        )
    print("sources:", [(s["doc_id"], s["pages"][:10]) for s in result["sources"]])


if __name__ == "__main__":
    main()
