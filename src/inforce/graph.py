import argparse
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from neo4j import Driver, GraphDatabase

from inforce.config import Settings
from inforce.ingest import MONTH, MONTHS, OUT_DIR

TEXT_DATE = rf"(?:{MONTH}\s+\d{{1,2}},?\s*\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH},?\s*\d{{4}})"
REGULATIONS = (
    r"(?:SEBI|Securities\s*(?:and|&)\s*Exchange\s*Board\s*of\s*India)\s*"
    r"(?:\(+[^()]{2,120}\)+\s*)+Regulations?,?\s*\d{4}"
)

DATED_RE = re.compile(rf"\s*,?\s*dated\s+({TEXT_DATE})", re.IGNORECASE)
MASTER_CIRCULAR_RE = re.compile(
    rf"\b(?:SEBI\s+)?(Master\s+Circular\s+for\s+Mutual\s+Funds)\s+dated\s+({TEXT_DATE})",
    re.IGNORECASE,
)
DATED_CIRCULAR_RE = re.compile(
    rf"\bCircular\s+dated\s+({TEXT_DATE})(?:\s+on\s+[‘'“\"]?([^’'”\".]{{5,120}}))?",
    re.IGNORECASE,
)
REGULATIONS_RE = re.compile(REGULATIONS, re.IGNORECASE)
REGULATION_NUMBER_RE = re.compile(
    rf"\bRegulations?\s+(\d+[A-Z]{{0,2}})(?:\s*\(\s*\w+\s*\))*\s+of\s+(?:the\s+)?({REGULATIONS})",
    re.IGNORECASE,
)
PARAGRAPH_RE = re.compile(
    r"\b(?:Paragraph|Para|Clause)s?\s+(\d+(?:\.\d+)+)(?:\s*\([a-z0-9]+\))*\s+"
    r"(?:above|below|of\s+this\s+Master\s+Circular)",
    re.IGNORECASE,
)
ALIAS_RE = re.compile(
    r"\((?:hereinafter\s+(?:referred\s+(?:to\s+)?as|called)\s*)?[“‘\"']([A-Z][^”’\"']{2,60})[”’\"']\s*\)"
)
NUMBERED_LINE_RE = re.compile(r"^(\d+[A-Z]{0,2}(?:\.\d+)*)\.(?:\s|$)", re.MULTILINE)
APPENDIX_ROW_RE = re.compile(
    rf"^(\d{{1,3}})\.\s+(.+?)\s+({TEXT_DATE})\s+(.+)$", re.IGNORECASE
)

REPLACE_RE = re.compile(
    r"\b(?:this|these)\s+(?:master\s+circular|circular|regulations)\b[^.;]{0,120}?"
    r"\bshall\s+replace\b([^;]*)",
    re.IGNORECASE,
)
REPEAL_RE = re.compile(
    rf"({REGULATIONS})\s+(?:are|is|stands?|shall\s+stand)\s+(?:hereby\s+)?repealed([^.;]*)",
    re.IGNORECASE,
)
RESCIND_RANGE_RE = re.compile(
    r"Sr\.?\s*Nos?\.?\s*(\d+)\s*(?:to|-)\s*(\d+)\s+in\s+the\s+Appendix\s+to\s+this\s+"
    r"Master\s+Circular[^.]*?\bshall\s+stand\s+rescinded",
    re.IGNORECASE,
)
MODIFIED_RE = re.compile(
    r"([^.;]*?)\bhas\s+been\s+(?:modified|amended)\b", re.IGNORECASE
)
UNCHANGED_RE = re.compile(
    r"\bAll\s+other\s+(?:provisions|conditions)([^.;]*?)\bshall\s+remain\s+unchanged",
    re.IGNORECASE,
)
CLAUSE_RE = re.compile(
    r"\b(?:clause|para(?:graph)?)\s+(\d+(?:\.\d+)*(?:\([a-z0-9]+\))*)\.?\s+"
    r"(?:(modified\s+as|shall\s+be\s+inserted)|of\s+(?:the\s+)?([^,.;]{3,80}))",
    re.IGNORECASE,
)
FOOTNOTE_AMENDMENT_RE = re.compile(
    rf"\b(inserted|substitut(?:ed|e)|omitted|renumbered)\b.{{0,200}}?"
    rf"(?:\bby\s+(?:the\s+)?({REGULATIONS})|\b(ibid)\b)",
    re.IGNORECASE,
)
IBID_RE = re.compile(
    r"^\W*(?:(inserted|substituted|omitted)\s+(?:by\s+)?)?ibid\b", re.IGNORECASE
)
WEF_RE = re.compile(
    rf"w\.?\s*e\.?\s*f\.?,?\s*({TEXT_DATE}|\d{{1,2}}\s*[-.]\s*\d{{1,2}}\s*[-.]\s*\d{{4}})",
    re.IGNORECASE,
)
MODAL_RE = re.compile(r"\b(?:shall|should|must|may|(?:is|are)\s+required\s+to)\b")
PREPOSITIONS = {
    "of",
    "for",
    "by",
    "under",
    "to",
    "with",
    "from",
    "in",
    "on",
    "than",
    "between",
}
DETERMINERS = {
    "a",
    "an",
    "the",
    "such",
    "all",
    "each",
    "every",
    "its",
    "their",
    "concerned",
}
LETTER_WORDS = re.compile(
    r"\b(?:letter|e-?mail|communication|notification|press\s+release)\b", re.IGNORECASE
)

ENTITIES = {
    "Asset Management Company": (
        "intermediary",
        r"asset\s+management\s+compan(?:y|ies)|AMCs?",
    ),
    "Mutual Fund": (
        "intermediary",
        r"mutual\s+funds?(?!\s+(?:schemes?|distributors?|industry|units?|lite))|MFs?",
    ),
    "Trustee": (
        "intermediary",
        r"board\s+of\s+trustees|trustee\s+compan(?:y|ies)|trustees?",
    ),
    "Sponsor": ("intermediary", r"sponsors?"),
    "Custodian": ("intermediary", r"custodians?"),
    "Registrar and Transfer Agent": (
        "intermediary",
        r"registrars?\s+(?:to\s+an\s+issue\s+)?and\s+(?:share\s+)?transfer\s+agents?|RTAs?",
    ),
    "Distributor": ("intermediary", r"(?:mutual\s+fund\s+)?distributors?|MFDs?"),
    "Stock Exchange": (
        "market_infrastructure",
        r"(?:recogni[sz]ed\s+)?stock\s+exchanges?",
    ),
    "Depository": ("market_infrastructure", r"depositor(?:y|ies)"),
    "AMFI": ("industry_body", r"AMFI|Association\s+of\s+Mutual\s+Funds\s+in\s+India"),
    "Investor": ("investor", r"investors?|unit\s*holders?"),
    "Designated Employee": ("individual", r"(?:designated|key)\s+employees?"),
}
ENTITY_RE = re.compile(
    "|".join(
        f"(?P<e{i}>\\b(?:{pattern})\\b)"
        for i, (_, pattern) in enumerate(ENTITIES.values())
    ),
    re.IGNORECASE,
)
ENTITY_NAMES = list(ENTITIES)


def number_key(number: str) -> str:
    text = number.upper().replace("`", "")
    text = re.sub(r"\bCIR\.?\s*/?\s*NO\b\.?\s*", "CIR/", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^SEBI/", "", text)
    text = re.sub(r"(?<=[/.])0+(?=[0-9])", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def title_key(title: str) -> str:
    text = re.sub(r"\[.*?\]", "", title.upper().replace("&", "AND"))
    text = re.sub(r"SECURITIES\s*AND\s*EXCHANGE\s*BOARD\s*OF\s*INDIA", "SEBI", text)
    text = re.sub(r"^\s*SEBI\s+(?=MASTER)", "", text)
    text = re.sub(r"\((?:MF|MUTUAL\s*FUNDS?)\)", "(MUTUAL FUNDS)", text)
    text = re.sub(r"\bREGULATIONS?\b", "REGULATIONS", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def tokens(text: str) -> set[str]:
    text = re.sub(r"isation", "ization", text.lower())
    return {w for w in re.findall(r"[a-z]{4,}", text)}


def parse_date(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text)
    if m := re.search(rf"({MONTH}) (\d{{1,2}}),? ?(\d{{4}})", text, re.IGNORECASE):
        month, day, year = m[1], m[2], m[3]
    elif (
        m := re.search(
            rf"(\d{{1,2}})(?:st|nd|rd|th)? ({MONTH}),? ?(\d{{4}})", text, re.IGNORECASE
        )
    ) or (m := re.search(r"(\d{1,2}) ?[-.] ?(\d{1,2}) ?[-.] ?(\d{4})", text)):
        day, month, year = m[1], m[2], m[3]
    else:
        return None
    if not month.isdigit():
        month = MONTHS.index(month.lower()) + 1
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def normalize_citations(text: str) -> str:
    text = re.sub(r"[ \t]*/[ \t]*", "/", text)
    text = re.sub(r"(?i)\bcir\s*[–-]\s*(?=\d)", "Cir-", text)
    return re.sub(
        r"(?<=/)(\d{1,3}) (\d{1,3})\b",
        lambda m: m[1] + m[2] if len(m[1] + m[2]) == 4 else m[0],
        text,
    )


def identifier_before(text: str, end: int) -> tuple[str, int] | None:
    parts = []
    for match in reversed(list(re.finditer(r"\S+", text[:end]))):
        raw = match.group()
        token = raw.rstrip(",;")
        if parts and raw != token:
            break
        adjacent_slash = parts and (
            parts[0].group().startswith("/") or token.endswith("/")
        )
        if (
            "/" in token
            or (token.isdigit() and adjacent_slash)
            or (parts and re.fullmatch(r"(?i)no\.?|cir\.?", token))
        ):
            parts.insert(0, match)
        else:
            break
    while parts and re.fullmatch(r"(?i)no\.?", parts[0].group()):
        parts.pop(0)
    if not parts:
        return None
    number = re.sub(r"\s*/\s*", "/", text[parts[0].start() : end]).strip(" ,;")
    number = re.sub(
        r"^(?:(?:cir(?:cular)?|no)\.?\s+|no\s*[.:–-]\s*|[–:-]\s*)+",
        "",
        number,
        flags=re.IGNORECASE,
    )
    return (number, parts[0].start()) if number.count("/") >= 2 else None


def new_graph() -> dict:
    return {
        "circulars": {},
        "provisions": [],
        "edges": {},
        "index": {"title_date": {}, "date": defaultdict(set)},
    }


def add_circular(graph: dict, node_id: str, **props) -> str:
    node = graph["circulars"].setdefault(
        node_id,
        {"id": node_id, "issuer": "SEBI", "in_corpus": False, "date_conflicts": []},
    )
    issued = props.get("issue_date")
    if issued and node.get("issue_date") and node["issue_date"] != issued:
        if issued not in node["date_conflicts"]:
            node["date_conflicts"].append(issued)
        props.pop("issue_date")
    for key, value in props.items():
        if value is not None and node.get(key) is None:
            node[key] = value
    if node.get("title") and node.get("issue_date"):
        graph["index"]["title_date"][(title_key(node["title"]), node["issue_date"])] = (
            node_id
        )
    if node.get("issue_date"):
        graph["index"]["date"][node["issue_date"]].add(node_id)
    return node_id


def add_edge(
    graph: dict, rel: str, start: tuple, end: tuple, props: dict, key=()
) -> None:
    edge_key = (rel, start, end, key)
    if edge_key in graph["edges"]:
        graph["edges"][edge_key]["mentions"] += 1
    else:
        graph["edges"][edge_key] = {**props, "mentions": 1}


def number_citations(graph: dict, text: str) -> list[tuple[str, int, int]]:
    found = []
    for dated in DATED_RE.finditer(text):
        identified = identifier_before(text, dated.start())
        if not identified:
            continue
        number, start = identified
        context = re.split(r"[,;]", text[max(0, start - 40) : start])[-1]
        if LETTER_WORDS.search(context) or re.search(r"(?:^|/)[O0]W/", number.upper()):
            continue
        if not re.search(r"circular|\bcir\b", context, re.IGNORECASE) and not (
            "CIR" in number.upper() or number.upper().startswith("HO/")
        ):
            continue
        node_id = add_circular(
            graph,
            f"cir:{number_key(number)}",
            number=number,
            doc_type="circular",
            issue_date=parse_date(dated.group(1)),
        )
        found.append((node_id, start, dated.end()))
    return found


def citations(
    graph: dict, text: str, aliases: dict[str, str]
) -> list[tuple[str, int, int]]:
    found = number_citations(graph, text)
    taken = [(s, e) for _, s, e in found]

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in taken)

    for m in MASTER_CIRCULAR_RE.finditer(text):
        if free(m.start(), m.end()):
            issued = parse_date(m[2])
            node_id = graph["index"]["title_date"].get(
                (title_key(m[1]), issued)
            ) or add_circular(
                graph,
                f"doc:{title_key(m[1])}:{issued}",
                title=re.sub(r"\s+", " ", m[1]),
                doc_type="master_circular",
                issue_date=issued,
            )
            found.append((node_id, m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for m in DATED_CIRCULAR_RE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        candidates = graph["index"]["date"].get(parse_date(m[1]), set())
        if m[2]:
            subject = tokens(m[2])
            candidates = {
                c
                for c in candidates
                if len(subject & tokens(graph["circulars"][c].get("title") or ""))
                >= 0.5 * len(subject)
            }
        if len(candidates) == 1:
            found.append((next(iter(candidates)), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for m in REGULATIONS_RE.finditer(text):
        if free(m.start(), m.end()):
            title = re.sub(r"\s+", " ", m.group())
            node_id = add_circular(
                graph, f"reg:{title_key(title)}", title=title, doc_type="regulations"
            )
            found.append((node_id, m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for alias, node_id in aliases.items():
        for m in re.finditer(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
            if free(m.start(), m.end()):
                found.append((node_id, m.start(), m.end()))
    return sorted(found, key=lambda f: f[1])


def document_id(doc: dict) -> str:
    if doc["circular_number"]:
        return f"cir:{number_key(doc['circular_number'])}"
    return f"reg:{title_key(doc['title'])}"


def is_index_chunk(text: str) -> bool:
    lines = text.splitlines()
    short = [line for line in lines if NUMBERED_LINE_RE.match(line) and len(line) <= 80]
    return len(lines) >= 4 and len(short) >= 0.6 * len(lines)


def obligated_entities(text: str) -> dict[str, str]:
    found = {}
    for modal in MODAL_RE.finditer(text):
        start = max(text.rfind(c, 0, modal.start()) for c in ".;:\n,")
        subject = text[start + 1 : modal.start()]
        subject = re.split(r"\b(?:that|which|who|where|if|unless|provided)\b", subject)[
            -1
        ]
        if len(subject.split()) > 12:
            continue
        for m in ENTITY_RE.finditer(subject):
            before = [w for w in re.findall(r"[a-z]+", subject[: m.start()].lower())]
            before = [w for w in before if w not in DETERMINERS]
            if before and before[-1] in PREPOSITIONS:
                continue
            name = ENTITY_NAMES[int(m.lastgroup[1:])]
            end = text.find(".", modal.end())
            found.setdefault(
                name, text[start + 1 : end if end > 0 else None].strip()[:300]
            )
    return found


def appendix_rows(doc: dict) -> list[dict]:
    rows, inside = [], False
    for chunk in doc["chunks"]:
        for line in chunk["text"].splitlines():
            if (
                re.sub(r"\W", "", line)
                .upper()
                .startswith("APPENDIXLISTOFRESCINDEDCIRCULARS")
            ):
                inside = True
            elif inside and line.strip().upper() == "ANNEXURES":
                return rows
            elif inside and (m := APPENDIX_ROW_RE.match(normalize_citations(line))):
                rows.append(
                    {
                        "sr_no": int(m[1]),
                        "number": re.sub(
                            rf"(?:\s+{MONTH})+$", "", m[2], flags=re.IGNORECASE
                        ),
                        "issue_date": parse_date(m[3]),
                        "title": m[4].strip(),
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                    }
                )
    return rows


def provenance(doc: dict, chunk: dict, evidence: str, method: str) -> dict:
    return {
        "source_doc": doc["doc_id"],
        "source_chunk": chunk["chunk_id"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "evidence": re.sub(r"\s+", " ", evidence).strip()[:400],
        "method": method,
    }


def extract(docs: list[dict]) -> dict:
    graph = new_graph()
    doc_ids = {}
    for doc in docs:
        doc_ids[doc["doc_id"]] = add_circular(
            graph,
            document_id(doc),
            doc_id=doc["doc_id"],
            number=doc["circular_number"],
            title=doc["title"],
            doc_type=doc["doc_type"],
            issue_date=doc["date"],
            effective_from=doc["effective_date"],
            url=doc["source_url"],
            domain="mutual_funds",
            page_count=doc["page_count"],
            in_corpus=True,
        )
        graph["circulars"][doc_ids[doc["doc_id"]]]["in_corpus"] = True

    appendices = {}
    for doc in docs:
        for row in appendix_rows(doc):
            node_id = add_circular(
                graph,
                f"cir:{number_key(row['number'])}",
                number=row["number"],
                title=row["title"],
                doc_type="circular",
                issue_date=row["issue_date"],
                domain="mutual_funds",
            )
            node = graph["circulars"][node_id]
            node.setdefault("status", "rescinded")
            node.setdefault("status_source", f"{doc['doc_id']} p.{row['page_start']}")
            appendices.setdefault(doc["doc_id"], {})[row["sr_no"]] = (node_id, row)

    for doc in docs:
        for chunk in doc["chunks"]:
            for text in (chunk["text"], *chunk["footnotes"].values()):
                number_citations(graph, normalize_citations(text))

    numbered = {}
    for doc in docs:
        index = numbered.setdefault(doc["doc_id"], {})
        for chunk in doc["chunks"]:
            if not is_index_chunk(chunk["text"]):
                for m in NUMBERED_LINE_RE.finditer(chunk["text"]):
                    index.setdefault(m[1], chunk["chunk_id"])
    regulations_docs = {
        doc_ids[d["doc_id"]]: d["doc_id"]
        for d in docs
        if d["doc_type"] == "regulations"
    }

    for doc in docs:
        own = doc_ids[doc["doc_id"]]
        aliases = {}
        for chunk in doc["chunks"]:
            for m in ALIAS_RE.finditer(chunk["text"]):
                window = chunk["text"][max(0, m.start() - 200) : m.start()]
                cited = citations(graph, window, {})
                if cited and not window[cited[-1][2] :].strip():
                    aliases[m[1]] = cited[-1][0]

        last_amender, last_wef = None, None
        footnote_order = sorted(
            (
                (int(ref), chunk, text)
                for chunk in doc["chunks"]
                for ref, text in chunk["footnotes"].items()
            ),
            key=lambda item: item[0],
        )
        for ref, chunk, text in (
            footnote_order if doc["doc_type"] == "regulations" else []
        ):
            m = FOOTNOTE_AMENDMENT_RE.search(text)
            if m and m[2]:
                last_amender = add_circular(
                    graph,
                    f"reg:{title_key(m[2])}",
                    title=re.sub(r"\s+", " ", m[2]),
                    doc_type="regulations",
                )
            elif not (m and m[3]) and not IBID_RE.match(text):
                continue
            if not last_amender or last_amender == own:
                continue
            after = m.end() if m else 0
            wef = WEF_RE.search(text, after, after + 40)
            if m and m[2]:
                last_wef = parse_date(wef[1]) if wef else None
            add_edge(
                graph,
                "AMENDS",
                ("Circular", last_amender),
                ("Circular", own),
                {
                    **provenance(doc, chunk, text, "regex:footnote_amendment"),
                    "scope": "partial",
                    "action": (m[1] if m else "ibid").lower(),
                    "provision": chunk["chunk_id"],
                    "footnote": str(ref),
                    "effective_from": last_wef,
                },
                key=(chunk["chunk_id"], ref),
            )

        for chunk in doc["chunks"]:
            para = NUMBERED_LINE_RE.match(
                chunk["section"] or ""
            ) or NUMBERED_LINE_RE.match(chunk["text"])
            graph["provisions"].append(
                {
                    "id": chunk["chunk_id"],
                    "circular_id": own,
                    "section": chunk["section"],
                    "para_number": para[1] if para else None,
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    "footnotes": json.dumps(chunk["footnotes"], ensure_ascii=False),
                }
            )
            provision = ("Provision", chunk["chunk_id"])
            add_edge(graph, "PART_OF", provision, ("Circular", own), {})

            sources = [("text", normalize_citations(chunk["text"]))] + [
                (f"footnote {ref}", normalize_citations(text))
                for ref, text in chunk["footnotes"].items()
            ]
            for via, text in sources:
                for m in REGULATION_NUMBER_RE.finditer(text):
                    regs = f"reg:{title_key(m[2])}"
                    target_doc = regulations_docs.get(regs)
                    target_chunk = (
                        numbered.get(target_doc, {}).get(m[1]) if target_doc else None
                    )
                    if target_chunk and target_doc != doc["doc_id"]:
                        add_edge(
                            graph,
                            "REFERENCES",
                            provision,
                            ("Provision", target_chunk),
                            {
                                **provenance(
                                    doc,
                                    chunk,
                                    text[m.start() : m.end() + 80],
                                    "regex:regulation_number",
                                ),
                                "via": via,
                            },
                        )
                for node_id, start, end in citations(graph, text, aliases):
                    if node_id != own:
                        add_edge(
                            graph,
                            "REFERENCES",
                            provision,
                            ("Circular", node_id),
                            {
                                **provenance(
                                    doc,
                                    chunk,
                                    text[max(0, start - 60) : end + 40],
                                    "regex:citation",
                                ),
                                "via": via,
                            },
                        )
            for m in PARAGRAPH_RE.finditer(chunk["text"]):
                target_chunk = numbered[doc["doc_id"]].get(m[1])
                if target_chunk and target_chunk != chunk["chunk_id"]:
                    add_edge(
                        graph,
                        "REFERENCES",
                        provision,
                        ("Provision", target_chunk),
                        {
                            **provenance(doc, chunk, m.group(), "regex:paragraph"),
                            "via": "text",
                        },
                    )
            for name, evidence in obligated_entities(chunk["text"]).items():
                add_edge(
                    graph,
                    "APPLIES_TO",
                    provision,
                    ("Entity", name),
                    provenance(doc, chunk, evidence, "regex:obligation_subject"),
                )

        statement_chunks = doc["chunks"]
        if doc["doc_type"] == "master_circular":
            first_chapter = next(
                (
                    i
                    for i, c in enumerate(doc["chunks"])
                    if (c["section"] or "").upper().startswith("CHAPTER")
                ),
                len(doc["chunks"]),
            )
            statement_chunks = doc["chunks"][:first_chapter]
        clauses = defaultdict(set)
        for chunk in statement_chunks:
            for m in CLAUSE_RE.finditer(chunk["text"]):
                targets = (
                    {c[0] for c in citations(graph, m[3], aliases)} if m[3] else {None}
                )
                for target in targets:
                    clauses[target].add(m[1])

        own_node = graph["circulars"][own]
        for chunk in statement_chunks:
            text = re.sub(r"\b(No|Nos|Sr)\.\n", r"\1. ", chunk["text"])
            for paragraph in text.splitlines():
                for m in REPLACE_RE.finditer(paragraph):
                    for target, _, _ in citations(graph, m[1], aliases):
                        if target != own:
                            add_edge(
                                graph,
                                "SUPERSEDES",
                                ("Circular", own),
                                ("Circular", target),
                                {
                                    **provenance(
                                        doc, chunk, m.group(), "regex:replace"
                                    ),
                                    "effective_from": own_node.get("effective_from"),
                                },
                            )
                for m in REPEAL_RE.finditer(paragraph):
                    target = f"reg:{title_key(m[1])}"
                    if target != own:
                        add_circular(
                            graph,
                            target,
                            title=re.sub(r"\s+", " ", m[1]),
                            doc_type="regulations",
                        )
                        on_commencement = re.search(
                            r"coming\s+into\s+force\s+of\s+these\s+regulations",
                            m[2],
                            re.IGNORECASE,
                        )
                        add_edge(
                            graph,
                            "SUPERSEDES",
                            ("Circular", own),
                            ("Circular", target),
                            {
                                **provenance(doc, chunk, m.group(), "regex:repeal"),
                                "effective_from": own_node.get("effective_from")
                                if on_commencement
                                else None,
                            },
                        )
                for m in RESCIND_RANGE_RE.finditer(paragraph):
                    for sr_no in range(int(m[1]), int(m[2]) + 1):
                        if sr_no not in appendices.get(doc["doc_id"], {}):
                            continue
                        target, row = appendices[doc["doc_id"]][sr_no]
                        add_edge(
                            graph,
                            "RESCINDS",
                            ("Circular", own),
                            ("Circular", target),
                            {
                                **provenance(
                                    doc, chunk, m.group(), "regex:appendix_rescission"
                                ),
                                "effective_from": own_node.get("effective_from"),
                                "appendix_sr_no": sr_no,
                                "appendix_pages": [row["page_start"], row["page_end"]],
                            },
                        )
                for pattern, method in (
                    (MODIFIED_RE, "regex:modified"),
                    (UNCHANGED_RE, "regex:remain_unchanged"),
                ):
                    for m in pattern.finditer(paragraph):
                        cited = [c[0] for c in citations(graph, m[1], aliases)]
                        for target in cited[-1:] if pattern is MODIFIED_RE else cited:
                            if target == own:
                                continue
                            add_edge(
                                graph,
                                "AMENDS",
                                ("Circular", own),
                                ("Circular", target),
                                {
                                    **provenance(doc, chunk, m.group(), method),
                                    "scope": "partial",
                                    "clauses": sorted(clauses[target] | clauses[None])
                                    or None,
                                    "effective_from": own_node.get("effective_from"),
                                },
                            )

    derive_status(graph)
    return graph


def derive_status(graph: dict) -> None:
    circulars = graph["circulars"]
    for (rel, start, end, _), props in graph["edges"].items():
        if rel not in ("SUPERSEDES", "RESCINDS", "AMENDS"):
            continue
        target = circulars[end[1]]
        if rel == "AMENDS":
            target["status"] = target.get("status") or "amended"
            continue
        status = "superseded" if rel == "SUPERSEDES" else "rescinded"
        if target.get("status") in (None, "amended", "rescinded"):
            target["status"] = status
            target["status_source"] = f"{props['source_doc']} p.{props['page_start']}"
        ends = props.get("effective_from")
        if ends and (
            target.get("effective_to") is None or ends < target["effective_to"]
        ):
            target["effective_to"] = ends


def load_documents(processed_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text()) for path in sorted(processed_dir.glob("*.json"))
    ]


def connect(settings: Settings) -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        notifications_min_severity="OFF",
    )


def write(driver: Driver, graph: dict, database: str) -> dict[str, int]:
    entities = [
        {"id": name, "name": name, "type": kind} for name, (kind, _) in ENTITIES.items()
    ]
    circulars = [
        {**node, "date_conflicts": node["date_conflicts"] or None}
        for node in graph["circulars"].values()
    ]
    grouped = defaultdict(list)
    for (rel, start, end, _), props in graph["edges"].items():
        grouped[(rel, start[0], end[0])].append(
            {"start": start[1], "end": end[1], "props": props}
        )
    with driver.session(database=database) as session:
        session.run(
            "MATCH (n) WHERE n:Circular OR n:Provision OR n:Entity DETACH DELETE n"
        )
        for label in ("Circular", "Provision", "Entity"):
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )
        for name, label, fields in (
            ("circular_text", "Circular", "n.title, n.number"),
            ("provision_text", "Provision", "n.section, n.text"),
        ):
            session.run(
                f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS FOR (n:{label}) ON EACH [{fields}] "
                "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}"
            )
        for label, rows in (
            ("Circular", circulars),
            ("Provision", graph["provisions"]),
            ("Entity", entities),
        ):
            session.run(
                f"UNWIND $rows AS row CREATE (n:{label}) SET n = row", rows=rows
            )
        for (rel, start_label, end_label), rows in grouped.items():
            session.run(
                f"UNWIND $rows AS row MATCH (a:{start_label} {{id: row.start}}) "
                f"MATCH (b:{end_label} {{id: row.end}}) CREATE (a)-[r:{rel}]->(b) SET r = row.props",
                rows=rows,
            )
        session.run("CALL db.awaitIndexes(120)")
    counts = defaultdict(int)
    for (rel, _, _), rows in grouped.items():
        counts[rel] += len(rows)
    return dict(counts)


CHECKS = {
    "provision without exactly one PART_OF": """
        MATCH (p:Provision) WITH p, COUNT { (p)-[:PART_OF]->(:Circular) } AS n
        WHERE n <> 1 RETURN p.id AS item""",
    "relationship without provenance": """
        MATCH ()-[r]->() WHERE type(r) <> 'PART_OF'
          AND (r.source_doc IS NULL OR r.page_start IS NULL OR r.evidence IS NULL)
        RETURN type(r) + ' ' + elementId(r) AS item""",
    "provenance page outside source document": """
        MATCH ()-[r]->() WHERE r.source_doc IS NOT NULL
        OPTIONAL MATCH (c:Circular {doc_id: r.source_doc})
        WITH r, c WHERE c IS NULL OR r.page_start < 1 OR r.page_end > c.page_count
        RETURN type(r) + ' from ' + r.source_doc + ' p.' + toString(r.page_start) AS item""",
    "self-referencing lifecycle relationship": """
        MATCH (c:Circular)-[r:SUPERSEDES|RESCINDS|AMENDS]->(c) RETURN c.id + ' ' + type(r) AS item""",
    "supersession/rescission cycle": """
        MATCH (c:Circular)-[:SUPERSEDES|RESCINDS*1..10]->(c) RETURN DISTINCT c.id AS item""",
    "circular superseded by more than one circular": """
        MATCH (old:Circular)<-[:SUPERSEDES]-(new:Circular)
        WITH old, collect(DISTINCT new.id) AS successors WHERE size(successors) > 1
        RETURN old.id + ' <- ' + reduce(s = '', x IN successors | s + x + ' ') AS item""",
    "effective_to before effective_from": """
        MATCH (c:Circular) WHERE c.effective_to IS NOT NULL AND c.effective_from IS NOT NULL
          AND c.effective_to < c.effective_from RETURN c.id AS item""",
    "superseded/rescinded circular whose end date disagrees with successor start": """
        MATCH (new:Circular)-[r:SUPERSEDES|RESCINDS]->(old:Circular)
        WHERE r.effective_from IS NOT NULL AND old.effective_to <> r.effective_from
          AND old.effective_to > r.effective_from
        RETURN old.id + ' ends ' + old.effective_to + ' but ' + new.id + ' starts ' + r.effective_from AS item""",
    "lifecycle target with no status": """
        MATCH (:Circular)-[:SUPERSEDES|RESCINDS|AMENDS]->(c:Circular) WHERE c.status IS NULL
        RETURN DISTINCT c.id AS item""",
}
WARNINGS = {
    "circular cited with conflicting dates in the source text": """
        MATCH (c:Circular) WHERE c.date_conflicts IS NOT NULL
        RETURN c.number + ' ' + c.issue_date + ' vs ' + reduce(s = '', d IN c.date_conflicts | s + d + ' ') AS item""",
    "possible duplicate circulars (same date and digits)": """
        MATCH (c:Circular) WHERE c.number IS NOT NULL AND c.issue_date IS NOT NULL
        WITH c, reduce(s = '', ch IN split(c.number, '') |
               CASE WHEN ch >= '0' AND ch <= '9' THEN s + ch ELSE s END) AS digits
        WITH c.issue_date AS issued, digits, collect(c.number) AS numbers WHERE size(numbers) > 1
        RETURN reduce(s = '', n IN numbers | s + n + ' ~ ') AS item""",
    "superseded or rescinded circular without an end date": """
        MATCH (c:Circular) WHERE c.status IN ['superseded', 'rescinded'] AND c.effective_to IS NULL
        RETURN c.id AS item""",
}


def validate(driver: Driver, database: str) -> tuple[dict, dict[str, list[str]]]:
    with driver.session(database=database) as session:
        nodes = {
            r["label"]: r["n"]
            for r in session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n"
            )
        }
        rels = {
            r["rel"]: r["n"]
            for r in session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n"
            )
        }
        issues = {
            (severity, name): [r["item"] for r in session.run(query)]
            for severity, checks in (("error", CHECKS), ("warning", WARNINGS))
            for name, query in checks.items()
        }
    return {"nodes": nodes, "relationships": rels}, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate the SEBI regulatory graph."
    )
    parser.add_argument("command", choices=["build", "validate"])
    args = parser.parse_args()
    settings = Settings()
    with connect(settings) as driver:
        if args.command == "build":
            docs = load_documents(OUT_DIR)
            if not docs:
                parser.exit(
                    1,
                    f"no processed documents in {OUT_DIR}; run inforce.ingest first\n",
                )
            graph = extract(docs)
            counts = write(driver, graph, settings.neo4j_database)
            print(
                f"wrote {len(graph['circulars'])} circulars, {len(graph['provisions'])} provisions"
            )
            print(json.dumps(counts, indent=2))
        summary, issues = validate(driver, settings.neo4j_database)
    print(json.dumps(summary, indent=2))
    failed = False
    for (severity, name), items in issues.items():
        failed = failed or (severity == "error" and bool(items))
        status = "ok  " if not items else ("FAIL" if severity == "error" else "WARN")
        print(
            f"{status} {name}" + (f": {len(items)} e.g. {items[:3]}" if items else "")
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
