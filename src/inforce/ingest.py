import csv
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
MANIFEST = Path("data/corpus.csv")

MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 300
MARKER_SIZE_RATIO = 0.75

MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
MONTH = rf"(?:{'|'.join(MONTHS)})"
DATE = (
    rf"(?:(?P<month1>{MONTH})\s+(?P<day1>\d{{1,2}}),?\s+(?P<year1>\d{{4}})"
    rf"|(?P<day2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month2>{MONTH}),?\s+(?P<year2>\d{{4}}))"
)
DATE_RE = re.compile(rf"\b{DATE}\b", re.IGNORECASE)
EFFECTIVE_RE = re.compile(
    rf"\b(?:this|these)\s+(?:master\s+)?(?:circular|regulations)\b[^.;]{{0,60}}?"
    rf"\b(?:come into force|applicable|effective|take effect)\b[^.;]{{0,30}}?\b(?:from|on)\s+{DATE}",
    re.IGNORECASE,
)
CIRCULAR_NUMBER_RE = re.compile(
    r"\bHO/\d+/\d+/\d+\(\d+\)\d{4}-[\w-]+/\w+/\d+/\d{4}|\b(?:[\w-]+/)*CIR/[\w/-]*\d"
)
SUBJECT_RE = re.compile(r"\bSub(?:ject)?\s*:\s*(.+)", re.IGNORECASE)
LIST_MARKER_RE = re.compile(
    r"\(?(?:\d{1,3}[A-Z]?(?:\.\d{1,3})*|[a-zA-Z]{1,2}|[ivxlcIVXLC]{1,6})[.)]"
)
ITEM_START_RE = re.compile(rf"{LIST_MARKER_RE.pattern}\s+[A-Z“‘\"(]")
PAGE_NUMBER_RE = re.compile(r"page\s*#(?:\s*of\s*#)?", re.IGNORECASE)
HEADING_WORD_RE = re.compile(
    r"\b(?:CHAPTER|SCHEDULE|ANNEXURE|APPENDIX)\b", re.IGNORECASE
)
TOC_LEADER_RE = re.compile(r"\.{6,}\s*\d*$")
FOOTNOTE_REF_RE = re.compile(r"\[\^(\d+)\]")


def read_pages(path: Path) -> list[list[dict]]:
    try:
        doc = pymupdf.open(path)
    except (pymupdf.FileNotFoundError, pymupdf.FileDataError) as exc:
        raise ValueError(f"unreadable PDF: {exc}") from exc
    with doc:
        if not doc.is_pdf:
            raise ValueError("not a PDF")
        if doc.needs_pass:
            raise ValueError("encrypted PDF")
        pages = []
        for page in doc:
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [
                        (
                            s["text"],
                            s["size"],
                            bool(s["flags"] & 16) or "bold" in s["font"].lower(),
                        )
                        for s in line["spans"]
                        if s["text"].strip()
                    ]
                    if spans:
                        top = line["bbox"][1] / page.rect.height
                        lines.append(
                            {"spans": spans, "top": top, "block": block["number"]}
                        )
            pages.append(lines)
    return pages


def span_text(spans) -> str:
    return "".join(text for text, _, _ in spans)


def is_marker(span, reference_size: float) -> bool:
    text, size, _ = span
    return text.strip().isdigit() and size <= MARKER_SIZE_RATIO * reference_size


def drop_running_lines(pages: list[list[dict]]) -> list[list[dict]]:
    def key(line):
        return span_text(line["spans"]).strip(), round(line["top"], 2)

    counts = Counter(k for lines in pages for k in {key(line) for line in lines})
    min_repeats = max(3, 0.05 * len(pages))

    def is_running(line) -> bool:
        text, top = key(line)
        if counts[(text, top)] >= min_repeats and sum(c.isalpha() for c in text) >= 4:
            return True
        pattern = re.sub(r"\d+", "#", text)
        if PAGE_NUMBER_RE.fullmatch(pattern):
            return top < 0.1 or top > 0.8
        return pattern == "#" and (top < 0.07 or top > 0.93)

    return [[line for line in lines if not is_running(line)] for lines in pages]


def split_footnotes(
    lines: list[dict], body_size: float
) -> tuple[list[dict], dict[str, str], str]:
    body_tops = [
        line["top"]
        for line in lines
        if max(s for _, s, _ in line["spans"]) >= 0.9 * body_size
    ]
    region = sorted(
        (line for line in lines if line["top"] > max(body_tops, default=1)),
        key=lambda l: l["top"],
    )
    if not region:
        return lines, {}, ""
    region_size = max(s for line in region for _, s, _ in line["spans"])
    notes: dict[str, str] = {}
    carried, current = "", None
    for line in region:
        if is_marker(line["spans"][0], region_size):
            current = line["spans"][0][0].strip()
            notes[current] = span_text(line["spans"][1:])
        elif current is None:
            carried += " " + span_text(line["spans"])
        else:
            notes[current] += " " + span_text(line["spans"])
    if not notes:
        return lines, {}, ""
    body = [line for line in lines if line["top"] <= max(body_tops)]
    clean = {ref: re.sub(r"\s+", " ", text).strip() for ref, text in notes.items()}
    return body, clean, re.sub(r"\s+", " ", carried).strip()


def line_text(spans) -> tuple[str, bool]:
    largest = max(size for _, size, _ in spans)
    parts, bold = [], True
    for span in spans:
        if is_marker(span, largest):
            parts.append(f"[^{span[0].strip()}]")
        else:
            parts.append(span[0])
            bold = bold and span[2]
    text = "".join("•" if 0xE000 <= ord(c) <= 0xF8FF else c for c in "".join(parts))
    text = re.sub(r"(\[\^\d+\])(?=[^\s.,;:)\]])", r"\1 ", text).strip()
    text = re.sub(rf"^({LIST_MARKER_RE.pattern})(?=[A-Z“‘\"(])", r"\1 ", text)
    return re.sub(r"\s+", " ", text), bold


def build_paragraphs(pages: list[list[dict]]) -> list[dict]:
    paragraphs: list[dict] = []
    current, pending_marker = None, None
    for page_number, lines in enumerate(pages, start=1):
        block = None
        for line in lines:
            text, bold = line_text(line["spans"])
            if LIST_MARKER_RE.fullmatch(text):
                pending_marker = f"{pending_marker} {text}" if pending_marker else text
                current = None
                continue
            continues = (
                current is not None
                and pending_marker is None
                and (
                    (line["block"] == block and not ITEM_START_RE.match(text))
                    or (
                        not LIST_MARKER_RE.match(text)
                        and not (bold or current["bold"])
                        and not current["text"].endswith((".", ":", ";", "?"))
                        and (block is not None or text[:1].islower())
                    )
                )
            )
            if continues:
                glue = "" if re.search(r"\w[-/]$", current["text"]) else " "
                current["text"] += glue + text
                current["bold"] = current["bold"] and bold
                current["page_end"] = page_number
            else:
                current = {
                    "text": f"{pending_marker} {text}" if pending_marker else text,
                    "bold": bold,
                    "page_start": page_number,
                    "page_end": page_number,
                }
                paragraphs.append(current)
                pending_marker = None
            block = line["block"]
    return paragraphs


def is_foreign(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    foreign = sum(0x0900 <= ord(c) <= 0x097F or 0xC0 <= ord(c) <= 0xFF for c in letters)
    return foreign > 0.1 * len(letters)


def to_iso(match: re.Match) -> str | None:
    month = MONTHS.index((match["month1"] or match["month2"]).lower()) + 1
    try:
        return date(
            int(match["year1"] or match["year2"]),
            month,
            int(match["day1"] or match["day2"]),
        ).isoformat()
    except ValueError:
        return None


def extract_metadata(paragraphs: list[dict]) -> dict:
    texts = [p["text"] for p in paragraphs]
    first_page = sum(p["page_start"] == paragraphs[0]["page_start"] for p in paragraphs)
    first_body = next(
        (i for i, text in enumerate(texts) if len(text) >= 200), len(texts)
    )
    subject = next(
        (i for i, text in enumerate(texts[:first_page]) if SUBJECT_RE.search(text)),
        None,
    )
    title, header = None, " ".join(texts[:first_body])
    if subject is not None:
        header = " ".join(texts[: subject + 1])
        parts = [SUBJECT_RE.search(texts[subject])[1]]
        for text in texts[subject + 1 : subject + 3]:
            if LIST_MARKER_RE.match(text) or len(text) > 250:
                break
            parts.append(text)
        title = " ".join(parts).strip()
    number = CIRCULAR_NUMBER_RE.search(header)
    issued = DATE_RE.search(header)
    body = " ".join(p["text"] for p in paragraphs)
    effective = {to_iso(m) for m in EFFECTIVE_RE.finditer(body)} - {None}
    return {
        "title": title,
        "circular_number": number[0] if number else None,
        "date": to_iso(issued) if issued else None,
        "effective_date": effective.pop() if len(effective) == 1 else None,
    }


def split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    units = []
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        while len(sentence) > MAX_CHUNK_CHARS:
            cut = sentence.rfind(" ", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > 0 else MAX_CHUNK_CHARS
            units.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        units.append(sentence)
    return units


def resolve_footnotes(text: str, page_notes: list[dict]) -> tuple[str, dict[str, str]]:
    footnotes: dict[str, str] = {}

    def lookup(ref: str) -> str | None:
        return next((notes[ref] for notes in page_notes if ref in notes), None)

    def replace(match: re.Match) -> str:
        ref = match[1]
        if lookup(ref):
            footnotes[ref] = lookup(ref)
            return match[0]
        for i in range(1, len(ref)):
            left, right = ref[:i], ref[i:]
            if lookup(left) and lookup(right):
                footnotes[left], footnotes[right] = lookup(left), lookup(right)
                return f"[^{left}][^{right}]"
        return match[0]

    return FOOTNOTE_REF_RE.sub(replace, text), footnotes


def build_chunks(paragraphs: list[dict], notes: list[dict], doc_id: str) -> list[dict]:
    chunks: list[dict] = []
    current, section = None, None
    for paragraph in paragraphs:
        if paragraph["heading"]:
            section = FOOTNOTE_REF_RE.sub("", paragraph["text"]).strip()
            if (
                current
                and not current["heading_only"]
                and len(current["text"]) >= MIN_CHUNK_CHARS
            ):
                current = None
        for i, unit in enumerate(split_long(paragraph["text"])):
            if current and (
                len(current["text"]) + len(unit) + 1 > MAX_CHUNK_CHARS
                or paragraph["page_start"] > current["page_end"] + 1
            ):
                current = None
            if current is None:
                current = {
                    "section": section,
                    "page_start": paragraph["page_start"],
                    "page_end": paragraph["page_end"],
                    "text": unit,
                    "heading_only": paragraph["heading"],
                }
                chunks.append(current)
                continue
            current["text"] += (" " if i else "\n") + unit
            current["page_end"] = paragraph["page_end"]
            if current["heading_only"] and paragraph["heading"]:
                current["section"] = section
            current["heading_only"] = current["heading_only"] and paragraph["heading"]

    result = []
    for index, chunk in enumerate(chunks):
        text, footnotes = resolve_footnotes(
            chunk["text"], notes[chunk["page_start"] - 1 : chunk["page_end"]]
        )
        result.append(
            {
                "chunk_id": f"{doc_id}-{index:04d}",
                "section": chunk["section"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": text,
                "footnotes": footnotes,
            }
        )
    return result


def ingest(path: Path, source: dict) -> dict:
    pages = drop_running_lines(read_pages(path))
    sizes = Counter()
    for lines in pages:
        for line in lines:
            for text, size, _ in line["spans"]:
                sizes[round(size, 1)] += len(text.strip())
    if not sizes:
        raise ValueError("no extractable text (scanned, image-only or damaged PDF)")
    body_size = sizes.most_common(1)[0][0]

    body_pages, notes = [], []
    for lines in pages:
        body, page_notes, carried = split_footnotes(lines, body_size)
        previous = notes[-1] if notes else None
        if carried and previous:
            last = next(reversed(previous))
            previous[last] = f"{previous[last]} {carried}"
        body_pages.append(body)
        notes.append(page_notes)

    paragraphs = [
        p
        for p in build_paragraphs(body_pages)
        if any(c.isalpha() for c in p["text"])
        and not is_foreign(p["text"])
        and not TOC_LEADER_RE.search(p["text"])
    ]
    if not paragraphs:
        raise ValueError("no extractable English text")
    for p in paragraphs:
        p["heading"] = (p["bold"] and len(p["text"]) <= 150) or (
            len(p["text"]) <= 60 and bool(HEADING_WORD_RE.search(p["text"]))
        )
    metadata = extract_metadata(paragraphs)
    return {
        "doc_id": path.stem,
        "file": path.name,
        "source_url": source.get("source_url") or None,
        "doc_type": source.get("doc_type") or None,
        **metadata,
        "title": source.get("title") or metadata["title"],
        "page_count": len(pages),
        "chunks": build_chunks(paragraphs, notes, path.stem),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    with MANIFEST.open(newline="") as f:
        for source in csv.DictReader(f):
            path = RAW_DIR / source["file"]
            output = OUT_DIR / f"{path.stem}.json"
            try:
                doc = ingest(path, source)
            except ValueError as exc:
                failed += 1
                output.unlink(missing_ok=True)
                print(f"FAILED {path}: {exc}", file=sys.stderr)
                continue
            output.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
            print(f"{path}: {doc['page_count']} pages -> {len(doc['chunks'])} chunks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
