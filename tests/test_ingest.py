import json

import pymupdf
import pytest

from inforce import ingest as ingest_module
from inforce.ingest import MAX_CHUNK_CHARS, build_chunks, extract_metadata, ingest, main

REGULAR, BOLD = pymupdf.Font("helv"), pymupdf.Font("hebo")


def footer(page_number):
    return [
        ((250, 800), f"Page {page_number} of 3", 11, REGULAR),
        (
            (150, 815),
            "Master Circular for Mutual Funds as on March 20, 2026",
            11,
            REGULAR,
        ),
    ]


SAMPLE_PAGES = [
    [
        ((72, 80), "CIRCULAR", 12, BOLD),
        ((72, 110), "HO/24/13/11(1)2026-IMD-POD-1/I/7602/2026", 12, REGULAR),
        ((72, 130), "March 20, 2026", 12, REGULAR),
        ((72, 170), "Subject: Master Circular for Mutual Funds", 12, REGULAR),
        (
            (72, 210),
            "1. This Master Circular shall come into force with effect from April 01, 2026.",
            12,
            REGULAR,
        ),
        ((72, 250), "1.1. Filing of Offer Document", 12, BOLD),
        ((72, 280), "1.1.1. The Offer Document shall have two parts", 12, REGULAR),
        (None, "1", 7, REGULAR),
        (None, " and shall be filed with the Board.", 12, REGULAR),
        ((72, 740), "1", 5, REGULAR),
        (
            None,
            " SEBI Circular No. SEBI/HO/IMD/DF2/CIR/P/2019/17 dated January 16, 2019",
            8,
            REGULAR,
        ),
        *footer(1),
    ],
    [
        ((72, 80), "1.2. Easy Availability of Offer Document", 12, BOLD),
        (
            (72, 110),
            "1.2.1. Trustees and AMCs shall ensure that the SID and SAI are available.",
            12,
            REGULAR,
        ),
        *footer(2),
    ],
    [
        ((72, 80), "1.3. New Fund Offer Period", 12, BOLD),
        (
            (72, 110),
            "1.3.1. The NFO shall be open for a minimum period of 3 working days.",
            12,
            REGULAR,
        ),
        *footer(3),
    ],
]


def write_pdf(path, pages, **save_options):
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        for position, text, size, font in items:
            writer.append(position or writer.last_point, text, font=font, fontsize=size)
        writer.write_text(page)
    doc.save(path, **save_options)
    doc.close()


def paragraph(text, page, heading=False):
    return {
        "text": text,
        "page_start": page,
        "page_end": page,
        "heading": heading,
        "bold": heading,
    }


@pytest.mark.parametrize(
    "number_line",
    [
        "SEBI/HO/IMD/IMD-I/DOF5/P/CIR/2021/553 April 28, 2021",
        "HO/24/11/24(62)2026-IMD-RAC4/I/11872/2026 April 28, 2021",
    ],
)
def test_extract_metadata_from_circular_header(number_line):
    paragraphs = [
        paragraph("CIRCULAR", 1),
        paragraph(number_line, 1),
        paragraph("Sir/Madam,", 1),
        paragraph("Subject: Alignment of interest of Key Employees", 1),
        paragraph("with the Unitholders of the Mutual Fund Schemes", 1),
        paragraph(
            "1. The provisions of this circular shall be applicable with effect from July 01, 2021.",
            1,
        ),
    ]

    assert extract_metadata(paragraphs) == {
        "title": "Alignment of interest of Key Employees with the Unitholders of the Mutual Fund Schemes",
        "circular_number": number_line.split()[0],
        "date": "2021-04-28",
        "effective_date": "2021-07-01",
    }


@pytest.mark.parametrize(
    "body",
    [
        "The said amendments shall be applicable from April 01, 2025.",
        (
            "This circular shall be applicable from April 01, 2025. "
            "Clause 3 of this circular shall come into force with effect from June 01, 2025."
        ),
    ],
)
def test_effective_date_is_not_guessed(body):
    paragraphs = [paragraph("Subject: Ease of doing business", 1), paragraph(body, 1)]

    assert extract_metadata(paragraphs)["effective_date"] is None


def test_build_chunks_follows_sections_size_limit_and_page_gaps():
    body = ("The AMC shall comply with the requirement. " * 10).strip()
    paragraphs = [
        paragraph("1. Scheme Documents", 1, heading=True),
        paragraph(body, 1),
        paragraph("2. Short Section", 2, heading=True),
        paragraph("Tiny.", 2),
        paragraph("3. Next Section", 2, heading=True),
        paragraph(" ".join([body] * 4), 2),
        paragraph(body, 5),
    ]

    chunks = build_chunks(paragraphs, [{}] * 5, "doc")

    assert all(len(c["text"]) <= MAX_CHUNK_CHARS for c in chunks)
    assert chunks[0]["section"] == "1. Scheme Documents"
    assert chunks[0]["text"].startswith("1. Scheme Documents\nThe AMC")
    assert "Tiny.\n3. Next Section" in chunks[1]["text"]
    assert chunks[-1]["page_start"] == 5
    assert chunks[-2]["page_end"] == 2
    assert [c["chunk_id"] for c in chunks] == [
        f"doc-{i:04d}" for i in range(len(chunks))
    ]


def test_ingest_pdf_removes_page_furniture_and_resolves_footnotes(tmp_path):
    path = tmp_path / "master-circular.pdf"
    write_pdf(path, SAMPLE_PAGES)

    doc = ingest(path, {"doc_type": "master_circular", "source_url": "", "title": ""})

    assert {k: doc[k] for k in ("doc_id", "doc_type", "source_url", "page_count")} == {
        "doc_id": "master-circular",
        "doc_type": "master_circular",
        "source_url": None,
        "page_count": 3,
    }
    assert (
        doc["title"],
        doc["circular_number"],
        doc["date"],
        doc["effective_date"],
    ) == (
        "Master Circular for Mutual Funds",
        "HO/24/13/11(1)2026-IMD-POD-1/I/7602/2026",
        "2026-03-20",
        "2026-04-01",
    )
    text = "\n".join(c["text"] for c in doc["chunks"])
    assert "Page 1 of 3" not in text
    assert "as on March 20, 2026" not in text
    assert "SEBI/HO/IMD/DF2/CIR/P/2019/17" not in text

    [marked] = [c for c in doc["chunks"] if "[^1]" in c["text"]]
    assert "two parts[^1] and shall be filed with the Board." in marked["text"]
    assert marked["footnotes"] == {
        "1": "SEBI Circular No. SEBI/HO/IMD/DF2/CIR/P/2019/17 dated January 16, 2019"
    }
    assert marked["page_start"] == 1
    assert next(c for c in doc["chunks"] if "1.3.1." in c["text"])["page_end"] == 3


@pytest.mark.parametrize("kind", ["missing", "empty", "garbage", "blank", "encrypted"])
def test_unusable_pdfs_raise_value_error(tmp_path, kind):
    path = tmp_path / f"{kind}.pdf"
    if kind == "empty":
        path.write_bytes(b"")
    elif kind == "garbage":
        path.write_bytes(b"%PDF-1.7 this is not a real pdf")
    elif kind == "blank":
        write_pdf(path, [[]])
    elif kind == "encrypted":
        write_pdf(
            path,
            SAMPLE_PAGES,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )

    with pytest.raises(ValueError):
        ingest(path, {})


def test_main_writes_good_documents_and_reports_failures(tmp_path, monkeypatch, capsys):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    write_pdf(raw / "good.pdf", SAMPLE_PAGES)
    (raw / "broken.pdf").write_bytes(b"not a pdf")
    manifest = tmp_path / "corpus.csv"
    manifest.write_text(
        "file,doc_type,title,source_url\n"
        "good.pdf,master_circular,,https://www.sebi.gov.in/good.pdf\n"
        "broken.pdf,circular,,\n"
        "missing.pdf,circular,,\n"
    )
    monkeypatch.setattr(ingest_module, "RAW_DIR", raw)
    monkeypatch.setattr(ingest_module, "OUT_DIR", out)
    monkeypatch.setattr(ingest_module, "MANIFEST", manifest)

    assert main() == 1
    good = json.loads((out / "good.json").read_text())
    assert good["source_url"] == "https://www.sebi.gov.in/good.pdf"
    assert good["chunks"]
    assert not (out / "broken.json").exists()
    errors = capsys.readouterr().err
    assert "FAILED" in errors and "broken.pdf" in errors and "missing.pdf" in errors
