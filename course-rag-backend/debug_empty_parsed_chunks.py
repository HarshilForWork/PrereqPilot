from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz

try:
    import pdfplumber
except Exception:
    pdfplumber = None

from src.processing.chunker import chunk_documents_optimized
from src.processing.parser import (
    extract_tables_with_pdfplumber,
    extract_text_with_pymupdf,
    parse_document_hybrid,
)


EMPTY_TEXT_BLOCKS_RE = re.compile(r"^Text Blocks:\s*0\s*$", re.IGNORECASE | re.MULTILINE)
HEADER_RE = re.compile(r"^([A-Za-z ]+):\s*(.+)$")


def ensure_utf8_console() -> None:
    # The parser prints unicode symbols; this avoids Windows charmap encode failures.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def find_empty_parsed_files(output_dir: Path) -> list[Path]:
    targets: list[Path] = []
    for parsed_file in sorted(output_dir.glob("*_parsed.txt")):
        content = parsed_file.read_text(encoding="utf-8", errors="ignore")
        if EMPTY_TEXT_BLOCKS_RE.search(content):
            targets.append(parsed_file)
    return targets


def read_parsed_header(parsed_file: Path) -> dict[str, Any]:
    header: dict[str, Any] = {}
    for line in parsed_file.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]:
        match = HEADER_RE.match(line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        header[key] = value
    return header


def get_fitz_page_stats(pdf_path: Path) -> dict[str, Any]:
    page_stats: list[dict[str, Any]] = []
    doc = fitz.open(str(pdf_path))

    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            words = page.get_text("words") or []
            blocks = page.get_text("blocks") or []
            images = page.get_images(full=True) or []
            drawings = page.get_drawings() or []
            fonts = page.get_fonts() or []

            page_stats.append(
                {
                    "page": i,
                    "chars": len(text.strip()),
                    "words": len(words),
                    "blocks": len(blocks),
                    "images": len(images),
                    "drawings": len(drawings),
                    "fonts": len(fonts),
                }
            )
    finally:
        doc.close()

    return {
        "total_pages": len(page_stats),
        "total_chars": sum(p["chars"] for p in page_stats),
        "total_words": sum(p["words"] for p in page_stats),
        "total_blocks": sum(p["blocks"] for p in page_stats),
        "total_images": sum(p["images"] for p in page_stats),
        "total_drawings": sum(p["drawings"] for p in page_stats),
        "total_fonts": sum(p["fonts"] for p in page_stats),
        "pages": page_stats,
    }


def get_pdfplumber_page_stats(pdf_path: Path) -> dict[str, Any]:
    if pdfplumber is None:
        return {
            "available": False,
            "note": "pdfplumber import failed",
            "total_pages": 0,
            "total_chars": 0,
            "total_words": 0,
            "total_tables": 0,
            "pages": [],
        }

    page_stats: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    words = page.extract_words(keep_blank_chars=False, use_text_flow=True) or []
                except Exception:
                    words = []

                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                try:
                    tables = page.find_tables() or []
                except Exception:
                    tables = []

                page_stats.append(
                    {
                        "page": i,
                        "chars": len(text.strip()),
                        "words": len(words),
                        "tables": len(tables),
                    }
                )
    except Exception as exc:
        return {
            "available": True,
            "error": str(exc),
            "total_pages": 0,
            "total_chars": 0,
            "total_words": 0,
            "total_tables": 0,
            "pages": [],
        }

    return {
        "available": True,
        "total_pages": len(page_stats),
        "total_chars": sum(p["chars"] for p in page_stats),
        "total_words": sum(p["words"] for p in page_stats),
        "total_tables": sum(p["tables"] for p in page_stats),
        "pages": page_stats,
    }


def preview_chunks(
    chunks: list[dict[str, Any]],
    preview_chars: int,
    max_chunk_preview: int,
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for chunk in chunks[:max_chunk_preview]:
        content = " ".join((chunk.get("content") or "").split())
        if len(content) > preview_chars:
            content = content[:preview_chars].rstrip() + "..."

        previews.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "chunk_index": chunk.get("chunk_index", -1),
                "content_type": chunk.get("metadata", {}).get("content_type", "unknown"),
                "char_count": chunk.get("char_count", len(chunk.get("content", ""))),
                "preview": content,
            }
        )
    return previews


def infer_likely_issue(
    parser_meta: dict[str, Any],
    fitz_stats: dict[str, Any],
    plumber_stats: dict[str, Any],
) -> str:
    parser_text_blocks = int(parser_meta.get("text_elements", 0) or 0)
    parser_tables = int(parser_meta.get("table_elements", 0) or 0)
    fitz_words = int(fitz_stats.get("total_words", 0) or 0)
    fitz_images = int(fitz_stats.get("total_images", 0) or 0)
    fitz_drawings = int(fitz_stats.get("total_drawings", 0) or 0)
    fitz_fonts = int(fitz_stats.get("total_fonts", 0) or 0)
    plumber_available = bool(plumber_stats.get("available", False))
    plumber_words = int(plumber_stats.get("total_words", 0) or 0) if plumber_available else 0

    if parser_text_blocks == 0 and parser_tables == 0 and fitz_words == 0 and plumber_words == 0:
        if fitz_drawings > 0 and fitz_fonts == 0:
            return (
                "PDF appears to be vector outlines (many drawing paths, no fonts/text layer). "
                "Text was likely converted to shapes, so standard extraction cannot recover it."
            )
        if fitz_images > 0:
            return "No embedded text detected by either engine. Likely scanned or image-only PDF."
        return "No embedded text detected by either engine and no usable text layer was found."
    if parser_text_blocks == 0 and fitz_words > 0:
        return "PyMuPDF finds raw words but parser text blocks are zero. Inspect filtering or merge logic."
    if parser_text_blocks == 0 and plumber_available and plumber_words > 0 and fitz_words == 0:
        return "pdfplumber sees words but PyMuPDF does not. Inspect extract_text_with_pymupdf path."
    if parser_text_blocks == 0 and parser_tables > 0:
        return "Only table content detected. Text extraction path may be filtering too aggressively."
    return "No single obvious failure mode. Compare page-level stats and overlap filtering."


def run_diagnostics(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    preview_chars: int,
    max_chunk_preview: int,
) -> dict[str, Any]:
    page_tables = extract_tables_with_pdfplumber(str(pdf_path))
    page_text_blocks = extract_text_with_pymupdf(str(pdf_path), page_tables)

    page_table_counts = {str(page_num + 1): len(items) for page_num, items in page_tables.items()}
    page_text_counts = {str(page_num + 1): len(items) for page_num, items in page_text_blocks.items()}

    parser_result = parse_document_hybrid(str(pdf_path), save_parsed_text=False)
    parser_meta = parser_result.get("metadata", {})
    total_pages = int(parser_result.get("total_pages", 0) or 0)

    chunks = chunk_documents_optimized(
        [parser_result],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    text_chunks = sum(
        1 for chunk in chunks if chunk.get("metadata", {}).get("content_type") == "text"
    )
    table_chunks = sum(
        1 for chunk in chunks if chunk.get("metadata", {}).get("content_type") == "table"
    )

    fitz_stats = get_fitz_page_stats(pdf_path)
    plumber_stats = get_pdfplumber_page_stats(pdf_path)

    pages_with_no_text_blocks = [
        page for page in range(1, total_pages + 1) if page_text_counts.get(str(page), 0) == 0
    ]
    pages_with_no_raw_words = [
        p["page"] for p in fitz_stats.get("pages", []) if p.get("words", 0) == 0
    ]

    return {
        "parser": {
            "method": parser_result.get("parsing_method", ""),
            "total_pages": total_pages,
            "metadata": parser_meta,
            "content_chars": len(parser_result.get("content", "")),
        },
        "low_level_extraction": {
            "page_table_counts": page_table_counts,
            "page_text_block_counts": page_text_counts,
            "total_tables_detected": sum(page_table_counts.values()),
            "total_text_blocks_detected": sum(page_text_counts.values()),
            "pages_with_no_text_blocks": pages_with_no_text_blocks,
            "fitz": fitz_stats,
            "pdfplumber": plumber_stats,
            "pages_with_no_raw_words": pages_with_no_raw_words,
        },
        "chunking": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks),
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "previews": preview_chunks(chunks, preview_chars, max_chunk_preview),
        },
        "likely_issue": infer_likely_issue(parser_meta, fitz_stats, plumber_stats),
    }


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Empty Parsed Diagnostics")
    lines.append("")
    lines.append(f"- Project root: {report.get('root', '')}")
    lines.append(f"- Parsed output folder: {report.get('output_dir', '')}")
    lines.append(f"- Catalog folder: {report.get('catalog_dir', '')}")
    lines.append(f"- Empty parsed targets: {len(report.get('targets', []))}")
    lines.append("")

    for entry in report.get("targets", []):
        lines.append(f"## {entry.get('document_key', 'unknown')}")
        lines.append(f"- Parsed file: {entry.get('parsed_file', '')}")
        lines.append(f"- Source PDF: {entry.get('source_pdf', '')}")
        lines.append(f"- Source exists: {entry.get('source_exists', False)}")

        if entry.get("error"):
            lines.append(f"- Error: {entry['error']}")
            lines.append("")
            continue

        diagnostics = entry.get("diagnostics", {})
        parser = diagnostics.get("parser", {})
        parser_meta = parser.get("metadata", {})
        extraction = diagnostics.get("low_level_extraction", {})
        chunking = diagnostics.get("chunking", {})
        fitz_diag = extraction.get("fitz", {})

        lines.append(f"- Parser text blocks: {parser_meta.get('text_elements', 0)}")
        lines.append(f"- Parser tables: {parser_meta.get('table_elements', 0)}")
        lines.append(f"- Parser extracted characters: {parser_meta.get('characters_extracted', 0)}")
        lines.append(f"- Low-level text blocks (sum): {extraction.get('total_text_blocks_detected', 0)}")
        lines.append(f"- Low-level tables (sum): {extraction.get('total_tables_detected', 0)}")
        lines.append(f"- PyMuPDF total words: {fitz_diag.get('total_words', 0)}")
        lines.append(f"- PyMuPDF total images: {fitz_diag.get('total_images', 0)}")
        lines.append(f"- PyMuPDF total drawings: {fitz_diag.get('total_drawings', 0)}")
        lines.append(f"- PyMuPDF total fonts: {fitz_diag.get('total_fonts', 0)}")
        lines.append(f"- Chunks generated: {chunking.get('total_chunks', 0)}")
        lines.append(f"- Likely issue: {diagnostics.get('likely_issue', '')}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug PDFs whose parsed outputs show Text Blocks: 0 and test chunking behavior."
    )
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--output-dir", default="output", help="Parsed output folder.")
    parser.add_argument("--catalog-dir", default="data/catalog_docs", help="Source PDF folder.")
    parser.add_argument(
        "--report-json",
        default="output/empty_parsed_debug_report.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--report-md",
        default="output/empty_parsed_debug_report.md",
        help="Markdown report path.",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size for debug chunking.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for debug chunking.")
    parser.add_argument("--preview-chars", type=int, default=180, help="Chunk preview characters.")
    parser.add_argument(
        "--max-chunk-preview",
        type=int,
        default=5,
        help="Max chunk previews per document.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated doc keys to debug, for example mit_cs_program_requirements,mit_eecs_program_requirements",
    )
    return parser.parse_args()


def main() -> None:
    ensure_utf8_console()
    args = parse_args()

    root = resolve_path(Path.cwd(), args.root)
    output_dir = resolve_path(root, args.output_dir)
    catalog_dir = resolve_path(root, args.catalog_dir)

    empty_parsed_files = find_empty_parsed_files(output_dir) if output_dir.exists() else []

    if args.only.strip():
        allowed = {
            item.strip().replace(".pdf", "").replace("_parsed", "")
            for item in args.only.split(",")
            if item.strip()
        }
        empty_parsed_files = [
            p for p in empty_parsed_files if p.name.replace("_parsed.txt", "") in allowed
        ]

    report: dict[str, Any] = {
        "root": root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "catalog_dir": catalog_dir.as_posix(),
        "targets": [],
    }

    for parsed_file in empty_parsed_files:
        document_key = parsed_file.name.replace("_parsed.txt", "")
        source_pdf = catalog_dir / f"{document_key}.pdf"

        entry: dict[str, Any] = {
            "document_key": document_key,
            "parsed_file": rel(parsed_file, root),
            "parsed_header": read_parsed_header(parsed_file),
            "source_pdf": rel(source_pdf, root),
            "source_exists": source_pdf.exists(),
        }

        if not source_pdf.exists():
            entry["error"] = "Source PDF not found."
            report["targets"].append(entry)
            continue

        try:
            entry["diagnostics"] = run_diagnostics(
                source_pdf,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                preview_chars=args.preview_chars,
                max_chunk_preview=args.max_chunk_preview,
            )
        except Exception as exc:
            entry["error"] = str(exc)

        report["targets"].append(entry)

    json_path = resolve_path(root, args.report_json)
    md_path = resolve_path(root, args.report_md)

    write_json_report(report, json_path)
    write_markdown_report(report, md_path)

    print(f"Empty parsed files found: {len(report['targets'])}")
    print(f"JSON report: {json_path.as_posix()}")
    print(f"Markdown report: {md_path.as_posix()}")


if __name__ == "__main__":
    main()