"""Explore project data files with plain-English guidance.

Usage:
  python explore_data_files.py
  python explore_data_files.py --root .
  python explore_data_files.py --save-parsed

This script helps you understand:
1. What files exist under data/ and related output folders.
2. What each important file is used for.
3. What is inside Chroma's sqlite catalog (tables and row counts).
4. A quick summary of the persisted graph file.
5. Optionally, save parser outputs from catalog PDFs for manual review.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


UUID_SEGMENT_DIR_RE = re.compile(r"data/chroma_db/[0-9a-f-]{36}$", re.IGNORECASE)

CHROMA_SEGMENT_FILE_GUIDE = {
	"header.bin": "HNSW index header (format and index settings).",
	"length.bin": "Byte-length/offset helper data used by the vector index.",
	"link_lists.bin": "HNSW neighbor link graph used for ANN traversal.",
	"data_level0.bin": "Vector payload storage for the base HNSW layer.",
	"index_metadata.pickle": "Python metadata blob for ID mappings and index internals.",
}


def human_size(num_bytes: int) -> str:
	units = ["B", "KB", "MB", "GB", "TB"]
	size = float(max(0, num_bytes))
	for unit in units:
		if size < 1024.0 or unit == units[-1]:
			if unit == "B":
				return f"{int(size)} {unit}"
			return f"{size:.2f} {unit}"
		size /= 1024.0
	return f"{num_bytes} B"


def print_header(title: str) -> None:
	bar = "=" * len(title)
	print(f"\n{title}\n{bar}")


def rel_path(path: Path, root: Path) -> str:
	try:
		return path.relative_to(root).as_posix()
	except ValueError:
		return path.as_posix()


def describe_entry(path: Path, root: Path) -> str:
	rel = rel_path(path, root)

	if path.is_dir():
		if rel == "data":
			return "Main persisted runtime data folder."
		if rel == "data/catalog_docs":
			return "Source PDFs used for ingestion."
		if rel == "data/chroma_db":
			return "Chroma persistent store root (sqlite + vector index segments)."
		if UUID_SEGMENT_DIR_RE.match(rel):
			return "One Chroma vector segment folder (HNSW index files)."
		if rel == "output":
			return "Saved parser output previews (if enabled)."
		if rel == "request_logs":
			return "Per-request trace snapshots for API calls."
		if rel == "logs":
			return "Application log files."
		return "Directory"

	name = path.name
	if rel == "data/chroma_db/chroma.sqlite3":
		return "Chroma sqlite catalog (collections, segments, metadata, document refs)."
	if rel == "data/graph_store.json":
		return "Persisted prerequisite graph (nodes/edges)."
	if rel.startswith("data/catalog_docs/") and path.suffix.lower() == ".pdf":
		return "Input PDF used by parser/chunker during ingestion."
	if rel.startswith("output/") and path.suffix.lower() == ".txt":
		return "Saved parsed text output from hybrid parser."
	if rel.startswith("request_logs/") and path.suffix.lower() == ".json":
		return "Captured request/response context for debugging."

	parent_rel = rel_path(path.parent, root)
	if UUID_SEGMENT_DIR_RE.match(parent_rel):
		return CHROMA_SEGMENT_FILE_GUIDE.get(name, "Chroma segment artifact file.")

	return "File"


def list_relevant_entries(root: Path) -> list[Path]:
	paths: list[Path] = []
	for folder in ("data", "output", "logs", "request_logs"):
		base = root / folder
		if not base.exists():
			continue
		for path in sorted(base.rglob("*")):
			paths.append(path)
	return paths


def print_filesystem_guide(root: Path) -> None:
	print_header("Filesystem Guide")
	entries = list_relevant_entries(root)
	if not entries:
		print("No data/output/log files were found yet.")
		return

	for path in entries:
		rel = rel_path(path, root)
		role = describe_entry(path, root)
		if path.is_dir():
			print(f"[DIR]  {rel}")
			print(f"       -> {role}")
		else:
			stat = path.stat()
			mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
			print(f"[FILE] {rel} | {human_size(stat.st_size)} | modified {mtime}")
			print(f"       -> {role}")


def sql_columns(cur: sqlite3.Cursor, table: str) -> list[str]:
	quoted = table.replace("\"", "\"\"")
	rows = cur.execute(f'PRAGMA table_info("{quoted}")').fetchall()
	return [row[1] for row in rows]


def safe_count(cur: sqlite3.Cursor, table: str) -> int | None:
	quoted = table.replace("\"", "\"\"")
	try:
		return int(cur.execute(f'SELECT COUNT(*) FROM "{quoted}"').fetchone()[0])
	except Exception:
		return None


def inspect_chroma_sqlite(root: Path, sample_rows: int) -> None:
	print_header("Chroma SQLite Inspection")
	db_path = root / "data" / "chroma_db" / "chroma.sqlite3"
	if not db_path.exists():
		print("No chroma sqlite file found at data/chroma_db/chroma.sqlite3")
		return

	print(f"Database: {rel_path(db_path, root)}")

	conn = sqlite3.connect(str(db_path))
	cur = conn.cursor()
	try:
		tables = [
			row[0]
			for row in cur.execute(
				"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
			).fetchall()
		]
		print(f"Tables ({len(tables)}): {', '.join(tables)}")

		print("\nTable row counts:")
		for table in tables:
			count = safe_count(cur, table)
			if count is None:
				print(f"- {table}: unable to count")
			else:
				print(f"- {table}: {count}")

		if "collections" in tables:
			cols = sql_columns(cur, "collections")
			wanted = [c for c in ("id", "name", "dimension", "database_id") if c in cols]
			if wanted:
				rows = cur.execute(
					f"SELECT {', '.join(wanted)} FROM collections LIMIT ?",
					(sample_rows,),
				).fetchall()
				print("\nCollections sample:")
				for row in rows:
					print(f"- {dict(zip(wanted, row))}")

		if "segments" in tables:
			cols = sql_columns(cur, "segments")
			wanted = [c for c in ("id", "type", "scope", "collection") if c in cols]
			if wanted:
				rows = cur.execute(
					f"SELECT {', '.join(wanted)} FROM segments LIMIT ?",
					(sample_rows,),
				).fetchall()
				print("\nSegments sample:")
				for row in rows:
					print(f"- {dict(zip(wanted, row))}")
	finally:
		conn.close()


def inspect_graph_store(root: Path) -> None:
	print_header("Graph Store Inspection")
	graph_path = root / "data" / "graph_store.json"
	if not graph_path.exists():
		print("No graph store found at data/graph_store.json")
		return

	try:
		with graph_path.open("r", encoding="utf-8") as f:
			payload = json.load(f)
		nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
		edges = []
		if isinstance(payload, dict):
			edges = payload.get("links", payload.get("edges", []))

		print(f"Path: {rel_path(graph_path, root)}")
		print(f"Nodes: {len(nodes)}")
		print(f"Edges: {len(edges)}")

		if nodes:
			sample = nodes[:5]
			print("Sample node entries (up to 5):")
			for node in sample:
				if isinstance(node, dict):
					label = node.get("id", "<no id>")
				else:
					label = str(node)
				print(f"- {label}")
	except Exception as exc:
		print(f"Failed to read graph store: {exc}")


def save_parser_outputs(root: Path) -> None:
	print_header("Saving Parser Outputs")

	try:
		from src.processing.parser import parse_document_hybrid
	except Exception as exc:
		print("Could not import parser from src.processing.parser")
		print(f"Import error: {exc}")
		print("Run this script from the project root where src/ is present.")
		return

	catalog_dir = root / "data" / "catalog_docs"
	if not catalog_dir.exists():
		print("Catalog directory not found: data/catalog_docs")
		return

	pdfs = sorted(catalog_dir.glob("*.pdf"))
	if not pdfs:
		print("No PDF files found in data/catalog_docs")
		return

	print(f"Found {len(pdfs)} PDF(s). Parsing and saving outputs...")
	for pdf in pdfs:
		print(f"- Parsing {pdf.name}")
		parse_document_hybrid(str(pdf), save_parsed_text=True)

	print("Saved parser outputs to output/*.txt")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Explore project data files and explain what each file does."
	)
	parser.add_argument(
		"--root",
		default=".",
		help="Project root path (defaults to current directory).",
	)
	parser.add_argument(
		"--sample-rows",
		type=int,
		default=5,
		help="How many sample rows to print from sqlite tables.",
	)
	parser.add_argument(
		"--save-parsed",
		action="store_true",
		help="Parse PDFs from data/catalog_docs and save parser outputs to output/*.txt.",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	root = Path(args.root).resolve()

	print_header("Data Explorer")
	print(f"Project root: {root}")
	print("This report explains file purpose and basic contents.")

	print_filesystem_guide(root)
	inspect_chroma_sqlite(root, max(1, args.sample_rows))
	inspect_graph_store(root)

	if args.save_parsed:
		save_parser_outputs(root)

	print_header("Quick Tips")
	print("- Run with --save-parsed to generate human-readable parser outputs in output/*.txt")
	print("- Compare output/*_parsed.txt with source PDFs to validate extraction quality.")
	print("- If Chroma row counts grow but retrieval quality drops, inspect chunk metadata fields.")


if __name__ == "__main__":
	main()
