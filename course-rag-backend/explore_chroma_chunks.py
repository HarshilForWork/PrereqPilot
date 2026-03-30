"""Explore chunks stored in ChromaDB and explain chroma.sqlite3.

Examples:
  python explore_chroma_chunks.py
  python explore_chroma_chunks.py --limit 20 --doc 6_coursestext.pdf
  python explore_chroma_chunks.py --contains "Prereq:" --content-type text
  python explore_chroma_chunks.py --show-doc-stats --show-metadata-keys
  python explore_chroma_chunks.py --export-jsonl output/chroma_chunk_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

try:
	import chromadb
except ImportError:  # pragma: no cover - dependency is expected in runtime env
	chromadb = None


SQL_TABLE_PURPOSES = {
	"tenants": "Logical tenant boundary (usually default tenant).",
	"databases": "Database namespace records under tenants.",
	"collections": "Collection definitions (name, dimension, database).",
	"collection_metadata": "Collection-level metadata key/value pairs.",
	"segments": "Physical segment records (VECTOR and METADATA backends).",
	"segment_metadata": "Metadata associated with segments.",
	"embeddings": "Core chunk-to-vector row mapping (one row per stored chunk).",
	"embedding_metadata": "Flattened metadata key/value entries per embedding row.",
	"embedding_metadata_array": "Array-type metadata storage (if used).",
	"embeddings_queue": "Write-ahead queue for ingest operations.",
	"embeddings_queue_config": "Configuration for the embeddings queue.",
	"embedding_fulltext_search": "FTS virtual table for document text search.",
	"embedding_fulltext_search_data": "FTS backing store (SQLite internals).",
	"embedding_fulltext_search_idx": "FTS term index (SQLite internals).",
	"embedding_fulltext_search_docsize": "FTS doc size stats (SQLite internals).",
	"embedding_fulltext_search_content": "FTS content mirror for chunk text.",
	"embedding_fulltext_search_config": "FTS configuration table.",
	"max_seq_id": "Monotonic sequence tracker for updates.",
	"migrations": "Applied schema migration history.",
	"maintenance_log": "Maintenance actions log (if any).",
	"acquire_write": "Internal lock/control table for coordinated writes.",
}


def header(title: str) -> None:
	line = "=" * len(title)
	print(f"\n{title}\n{line}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Explore chunks in ChromaDB and explain chroma.sqlite3 internals."
	)
	parser.add_argument("--root", default=".", help="Project root path.")
	parser.add_argument(
		"--chroma-dir",
		default="data/chroma_db",
		help="Path to Chroma persistent directory (relative to --root).",
	)
	parser.add_argument(
		"--sqlite-file",
		default="data/chroma_db/chroma.sqlite3",
		help="Path to Chroma sqlite file (relative to --root).",
	)
	parser.add_argument(
		"--collection",
		default="course_catalog",
		help="Collection name to inspect.",
	)
	parser.add_argument("--limit", type=int, default=10, help="Number of chunks to show.")
	parser.add_argument("--offset", type=int, default=0, help="Offset for chunk listing.")
	parser.add_argument(
		"--preview-chars",
		type=int,
		default=260,
		help="Preview length for chunk text.",
	)
	parser.add_argument(
		"--doc",
		default="",
		help="Filter by metadata.document_name (exact match).",
	)
	parser.add_argument(
		"--content-type",
		default="",
		help="Filter by metadata.content_type (for example: text, table).",
	)
	parser.add_argument(
		"--contains",
		default="",
		help="Only show chunks whose text contains this substring (client-side filter).",
	)
	parser.add_argument(
		"--show-doc-stats",
		action="store_true",
		help="Print per-document and per-content_type counts from metadata.",
	)
	parser.add_argument(
		"--show-metadata-keys",
		action="store_true",
		help="Show metadata key frequency across the collection.",
	)
	parser.add_argument(
		"--export-jsonl",
		default="",
		help="Export displayed chunks to JSONL file.",
	)
	return parser.parse_args()


def normalize_preview(text: str, max_chars: int) -> str:
	norm = " ".join((text or "").split())
	if len(norm) <= max_chars:
		return norm
	return norm[:max_chars].rstrip() + "..."


def build_where_filter(args: argparse.Namespace) -> dict[str, Any] | None:
	filters: list[dict[str, Any]] = []
	if args.doc:
		filters.append({"document_name": args.doc})
	if args.content_type:
		filters.append({"content_type": args.content_type})

	if not filters:
		return None
	if len(filters) == 1:
		return filters[0]
	return {"$and": filters}


def open_collection(chroma_dir: Path, collection_name: str):
	if chromadb is None:
		raise RuntimeError(
			"chromadb is not installed in this environment. Install requirements first."
		)
	client = chromadb.PersistentClient(path=str(chroma_dir))
	return client.get_collection(collection_name)


def filtered_chunks(data: dict[str, Any], contains: str) -> list[dict[str, Any]]:
	ids = data.get("ids", []) or []
	docs = data.get("documents", []) or []
	metadatas = data.get("metadatas", []) or []

	needle = contains.lower().strip()
	rows: list[dict[str, Any]] = []
	for idx, chunk_id in enumerate(ids):
		doc_text = docs[idx] if idx < len(docs) else ""
		meta = metadatas[idx] if idx < len(metadatas) else {}
		meta = meta or {}
		if needle and needle not in (doc_text or "").lower():
			continue
		rows.append({"id": chunk_id, "document": doc_text or "", "metadata": meta})
	return rows


def print_chunk_rows(rows: list[dict[str, Any]], preview_chars: int) -> None:
	header("Chunk Samples")
	if not rows:
		print("No chunks matched your filters.")
		return

	for idx, row in enumerate(rows, start=1):
		meta = row.get("metadata", {}) or {}
		print(f"[{idx}] id={row['id']}")
		print(
			"    doc={doc} | chunk_index={chunk_index} | content_type={ctype} | chars={chars}".format(
				doc=meta.get("document_name", ""),
				chunk_index=meta.get("chunk_index", ""),
				ctype=meta.get("content_type", ""),
				chars=meta.get("char_count", len(row.get("document", ""))),
			)
		)
		print(f"    preview: {normalize_preview(row.get('document', ''), preview_chars)}")
		print()


def print_doc_stats(collection, where_filter: dict[str, Any] | None) -> None:
	header("Chunk Distribution")
	count = collection.count()
	if count == 0:
		print("Collection is empty.")
		return

	data = collection.get(
		where=where_filter,
		limit=count,
		offset=0,
		include=["metadatas"],
	)
	metadatas = data.get("metadatas", []) or []
	doc_counter = Counter((m or {}).get("document_name", "<missing>") for m in metadatas)
	ctype_counter = Counter((m or {}).get("content_type", "<missing>") for m in metadatas)

	print("Per document:")
	for doc, doc_count in sorted(doc_counter.items(), key=lambda item: (-item[1], item[0])):
		print(f"- {doc}: {doc_count}")

	print("\nPer content_type:")
	for ctype, ctype_count in sorted(ctype_counter.items(), key=lambda item: (-item[1], item[0])):
		print(f"- {ctype}: {ctype_count}")


def print_metadata_key_stats(collection, where_filter: dict[str, Any] | None) -> None:
	header("Metadata Key Frequency")
	count = collection.count()
	if count == 0:
		print("Collection is empty.")
		return

	data = collection.get(
		where=where_filter,
		limit=count,
		offset=0,
		include=["metadatas"],
	)
	metadatas = data.get("metadatas", []) or []
	key_counter = Counter()
	for meta in metadatas:
		for key in (meta or {}).keys():
			key_counter[key] += 1

	if not key_counter:
		print("No metadata keys found.")
		return

	for key, freq in sorted(key_counter.items(), key=lambda item: (-item[1], item[0])):
		print(f"- {key}: {freq}")


def export_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=True) + "\n")
	print(f"Exported {len(rows)} chunk row(s) to {output_path.as_posix()}")


def inspect_sqlite(sqlite_file: Path) -> None:
	header("What chroma.sqlite3 Is")
	print(
		"chroma.sqlite3 is Chroma's catalog and metadata database. "
		"It stores collection definitions, chunk text indexes, metadata, segment registry, "
		"migration history, and ingest queue state."
	)
	print(
		"Dense vector index files are stored separately in UUID-named segment folders under data/chroma_db/."
	)

	header("SQLite Table Guide")
	if not sqlite_file.exists():
		print(f"File not found: {sqlite_file.as_posix()}")
		return

	conn = sqlite3.connect(str(sqlite_file))
	cur = conn.cursor()
	try:
		tables = [
			row[0]
			for row in cur.execute(
				"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
			).fetchall()
		]

		print(f"File: {sqlite_file.as_posix()}")
		print(f"Tables found: {len(tables)}")
		for table in tables:
			count = cur.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
			purpose = SQL_TABLE_PURPOSES.get(table, "Chroma internal table.")
			print(f"- {table}: {count} rows")
			print(f"  purpose: {purpose}")

		if "collections" in tables:
			print("\nCollections:")
			for row in cur.execute(
				"SELECT id, name, dimension, database_id FROM collections"
			).fetchall():
				print(
					"- id={id}, name={name}, dim={dim}, database_id={dbid}".format(
						id=row[0], name=row[1], dim=row[2], dbid=row[3]
					)
				)

		if "segments" in tables:
			print("\nSegments:")
			for row in cur.execute(
				"SELECT id, type, scope, collection FROM segments"
			).fetchall():
				print(
					"- id={id}, type={typ}, scope={scope}, collection={coll}".format(
						id=row[0], typ=row[1], scope=row[2], coll=row[3]
					)
				)
	finally:
		conn.close()


def main() -> None:
	args = parse_args()
	root = Path(args.root).resolve()
	chroma_dir = (root / args.chroma_dir).resolve()
	sqlite_file = (root / args.sqlite_file).resolve()

	header("Chroma Chunk Explorer")
	print(f"Root: {root.as_posix()}")
	print(f"Chroma dir: {chroma_dir.as_posix()}")
	print(f"Collection: {args.collection}")

	where_filter = build_where_filter(args)
	if where_filter:
		print(f"Active metadata filter: {where_filter}")

	try:
		collection = open_collection(chroma_dir, args.collection)
	except Exception as exc:
		print(f"Failed to open collection: {exc}")
		inspect_sqlite(sqlite_file)
		return

	total_count = collection.count()
	print(f"Total chunks in collection: {total_count}")

	fetch_limit = max(1, int(args.limit))
	fetch_offset = max(0, int(args.offset))

	data = collection.get(
		where=where_filter,
		limit=fetch_limit,
		offset=fetch_offset,
		include=["documents", "metadatas"],
	)

	rows = filtered_chunks(data, contains=args.contains)
	print_chunk_rows(rows, preview_chars=max(40, int(args.preview_chars)))

	if args.export_jsonl:
		export_jsonl(rows, (root / args.export_jsonl).resolve())

	if args.show_doc_stats:
		print_doc_stats(collection, where_filter)

	if args.show_metadata_keys:
		print_metadata_key_stats(collection, where_filter)

	inspect_sqlite(sqlite_file)


if __name__ == "__main__":
	main()
