from src.ingestion.enricher import enrich_chunk_metadata


def test_enrich_chunk_metadata_returns_new_dict_and_scalar_metadata() -> None:
	raw_chunk = {
		"chunk_id": "doc_0_abcd1234",
		"document_name": "catalog.pdf",
		"content": "Prerequisite: CS101 or CS102. Minimum grade of B. 3 credit hours.",
		"chunk_index": 0,
		"char_count": 75,
		"metadata": {
			"chunking_method": "optimized_text",
			"content_type": "text",
			"source_document": "catalog.pdf",
			"chunk_size_config": 800,
			"overlap_config": 150,
		},
	}
	parser_result = {
		"document_name": "catalog.pdf",
		"content": "Academic Catalog 2024-2025",
		"metadata": {"institution": "Example University"},
	}

	enriched = enrich_chunk_metadata(raw_chunk, parser_result)

	assert enriched is not raw_chunk
	assert enriched["chunk_id"] == "doc_0_abcd1234"
	assert enriched["metadata"]["has_prerequisites"] is True
	assert enriched["metadata"]["has_grade_requirements"] is True
	assert enriched["metadata"]["has_credit_info"] is True
	assert enriched["metadata"]["course_codes_mentioned"] in {"CS101,CS102", "CS102,CS101"}
	assert enriched["metadata"]["catalog_year"] == "2024-2025"

	for value in enriched["metadata"].values():
		assert isinstance(value, (str, int, float, bool))
