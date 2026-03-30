from src.ingestion.graph_builder import (
	_extract_global_gpa_requirements,
	_extract_requirement_text,
	_parse_prereq_expression,
)


def test_extract_global_gpa_requirements_parses_letter_threshold_with_scale() -> None:
	text = (
		"An MIT undergraduate must achieve all of the following standards: "
		"Have a cumulative grade point average (GPA) of at least a C "
		"(3.0 on MITs 5.0 scale)."
	)
	requirements = _extract_global_gpa_requirements(
		text,
		chunk_id="academic-performance-grades.pdf_2_demo",
		document_name="academic-performance-grades.pdf",
	)

	assert len(requirements) == 1
	requirement = requirements[0]
	assert requirement["operator"] == ">="
	assert requirement["threshold"] == 3.0
	assert requirement["scale_max"] == 5.0
	assert requirement["audience"] == "undergraduate"


def test_extract_global_gpa_requirements_parses_exceeding_rule() -> None:
	text = (
		"An MIT graduate student must have a cumulative grade point average (GPA) "
		"exceeding 4.0 on MITs 5.0 scale."
	)
	requirements = _extract_global_gpa_requirements(
		text,
		chunk_id="academic-performance-grades.pdf_5_demo",
		document_name="academic-performance-grades.pdf",
	)

	assert len(requirements) == 1
	requirement = requirements[0]
	assert requirement["operator"] == ">"
	assert requirement["threshold"] == 4.0
	assert requirement["scale_max"] == 5.0
	assert requirement["audience"] == "graduate"


def test_extract_global_gpa_requirements_keeps_distinct_audiences() -> None:
	text = (
		"Undergraduate students must keep a GPA of at least 3.0 on 5.0 scale. "
		"Graduate students must keep a GPA of at least 3.0 on 5.0 scale."
	)
	requirements = _extract_global_gpa_requirements(
		text,
		chunk_id="mixed_policy_chunk",
		document_name="academic-performance-grades.pdf",
	)

	assert len(requirements) == 2
	audiences = {str(item.get("audience", "")) for item in requirements}
	assert audiences == {"undergraduate", "graduate"}


def test_extract_global_gpa_requirements_marks_financial_aid_scope() -> None:
	text = (
		"An MIT graduate student must achieve all of the following standards to qualify "
		"and maintain federal student financial assistance: Have a cumulative grade point "
		"average (GPA) exceeding 4.0 on MITs 5.0 scale."
	)
	requirements = _extract_global_gpa_requirements(
		text,
		chunk_id="academic-performance-grades.pdf_13_demo",
		document_name="academic-performance-grades.pdf",
	)

	assert len(requirements) == 1
	assert requirements[0]["scope"] == "financial_aid"


def test_extract_global_gpa_requirements_marks_financial_aid_scope_from_pace_signal() -> None:
	text = (
		"Graduate Academic Standards Have a cumulative grade point average (GPA) of at least a C "
		"(3.0 on MITs 5.0 scale). Pass 67% of cumulative units attempted (defined as pace)."
	)
	requirements = _extract_global_gpa_requirements(
		text,
		chunk_id="academic-performance-grades.pdf_7_demo",
		document_name="academic-performance-grades.pdf",
	)

	assert len(requirements) == 1
	assert requirements[0]["scope"] == "financial_aid"


def test_extract_requirement_text_supports_multiline_prereq() -> None:
	content = (
		"6.1234 Some Course Name\n"
		"Prereq: 6.100A or\n"
		"  6.100B with grade of B or better\n"
		"U (Fall)\n"
		"2-4-6 units"
	)

	prereq = _extract_requirement_text(content, "Prereq")

	assert prereq == "6.100A or 6.100B with grade of B or better"


def test_parse_prereq_expression_extracts_min_grade_from_common_phrases() -> None:
	raw = "6.100A with grade of C or better"
	edges = _parse_prereq_expression(raw, target="6.1010", chunk_id="demo_chunk")

	assert len(edges) == 1
	assert edges[0]["from"] == "6.100A"
	assert edges[0]["to"] == "6.1010"
	assert edges[0]["min_grade"] == "C"
