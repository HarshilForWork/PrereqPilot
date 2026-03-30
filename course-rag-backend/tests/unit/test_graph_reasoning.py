from src.graph.reasoning import check_eligibility, find_path_to_course, get_full_prereq_chain
from src.graph.store import get_graph


def _build_sample_graph() -> None:
	graph = get_graph()
	graph.clear()
	graph.add_node("CS101", course_name="Intro CS", credits=3, description_chunk_id="c1", document_name="catalog.pdf")
	graph.add_node("CS201", course_name="Data Structures", credits=3, description_chunk_id="c2", document_name="catalog.pdf")
	graph.add_node("CS301", course_name="Algorithms", credits=3, description_chunk_id="c3", document_name="catalog.pdf")
	graph.add_edge("CS101", "CS201", requirement_type="required", min_grade="C", source_chunk_id="c4")
	graph.add_edge("CS201", "CS301", requirement_type="required", min_grade="B", source_chunk_id="c5")


def test_check_eligibility_reports_missing_prereqs() -> None:
	_build_sample_graph()
	result = check_eligibility("CS301", completed=["CS101"], grades={"CS101": "B"})
	assert result["eligible"] is False
	assert "CS201" in result["missing_prereqs"]
	assert result["decision"] == "Not eligible"


def test_get_full_prereq_chain_returns_ordered_ancestors() -> None:
	_build_sample_graph()
	chain = get_full_prereq_chain("CS301")
	assert chain == ["CS101", "CS201"]


def test_find_path_to_course_returns_dag_payload() -> None:
	_build_sample_graph()
	payload = find_path_to_course("CS301", completed=["CS101"])
	assert payload["shortest_path"] == ["CS101", "CS201", "CS301"]
	assert len(payload["nodes"]) == 3
	assert len(payload["edges"]) == 2


def test_check_eligibility_requires_gpa_when_policy_exists() -> None:
	_build_sample_graph()
	graph = get_graph()
	graph.graph["global_policy_requirements"] = [
		{
			"type": "gpa",
			"operator": ">=",
			"threshold": 3.0,
			"scale_max": 5.0,
			"source_chunk_id": "policy_chunk_1",
			"document_name": "academic-performance-grades.pdf",
		}
	]
	result = check_eligibility(
		"CS301",
		completed=["CS101", "CS201"],
		grades={"CS101": "A", "CS201": "B"},
	)
	assert result["eligible"] is False
	assert result["decision"] == "Need more info"
	assert any("GPA not provided" in issue for issue in result["gpa_issues"])


def test_check_eligibility_blocks_when_gpa_is_below_policy() -> None:
	_build_sample_graph()
	graph = get_graph()
	graph.graph["global_policy_requirements"] = [
		{
			"type": "gpa",
			"operator": ">",
			"threshold": 4.0,
			"scale_max": 5.0,
			"source_chunk_id": "policy_chunk_2",
			"document_name": "academic-performance-grades.pdf",
		}
	]
	result = check_eligibility(
		"CS301",
		completed=["CS101", "CS201"],
		grades={"CS101": "A", "CS201": "B"},
		student_gpa=3.5,
	)
	assert result["eligible"] is False
	assert result["decision"] == "Not eligible"
	assert any("got 3.5" in issue for issue in result["gpa_issues"])


def test_check_eligibility_requests_student_level_for_audience_specific_policy() -> None:
	_build_sample_graph()
	graph = get_graph()
	graph.graph["global_policy_requirements"] = [
		{
			"type": "gpa",
			"operator": ">",
			"threshold": 4.0,
			"scale_max": 5.0,
			"audience": "graduate",
			"source_chunk_id": "policy_chunk_3",
			"document_name": "academic-performance-grades.pdf",
		}
	]
	result = check_eligibility(
		"CS301",
		completed=["CS101", "CS201"],
		grades={"CS101": "A", "CS201": "B"},
		student_gpa=3.9,
	)
	assert result["eligible"] is False
	assert result["decision"] == "Need more info"
	assert any("student level not provided" in issue for issue in result["gpa_issues"])


def test_check_eligibility_applies_audience_specific_policy_when_level_is_provided() -> None:
	_build_sample_graph()
	graph = get_graph()
	graph.graph["global_policy_requirements"] = [
		{
			"type": "gpa",
			"operator": ">",
			"threshold": 4.0,
			"scale_max": 5.0,
			"audience": "graduate",
			"source_chunk_id": "policy_chunk_4",
			"document_name": "academic-performance-grades.pdf",
		}
	]
	result = check_eligibility(
		"CS301",
		completed=["CS101", "CS201"],
		grades={"CS101": "A", "CS201": "B"},
		student_gpa=3.9,
		student_level="graduate",
	)
	assert result["eligible"] is False
	assert result["decision"] == "Not eligible"
	assert any("got 3.9" in issue for issue in result["gpa_issues"])


def test_check_eligibility_ignores_financial_aid_policy_by_default() -> None:
	_build_sample_graph()
	graph = get_graph()
	graph.graph["global_policy_requirements"] = [
		{
			"type": "gpa",
			"operator": ">=",
			"threshold": 3.0,
			"scale_max": 5.0,
			"source_chunk_id": "policy_chunk_5",
			"document_name": "academic-performance-grades.pdf",
			"evidence_text": "Must maintain federal student nancial assistance with cumulative GPA of at least 3.0.",
		}
	]
	result = check_eligibility(
		"CS301",
		completed=["CS101", "CS201"],
		grades={"CS101": "A", "CS201": "B"},
	)
	assert result["eligible"] is True
	assert result["decision"] == "Eligible"
	assert result["gpa_issues"] == []


def test_check_eligibility_applies_financial_aid_policy_when_opted_in() -> None:
	_build_sample_graph()
	graph = get_graph()
	graph.graph["global_policy_requirements"] = [
		{
			"type": "gpa",
			"operator": ">=",
			"threshold": 3.0,
			"scale_max": 5.0,
			"source_chunk_id": "policy_chunk_6",
			"document_name": "academic-performance-grades.pdf",
			"evidence_text": "Must maintain federal student nancial assistance with cumulative GPA of at least 3.0.",
		}
	]
	result = check_eligibility(
		"CS301",
		completed=["CS101", "CS201"],
		grades={"CS101": "A", "CS201": "B"},
		include_financial_aid_policies=True,
	)
	assert result["eligible"] is False
	assert result["decision"] == "Need more info"
	assert any("GPA not provided" in issue for issue in result["gpa_issues"])


def test_check_eligibility_infers_financial_aid_from_pace_signal() -> None:
	_build_sample_graph()
	graph = get_graph()
	graph.graph["global_policy_requirements"] = [
		{
			"type": "gpa",
			"operator": ">=",
			"threshold": 3.0,
			"scale_max": 5.0,
			"source_chunk_id": "policy_chunk_7",
			"document_name": "academic-performance-grades.pdf",
			"evidence_text": "Have a cumulative GPA of at least 3.0 on MITs 5.0 scale. Pass 67% of cumulative units attempted (pace).",
		}
	]

	default_result = check_eligibility(
		"CS301",
		completed=["CS101", "CS201"],
		grades={"CS101": "A", "CS201": "B"},
	)
	assert default_result["eligible"] is True
	assert default_result["gpa_issues"] == []

	aid_result = check_eligibility(
		"CS301",
		completed=["CS101", "CS201"],
		grades={"CS101": "A", "CS201": "B"},
		include_financial_aid_policies=True,
	)
	assert aid_result["eligible"] is False
	assert aid_result["decision"] == "Need more info"
	assert any("GPA not provided" in issue for issue in aid_result["gpa_issues"])
