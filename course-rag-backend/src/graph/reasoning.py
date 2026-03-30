"""
Graph reasoning layer — all prerequisite logic lives here.
Operates on the singleton DiGraph from graph.store.
"""
import re
import networkx as nx
from src.graph.store import get_graph
from src.core.logger import get_logger

log = get_logger(__name__)

_FINANCIAL_AID_SCOPE_RE = re.compile(
    r"""
    \b(?:
        federal\s*student\s*(?:financial|nancial)\s*assistance|
        (?:financial|nancial)\s*aid|
        federal\s*aid|
        federal\s*regulations|
        satisfactory\s*academic\s*progress|
        (?:financial|nancial)\s*aid\s*(?:warning|probation)|
        67\s*%|
        150\s*%|
        \bpace\b
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(code: str) -> str:
    """Uppercase and strip whitespace so lookups are consistent."""
    return code.strip().upper()


def _predecessors_of(G: nx.DiGraph, course: str) -> list[tuple[str, dict]]:
    """Return (prereq_code, edge_attrs) for all direct prerequisites."""
    return [(u, G[u][course]) for u in G.predecessors(course)]


def _grade_ok(grade_received: str, min_grade: str) -> bool:
    """
    Simple letter-grade comparison.
    Order: A > B > C > D > F
    Returns True when grade_received meets or exceeds min_grade.
    """
    order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1, "": 0}
    received = grade_received.strip().upper()[:1]
    minimum  = min_grade.strip().upper()[:1]
    return order.get(received, 0) >= order.get(minimum, 0)


def _coerce_gpa_value(value: object) -> float | None:
    """Convert numeric/string GPA to float when possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _format_gpa(value: float) -> str:
    """Format GPA values consistently for messages."""
    return f"{value:.1f}"


def _normalize_student_level(value: object) -> str:
    """Normalize audience level labels used by policy rules."""
    if value is None:
        return ""
    lowered = str(value).strip().lower()
    if lowered.startswith("under"):
        return "undergraduate"
    if lowered.startswith("grad"):
        return "graduate"
    return ""


def _normalize_policy_scope(value: object) -> str:
    """Normalize policy scope labels used to decide applicability."""
    if value is None:
        return ""
    lowered = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if lowered in {"financial_aid", "aid", "federal_aid"}:
        return "financial_aid"
    if lowered in {"general", "enrollment", "academic", "all"}:
        return "general"
    return ""


def _infer_policy_scope(rule: dict) -> str:
    """
    Infer policy scope for backward compatibility with previously ingested data.

    Old graph metadata may not include a scope field, so we infer from evidence
    text and treat policies as general unless they explicitly mention aid.
    """
    explicit = _normalize_policy_scope(rule.get("scope"))
    if explicit:
        return explicit

    evidence_blob = " ".join(
        [
            str(rule.get("evidence_text", "") or ""),
            str(rule.get("document_name", "") or ""),
        ]
    )
    if _FINANCIAL_AID_SCOPE_RE.search(evidence_blob):
        return "financial_aid"

    return "general"


def _filter_policy_requirements(
    policy_requirements: list[dict],
    include_financial_aid_policies: bool,
) -> list[dict]:
    """Return only policies relevant to the current eligibility context."""
    filtered: list[dict] = []

    for rule in policy_requirements:
        scope = _infer_policy_scope(rule)
        if scope == "financial_aid" and not include_financial_aid_policies:
            continue

        normalized_rule = dict(rule)
        normalized_rule["scope"] = scope
        filtered.append(normalized_rule)

    return filtered


def _evaluate_global_gpa_requirements(
    policy_requirements: list[dict],
    student_gpa: float | None,
    student_level: str,
) -> tuple[list[str], bool, bool, bool]:
    """
    Returns (issues, missing_gpa_info, has_hard_violation, missing_level_info).
    """
    issues: list[str] = []
    missing_gpa_info = False
    has_hard_violation = False
    missing_level_info = False
    seen_rules: set[tuple[str, float, float | None]] = set()
    missing_level_audiences: set[str] = set()

    for rule in policy_requirements:
        if str(rule.get("type", "")).lower() != "gpa":
            continue

        audience = _normalize_student_level(rule.get("audience", "all"))
        if not audience:
            audience = "all"

        if audience != "all" and not student_level:
            missing_level_audiences.add(audience)
            continue

        if audience != "all" and audience != student_level:
            continue

        threshold = _coerce_gpa_value(rule.get("threshold"))
        if threshold is None:
            continue

        operator = str(rule.get("operator", ">=") or ">=")
        scale_max = _coerce_gpa_value(rule.get("scale_max"))
        dedupe_key = (operator, threshold, scale_max)
        if dedupe_key in seen_rules:
            continue
        seen_rules.add(dedupe_key)

        rule_text = f"cumulative GPA {operator} {_format_gpa(threshold)}"
        if scale_max is not None:
            rule_text += f" on {_format_gpa(scale_max)} scale"

        if student_gpa is None:
            issues.append(f"Requires {rule_text} - GPA not provided")
            missing_gpa_info = True
            continue

        if operator == ">":
            ok = student_gpa > threshold
        else:
            ok = student_gpa >= threshold

        if not ok:
            issues.append(f"Requires {rule_text}, got {_format_gpa(student_gpa)}")
            has_hard_violation = True

    if missing_level_audiences:
        ordered = ", ".join(sorted(missing_level_audiences))
        issues.append(
            f"GPA policy depends on student level ({ordered}) - student level not provided"
        )
        missing_level_info = True

    return issues, missing_gpa_info, has_hard_violation, missing_level_info


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_eligibility(
    course_code: str,
    completed: list[str],
    grades: dict[str, str] | None = None,
    student_gpa: float | str | None = None,
    student_level: str | None = None,
    enforce_gpa_policies: bool = True,
    include_financial_aid_policies: bool = False,
) -> dict:
    """
    Check whether a student is eligible to enrol in course_code.

    By default, GPA policies classified as financial-aid rules are excluded so
    non-aid prerequisite checks are not over-constrained.

    Returns
    -------
    {
        "eligible":           bool,
        "decision":           "Eligible" | "Not eligible" | "Need more info",
        "missing_prereqs":    list[str],
        "missing_coreqs":     list[str],
        "grade_issues":       list[str],
        "gpa_issues":         list[str],
        "policy_requirements": list[dict],
        "prereq_path":        list[str],   # full ordered chain
        "either_or_options":  list[list[str]]
    }
    """
    G = get_graph()
    course = _normalize(course_code)
    normalized_grades = grades or {}
    gpa_value = _coerce_gpa_value(student_gpa)
    level_value = _normalize_student_level(student_level)
    completed_set = {_normalize(c) for c in completed}
    raw_policy_requirements = (
        list(G.graph.get("global_policy_requirements", [])) if enforce_gpa_policies else []
    )
    policy_requirements = _filter_policy_requirements(
        raw_policy_requirements,
        include_financial_aid_policies,
    )

    result = {
        "eligible": False,
        "decision": "Need more info",
        "missing_prereqs": [],
        "missing_coreqs": [],
        "grade_issues": [],
        "gpa_issues": [],
        "prereq_path": [],
        "either_or_options": [],
        "policy_requirements": policy_requirements,
    }

    if course not in G:
        log.warning(f"Course {course} not found in graph.")
        result["decision"] = "Need more info"
        return result

    missing_prereqs: list[str] = []
    missing_coreqs: list[str] = []
    grade_issues: list[str] = []
    gpa_issues, missing_gpa_info, gpa_hard_violation, missing_level_info = _evaluate_global_gpa_requirements(
        policy_requirements,
        gpa_value,
        level_value,
    )
    either_or_options: list[list[str]] = []

    # Group predecessors by requirement_type
    required_edges: list[tuple[str, dict]] = []
    either_or_groups: dict[str, dict[str, list[str]]] = {}
    coreq_edges: list[tuple[str, dict]] = []

    for prereq, attrs in _predecessors_of(G, course):
        req_type = attrs.get("requirement_type", "required")
        group    = attrs.get("either_or_group", None)

        if req_type == "corequisite":
            coreq_edges.append((prereq, attrs))
        elif req_type == "either_or" and group:
            branch_ids = attrs.get("either_or_branches", []) or []
            if not branch_ids:
                single_branch = attrs.get("either_or_branch")
                if single_branch:
                    branch_ids = [single_branch]
            if not branch_ids:
                # Backward compatibility for previously-ingested edges with no branch metadata.
                branch_ids = [prereq]

            group_branches = either_or_groups.setdefault(group, {})
            for branch_id in branch_ids:
                branch_key = str(branch_id)
                group_branches.setdefault(branch_key, []).append(prereq)
        else:
            required_edges.append((prereq, attrs))

    # ── Check hard required prereqs ──────────────────────────────────────────
    for prereq, attrs in required_edges:
        min_grade = attrs.get("min_grade", "")
        if prereq not in completed_set:
            missing_prereqs.append(prereq)
        elif min_grade:
            received = normalized_grades.get(prereq, "")
            if received and not _grade_ok(received, min_grade):
                grade_issues.append(
                    f"{prereq} requires grade {min_grade}, got {received}"
                )
            elif not received:
                # Grade unknown — flag but don't block
                grade_issues.append(
                    f"{prereq} requires grade {min_grade} — grade not provided"
                )

    has_hard_missing_prereqs = bool(missing_prereqs)

    # ── Check either/or groups ───────────────────────────────────────────────
    for group_key, branch_map in either_or_groups.items():
        if has_hard_missing_prereqs:
            # Defer alternative-group guidance until hard required prerequisites are met.
            continue

        branch_options: list[list[str]] = []
        for courses in branch_map.values():
            unique_courses: list[str] = []
            for code in courses:
                if code not in unique_courses:
                    unique_courses.append(code)
            if unique_courses:
                branch_options.append(unique_courses)
        if not branch_options:
            continue

        common_courses = set(branch_options[0])
        for branch in branch_options[1:]:
            common_courses &= set(branch)
        ordered_common = [code for code in branch_options[0] if code in common_courses]

        for code in ordered_common:
            if code not in completed_set and code not in missing_prereqs:
                missing_prereqs.append(code)

        common_satisfied = all(code in completed_set for code in ordered_common)
        if not common_satisfied:
            # Shared prerequisites are still missing; alternatives are not actionable yet.
            continue

        reduced_options: list[list[str]] = []
        for branch in branch_options:
            reduced = [code for code in branch if code not in common_courses]
            if reduced and reduced not in reduced_options:
                reduced_options.append(reduced)

        # No residual options means the group effectively reduced to shared required courses.
        if not reduced_options:
            continue

        satisfied = any(all(course in completed_set for course in branch) for branch in reduced_options)
        if not satisfied:
            option_labels: list[str] = []
            for branch in reduced_options:
                if len(branch) == 1:
                    option_labels.append(branch[0])
                else:
                    option_labels.append(" and ".join(branch))
            if option_labels:
                either_or_options.append(option_labels)

    # ── Check coreqs ─────────────────────────────────────────────────────────
    for coreq, attrs in coreq_edges:
        if coreq not in completed_set:
            # Coreqs can be taken concurrently — flag but handle separately
            missing_coreqs.append(coreq)

    # ── Build prereq path ────────────────────────────────────────────────────
    try:
        prereq_path = get_full_prereq_chain(course)
    except Exception:
        prereq_path = []

    result["prereq_path"]       = prereq_path
    result["missing_prereqs"]   = missing_prereqs
    result["missing_coreqs"]    = missing_coreqs
    result["grade_issues"]      = grade_issues
    result["gpa_issues"]        = gpa_issues
    result["either_or_options"] = either_or_options

    # ── Final decision ───────────────────────────────────────────────────────
    has_missing_either_or = bool(either_or_options)

    if not missing_prereqs and not has_missing_either_or and not grade_issues and not gpa_issues:
        result["eligible"] = True
        result["decision"] = "Eligible"
    elif (
        (missing_gpa_info or missing_level_info)
        and not missing_prereqs
        and not has_missing_either_or
        and not grade_issues
        and not gpa_hard_violation
    ):
        result["eligible"] = False
        result["decision"] = "Need more info"
    elif missing_prereqs or has_missing_either_or:
        result["eligible"] = False
        result["decision"] = "Not eligible"
    else:
        # Grade and/or GPA policy issues remain
        result["eligible"] = False
        result["decision"] = "Not eligible"

    return result


def get_full_prereq_chain(course_code: str) -> list[str]:
    """
    Return a topologically sorted list of ALL courses that must be
    completed before enrolling in course_code (transitive closure).
    """
    G = get_graph()
    course = _normalize(course_code)

    if course not in G:
        return []

    # Collect all ancestors
    ancestors = nx.ancestors(G, course)
    if not ancestors:
        return []

    # Build subgraph of just ancestors + target, then topo sort
    sub = G.subgraph(ancestors | {course}).copy()
    self_loops = list(nx.selfloop_edges(sub))
    if self_loops:
        sub.remove_edges_from(self_loops)
        log.warning(
            f"Ignored {len(self_loops)} self-loop edge(s) while computing prereq chain for {course}"
        )
    try:
        ordered = list(nx.topological_sort(sub))
        # Remove the target course itself from the chain
        return [n for n in ordered if n != course]
    except nx.NetworkXUnfeasible:
        log.warning(f"Cycle detected in graph while computing prereq chain for {course}")
        return sorted(ancestors)


def get_available_next_courses(completed: list[str]) -> list[str]:
    """
    Return all courses a student can now enrol in, given their
    completed course list (ignoring grade requirements for simplicity).
    """
    G = get_graph()
    completed_set = {_normalize(c) for c in completed}
    available: list[str] = []

    for node in G.nodes():
        if node in completed_set:
            continue
        preds = list(G.predecessors(node))
        if not preds:
            # No prereqs — always available
            available.append(node)
            continue
        # Check if all required prereqs are met (simple version, ignores grades)
        result = check_eligibility(node, completed, {}, enforce_gpa_policies=False)
        if result["eligible"]:
            available.append(node)

    return sorted(available)


def find_path_to_course(target_course: str, completed: list[str]) -> dict:
    """
    Build a DAG-ready payload for the frontend path visualiser.

    Returns
    -------
    {
        "nodes":             [{"id", "label", "credits", "completed": bool}],
        "edges":             [{"from", "to", "type", "min_grade"}],
        "paths":             list[list[str]],
        "shortest_path":     list[str],
        "student_completed": list[str]
    }
    """
    G = get_graph()
    target  = _normalize(target_course)
    comp_set = {_normalize(c) for c in completed}

    if target not in G:
        return {
            "nodes": [], "edges": [], "paths": [],
            "shortest_path": [], "student_completed": list(comp_set),
        }

    ancestors = nx.ancestors(G, target) | {target}
    sub = G.subgraph(ancestors)

    # Build nodes
    nodes = []
    for n in sub.nodes():
        attrs = G.nodes[n]
        nodes.append({
            "id":        n,
            "label":     attrs.get("course_name", n),
            "credits":   attrs.get("credits", 0),
            "completed": n in comp_set,
        })

    # Build edges
    edges = []
    for u, v, attrs in sub.edges(data=True):
        edges.append({
            "from":      u,
            "to":        v,
            "type":      attrs.get("requirement_type", "required"),
            "min_grade": attrs.get("min_grade", ""),
        })

    # Find all simple paths from any root (no prereqs) to target
    roots = [n for n in sub.nodes() if sub.in_degree(n) == 0]
    all_paths: list[list[str]] = []
    for root in roots:
        try:
            for path in nx.all_simple_paths(G, root, target):
                all_paths.append(path)
        except nx.NetworkXNoPath:
            pass

    # Shortest path: fewest hops from any completed course or root
    shortest: list[str] = []
    if all_paths:
        shortest = min(all_paths, key=len)

    return {
        "nodes":             nodes,
        "edges":             edges,
        "paths":             all_paths,
        "shortest_path":     shortest,
        "student_completed": list(comp_set),
    }