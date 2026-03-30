"""
graph_builder.py
────────────────
Parses prerequisite/corequisite rules from enriched chunks and
populates the NetworkX DiGraph singleton in src/graph/store.py.

MIT catalog prereq text patterns handled:
  "Prereq: None"
  "Prereq: 6.100A"
  "Prereq: 6.1010 and 6.1200[J]"
  "Prereq: 6.1000 or (6.100A and (6.100B or 16.C20[J]))"
  "Prereq: Permission of instructor"
  "Coreq: 6.1903 or 6.1904"
  "Prereq: 6.1020, 6.1210, and 6.1910"
"""
import re
import uuid
from src.graph.store import get_graph, save
from src.core.logger import get_logger
from src.core.constants import COURSE_CODE_PAT, is_valid_course_code, normalize_course_code

log = get_logger(__name__)

_GPA_SCALE_RE = re.compile(
    r"(\d(?:\.\d+)?)\s*on\s*(?:MIT'?s\s*)?(\d(?:\.\d+)?)\s*scale",
    re.IGNORECASE,
)
_GPA_TRIGGER_RE = re.compile(r"\b(?:gpa|grade point average)\b", re.IGNORECASE)
_GPA_COMPARATOR_RE = re.compile(
    r"\b(?:at least|minimum|not less than|greater than or equal to|exceeding|greater than|above)\b|>=|>",
    re.IGNORECASE,
)
_GPA_THRESHOLD_RE = re.compile(
    r"(?:at least|minimum|not less than|greater than or equal to|>=|exceeding|greater than|above)\s*(?:a\s+)?([A-D]|\d(?:\.\d+)?)",
    re.IGNORECASE,
)
_LETTER_GPA_MAP = {
    "A": 5.0,
    "B": 4.0,
    "C": 3.0,
    "D": 2.0,
}

_MIN_GRADE_HINT_RE = re.compile(
    r"""
    grade\s+of\s+([A-F][+-]?)|
    minimum\s+grade(?:\s+of)?\s+([A-F][+-]?)|
    ([A-F][+-]?)\s+or\s+better|
    at\s+least\s+(?:a|an)\s+([A-F][+-]?)|
    no\s+grade\s+lower\s+than\s+([A-F][+-]?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

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
# Internal parsers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_course_codes(text: str) -> list[str]:
    """Pull all course codes from a raw text string."""
    codes: list[str] = []
    for match in COURSE_CODE_PAT.findall(text):
        normalized = normalize_course_code(match)
        if is_valid_course_code(normalized):
            codes.append(normalized)
    return codes


def _is_permission_only(text: str) -> bool:
    """True when the only requirement is instructor permission."""
    t = text.lower().strip()
    return (
        t in ("none", "permission of instructor", "permission of department")
        or t.startswith("permission of")
        or t == "none."
    )


def _extract_requirement_text(content: str, label: str) -> str | None:
    """
    Extract a Prereq/Coreq payload from potentially wrapped multiline text.

    The chunker can split course entries across lines, so requirement text may
    continue on the next line before the "U ("/"G (" schedule marker.
    """
    if label.lower().startswith("pre"):
        pattern = re.compile(
            r"Prereq(?:uisite)?s?:\s*(.+?)(?=\bCoreq(?:uisite)?s?:|\b[UG]\s*\(|\n\s*\d+\s*-\s*\d+\s*-\s*\d+\s+units?\b|$)",
            re.IGNORECASE | re.DOTALL,
        )
    else:
        pattern = re.compile(
            r"Coreq(?:uisite)?s?:\s*(.+?)(?=\bPrereq(?:uisite)?s?:|\b[UG]\s*\(|\n\s*\d+\s*-\s*\d+\s*-\s*\d+\s+units?\b|$)",
            re.IGNORECASE | re.DOTALL,
        )

    match = pattern.search(content)
    if not match:
        return None

    merged = re.sub(r"\s+", " ", match.group(1)).strip(" .;")
    return merged or None


def _parse_or_group(expr: str) -> list[list[str]]:
    """
    Parse a prerequisite boolean expression into DNF-style branches.

    Each returned branch is an AND-list of courses; branches are OR-ed.
    Example:
        "6.1000 or (6.100A and (6.100B or 16.C20[J]))"
        -> [["6.1000"], ["6.100A", "6.100B"], ["6.100A", "16.C20[J]"]]
    """
    # Tokenize into course codes/operators while skipping non-boolean filler text.
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(ch)
            i += 1
            continue
        if ch == ")":
            tokens.append(ch)
            i += 1
            continue
        if ch == ",":
            # Catalog text often uses ",or" as an OR separator.
            j = i + 1
            while j < len(expr) and expr[j].isspace():
                j += 1
            comma_or_match = re.match(r"(?i)or\b", expr[j:])
            if comma_or_match:
                tokens.append("OR")
                i = j + len(comma_or_match.group(0))
            else:
                tokens.append(",")
                i += 1
            continue

        course_match = COURSE_CODE_PAT.match(expr, i)
        if course_match:
            tokens.append(normalize_course_code(course_match.group(0)))
            i = course_match.end()
            continue

        and_match = re.match(r"(?i)\band\b", expr[i:])
        if and_match:
            tokens.append("AND")
            i += len(and_match.group(0))
            continue

        or_match = re.match(r"(?i)\bor\b", expr[i:])
        if or_match:
            tokens.append("OR")
            i += len(or_match.group(0))
            continue

        # Ignore unsupported tokens (e.g., permission wording).
        i += 1

    if not tokens:
        return []

    def _resolve_comma_tokens(items: list[str]) -> list[str]:
        """
        Resolve comma separators contextually.

        Catalog text commonly uses "A, B, C, or D" to mean OR between each item.
        When an OR appears before any same-level AND or closing paren, treat commas
        in that segment as OR; otherwise keep comma semantics as AND.
        """
        resolved = list(items)
        for idx, token in enumerate(resolved):
            if token != ",":
                continue

            depth = 0
            saw_or = False
            saw_and = False
            lookahead = idx + 1

            while lookahead < len(resolved):
                nxt = resolved[lookahead]
                if nxt == "(":
                    depth += 1
                elif nxt == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif depth == 0:
                    if nxt == "AND":
                        saw_and = True
                        break
                    if nxt == "OR":
                        saw_or = True
                        break
                lookahead += 1

            resolved[idx] = "OR" if saw_or and not saw_and else "AND"

        return resolved

    tokens = _resolve_comma_tokens(tokens)

    def _is_course_token(token: str) -> bool:
        return token not in {"AND", "OR", "(", ")", ","}

    def _dedupe_branches(branches: list[list[str]]) -> list[list[str]]:
        seen: set[tuple[str, ...]] = set()
        result: list[list[str]] = []
        for branch in branches:
            cleaned_branch: list[str] = []
            for code in branch:
                if code not in cleaned_branch:
                    cleaned_branch.append(code)
            if not cleaned_branch:
                continue
            key = tuple(cleaned_branch)
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned_branch)
        return result

    def _combine_or(left: list[list[str]], right: list[list[str]]) -> list[list[str]]:
        return _dedupe_branches(left + right)

    def _combine_and(left: list[list[str]], right: list[list[str]]) -> list[list[str]]:
        if not left:
            return right
        if not right:
            return left

        combined: list[list[str]] = []
        for l_branch in left:
            for r_branch in right:
                combined.append(l_branch + r_branch)
        return _dedupe_branches(combined)

    class _Parser:
        def __init__(self, items: list[str]) -> None:
            self.items = items
            self.idx = 0

        def peek(self) -> str | None:
            if self.idx >= len(self.items):
                return None
            return self.items[self.idx]

        def take(self) -> str | None:
            token = self.peek()
            if token is not None:
                self.idx += 1
            return token

        def parse_expression(self) -> list[list[str]]:
            left = self.parse_term()
            while self.peek() == "OR":
                self.take()
                right = self.parse_term()
                left = _combine_or(left, right)
            return left

        def parse_term(self) -> list[list[str]]:
            left = self.parse_factor()
            while True:
                nxt = self.peek()
                if nxt == "AND":
                    self.take()
                    right = self.parse_factor()
                    left = _combine_and(left, right)
                    continue
                # Support implicit AND between adjacent factors.
                if nxt is not None and nxt not in {"OR", ")"}:
                    right = self.parse_factor()
                    left = _combine_and(left, right)
                    continue
                return left

        def parse_factor(self) -> list[list[str]]:
            token = self.peek()
            if token is None:
                return []
            if token == "(":
                self.take()
                nested = self.parse_expression()
                if self.peek() == ")":
                    self.take()
                return nested
            if _is_course_token(token):
                self.take()
                return [[token]]
            # Skip unsupported token and continue.
            self.take()
            return []

    parser = _Parser(tokens)
    return _dedupe_branches(parser.parse_expression())


def _parse_prereq_expression(raw: str, target: str, chunk_id: str) -> list[dict]:
    """
    Parse a raw prereq string into a list of edge descriptors:
    [
        {
            "from": str,         # prerequisite course
            "to":   str,         # target course
            "requirement_type": "required" | "either_or",
            "min_grade": str,    # e.g. "C", or "" if not specified
            "either_or_group": str | None,
            "either_or_branch": str | None,
            "source_chunk_id": str,
        }
    ]
    """
    if _is_permission_only(raw):
        return []

    edges: list[dict] = []

    # Check for min grade hints like "grade of C or better"
    min_grade = ""
    grade_match = _MIN_GRADE_HINT_RE.search(raw)
    if grade_match:
        token = next((g for g in grade_match.groups() if g), "")
        if token:
            # Keep edge constraints compatible with letter-grade comparison logic.
            min_grade = token.upper()[0]

    # Strip grade hint text before parsing courses
    clean_raw = re.sub(
        r'(?:grade of|at least a?n?|or better)[^;,\n]*', '', raw, flags=re.IGNORECASE
    )
    branches = _parse_or_group(clean_raw)
    if not branches:
        # Fallback to simple extraction when boolean parsing fails.
        branches = [[code] for code in _extract_course_codes(clean_raw)]

    if not branches:
        # No parseable course prerequisites were found.
        return []

    if len(branches) == 1:
        # Pure AND list — all codes are required
        for code in branches[0]:
            edges.append({
                "from": code, "to": target,
                "requirement_type": "required",
                "min_grade": min_grade,
                "either_or_group": None,
                "either_or_branch": None,
                "source_chunk_id": chunk_id,
            })
    else:
        # Pull shared factors out of DNF branches so
        # A and (B or C) is represented as required A plus either_or(B, C).
        common_courses = set(branches[0])
        for branch in branches[1:]:
            common_courses &= set(branch)

        ordered_common = [code for code in branches[0] if code in common_courses]
        for code in ordered_common:
            edges.append({
                "from": code, "to": target,
                "requirement_type": "required",
                "min_grade": min_grade,
                "either_or_group": None,
                "either_or_branch": None,
                "source_chunk_id": chunk_id,
            })

        reduced_branches: list[list[str]] = []
        for branch in branches:
            reduced = [code for code in branch if code not in common_courses]
            if reduced and reduced not in reduced_branches:
                reduced_branches.append(reduced)

        if reduced_branches:
            group_id = str(uuid.uuid4())[:8]
            for idx, branch in enumerate(reduced_branches):
                branch_id = f"b{idx}"
                for code in branch:
                    edges.append({
                        "from": code, "to": target,
                        "requirement_type": "either_or",
                        "min_grade": min_grade,
                        "either_or_group": group_id,
                        "either_or_branch": branch_id,
                        "source_chunk_id": chunk_id,
                    })

    return edges


def _extract_global_gpa_requirements(text: str, chunk_id: str, document_name: str) -> list[dict]:
    """
    Pull policy-level GPA requirements from free-form text.

    These rules are not tied to a specific prerequisite edge, so they are
    stored in graph metadata for reasoning-time checks.
    """
    requirements: list[dict] = []
    seen: set[tuple[str, float, float | None, str, str]] = set()

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        snippet = re.sub(r"\s+", " ", sentence).strip()
        if not snippet:
            continue

        if not _GPA_TRIGGER_RE.search(snippet):
            continue
        if not _GPA_COMPARATOR_RE.search(snippet):
            continue

        operator = ">="
        lowered = snippet.lower()
        if re.search(r"\b(exceeding|greater than|above)\b", lowered) or re.search(
            r"(?<![<>=])>(?!=)", snippet
        ):
            operator = ">"

        scale_match = _GPA_SCALE_RE.search(snippet)
        threshold_value: float | None = None
        scale_max: float | None = None

        if scale_match:
            threshold_value = float(scale_match.group(1))
            scale_max = float(scale_match.group(2))
        else:
            threshold_match = _GPA_THRESHOLD_RE.search(snippet)
            if threshold_match:
                token = threshold_match.group(1).upper()
                if token in _LETTER_GPA_MAP:
                    threshold_value = _LETTER_GPA_MAP[token]
                    scale_max = 5.0
                else:
                    threshold_value = float(token)

        if threshold_value is None:
            continue

        audience = "all"
        if re.search(r"\bundergraduate\b", lowered):
            audience = "undergraduate"
        elif re.search(r"\bgraduate\b", lowered):
            audience = "graduate"

        scope = "financial_aid" if _FINANCIAL_AID_SCOPE_RE.search(lowered) else "general"

        dedupe_key = (operator, threshold_value, scale_max, audience, scope)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        requirements.append(
            {
                "type": "gpa",
                "operator": operator,
                "threshold": threshold_value,
                "scale_max": scale_max,
                "audience": audience,
                "scope": scope,
                "source_chunk_id": chunk_id,
                "document_name": document_name,
                "evidence_text": snippet[:240],
            }
        )

    return requirements


def _parse_coreq_expression(raw: str, target: str, chunk_id: str) -> list[dict]:
    """Parse corequisite string → corequisite edges."""
    if _is_permission_only(raw):
        return []
    codes = _extract_course_codes(raw)
    return [
        {
            "from": code, "to": target,
            "requirement_type": "corequisite",
            "min_grade": "",
            "either_or_group": None,
            "source_chunk_id": chunk_id,
        }
        for code in codes
    ]


def _extract_course_header(content: str) -> tuple[str | None, str, int]:
    """
    From a chunk's content, extract:
      - primary course code (e.g. "6.1010")
      - course name (e.g. "Fundamentals of Programming")
      - credits (int)

    MIT format:
      6.1010 Fundamentals of Programming
      Prereq: ...
      U (Fall, Spring)
      2-4-6 units
    """
    # Primary pattern: classic line-based header
    line_header_re = re.compile(
        r'^([\d]{1,2}\.[A-Z0-9]+(?:\[J\])?)\s+(.+?)(?=\n)',
        re.MULTILINE,
    )
    match = line_header_re.search(content)

    # Fallback pattern: single-line packed chunk where newline boundaries were removed
    # Example:
    #   "6.6000 CMOS Analog and Mixed-Signal Circuit Design Prereq: 6.2090 ..."
    if not match:
        inline_header_re = re.compile(
            r'^\s*([\d]{1,2}\.[A-Z0-9]+(?:\[J\])?)\s+(.+?)(?=\s+(?:Prereq(?:uisite)?s?:|Coreq(?:uisite)?s?:|[UG]\s*\(|\d+-\d+-\d+\s+units\b))',
            re.IGNORECASE | re.DOTALL,
        )
        match = inline_header_re.search(content)

    if not match:
        return None, "", 0

    code = match.group(1).strip().upper()
    name = re.sub(r'\s+', ' ', match.group(2)).strip()

    # Extract units: "2-4-6 units" → sum = 12
    units_match = re.search(r'(\d+)-(\d+)-(\d+)\s+units', content)
    credits = 0
    if units_match:
        credits = sum(int(x) for x in units_match.groups())

    return code, name, credits


def _to_int(value: object, default: int = 0) -> int:
    """Best-effort int cast for node attribute comparisons."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _merge_node_metadata(
    node_attrs: dict,
    course_name: str,
    credits: int,
    chunk_id: str,
    document_name: str,
    header_matched: bool,
) -> None:
    """
    Merge node metadata without allowing weak chunks to erase richer values.

    Rules:
      - Never overwrite a non-empty course_name with empty text.
      - Only overwrite course_name when incoming non-empty name is longer.
      - Only set credits when incoming credits are > 0 and existing credits are missing/zero.
      - description_chunk_id/document_name track the chunk that contributed strongest metadata.
    """
    incoming_name = (course_name or "").strip()
    existing_name = str(node_attrs.get("course_name", "") or "").strip()

    incoming_credits = _to_int(credits, 0)
    existing_credits = _to_int(node_attrs.get("credits", 0), 0)

    # Ensure stable keys for downstream serializers/clients.
    node_attrs.setdefault("course_name", "")
    node_attrs.setdefault("credits", 0)

    updated = False

    if incoming_name and (not existing_name or len(incoming_name) > len(existing_name)):
        node_attrs["course_name"] = incoming_name
        updated = True

    if incoming_credits > 0 and existing_credits <= 0:
        node_attrs["credits"] = incoming_credits
        updated = True

    # Only bind provenance to this chunk when it contributed useful metadata,
    # or when this is the first reliable header-backed observation.
    if updated or (header_matched and not node_attrs.get("description_chunk_id") and chunk_id):
        node_attrs["description_chunk_id"] = chunk_id
        if document_name:
            node_attrs["document_name"] = document_name


def _prune_isolated_sparse_nodes(G) -> int:
    """
    Remove nodes that are both:
      - sparse metadata (blank course_name and non-positive credits), and
      - isolated (no incoming or outgoing edges).

    This keeps connected prerequisite placeholders intact while cleaning
    dead-end artifacts from fallback extraction.
    """
    to_remove: list[str] = []

    for node_id, attrs in G.nodes(data=True):
        course_name = str(attrs.get("course_name", "") or "").strip()
        credits = _to_int(attrs.get("credits", 0), 0)
        if course_name or credits > 0:
            continue
        if G.in_degree(node_id) == 0 and G.out_degree(node_id) == 0:
            to_remove.append(node_id)

    if to_remove:
        G.remove_nodes_from(to_remove)

    return len(to_remove)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_from_chunks(enriched_chunks: list[dict]) -> None:
    """
    Main entry point called by ingestion/pipeline.py after enrichment.

    For each enriched chunk:
      1. Extract the target course code from the chunk content
      2. Add it as a node with attributes
      3. Parse Prereq / Coreq lines → add directed edges
    """
    G = get_graph()
    total_nodes = 0
    total_edges = 0
    skipped     = 0
    global_policy_requirements: list[dict] = []
    policy_seen: set[tuple[str, float, float | None, str, str, str]] = set()

    for chunk in enriched_chunks:
        content   = chunk.get("content", "")
        chunk_id  = chunk.get("chunk_id", "")
        metadata  = chunk.get("metadata", {})
        document_name = str(metadata.get("document_name", "") or "")

        for requirement in _extract_global_gpa_requirements(content, chunk_id, document_name):
            dedupe_key = (
                str(requirement.get("operator", ">=")),
                float(requirement.get("threshold", 0.0) or 0.0),
                float(requirement["scale_max"]) if requirement.get("scale_max") is not None else None,
                str(requirement.get("audience", "all")),
                str(requirement.get("scope", "general")),
                str(requirement.get("source_chunk_id", "")),
            )
            if dedupe_key in policy_seen:
                continue
            policy_seen.add(dedupe_key)
            global_policy_requirements.append(requirement)

        # ── Step 1: identify the target course code ───────────────────────
        target, course_name, credits = _extract_course_header(content)
        header_matched = bool(target)

        if target:
            target = normalize_course_code(target)
            if not is_valid_course_code(target):
                target = None
                course_name = ""
                credits = 0
                header_matched = False

        if not target:
            # Fall back to course_codes_mentioned if header parse failed
            mentioned = metadata.get("course_codes_mentioned", "")
            codes = [
                normalize_course_code(c)
                for c in mentioned.split(",")
                if c.strip() and is_valid_course_code(c)
            ]
            if not codes:
                # Last-resort fallback for chunks that still contain course mentions.
                codes = _extract_course_codes(content)
            if not codes:
                skipped += 1
                continue
            target = codes[0]
            course_name = ""
            credits = 0

        if not is_valid_course_code(target):
            skipped += 1
            continue

        # ── Step 2: add node ──────────────────────────────────────────────
        if target not in G:
            G.add_node(target)
            total_nodes += 1

        _merge_node_metadata(
            node_attrs=G.nodes[target],
            course_name=course_name,
            credits=credits,
            chunk_id=chunk_id,
            document_name=document_name,
            header_matched=header_matched,
        )

        # Only parse prerequisite/corequisite edges when the chunk has an
        # explicit course header. Fallback targets from "course_codes_mentioned"
        # are useful for node hints but can mis-attach edges from mixed chunks.
        if not header_matched:
            continue

        # ── Step 3: parse prerequisite line ──────────────────────────────
        prereq_raw = _extract_requirement_text(content, "Prereq")
        if prereq_raw:
            edges = _parse_prereq_expression(prereq_raw, target, chunk_id)
            for edge in edges:
                src = edge["from"]
                if not is_valid_course_code(src):
                    continue
                if src == target:
                    log.warning(f"Skipping self-loop prereq edge {src} -> {target} from chunk {chunk_id}")
                    continue
                if src not in G:
                    G.add_node(src)
                    total_nodes += 1

                req_type = edge["requirement_type"]
                group_id = edge["either_or_group"]
                branch_id = edge.get("either_or_branch")
                branch_memberships: list[str] = []
                if req_type == "either_or" and group_id and branch_id:
                    branch_memberships = [str(branch_id)]

                if G.has_edge(src, target):
                    existing = G.get_edge_data(src, target) or {}
                    if (
                        req_type == "either_or"
                        and str(existing.get("requirement_type", "")) == "either_or"
                        and str(existing.get("either_or_group", "")) == str(group_id)
                    ):
                        existing_branches = existing.get("either_or_branches", []) or []
                        branch_memberships = [str(item) for item in existing_branches if str(item).strip()]
                        if branch_id:
                            branch_memberships.append(str(branch_id))
                        branch_memberships = sorted(set(branch_memberships))

                G.add_edge(
                    src, target,
                    requirement_type = req_type,
                    min_grade        = edge["min_grade"],
                    either_or_group  = group_id,
                    either_or_branches = branch_memberships,
                    source_chunk_id  = edge["source_chunk_id"],
                )
                total_edges += 1

        # ── Step 4: parse corequisite line ────────────────────────────────
        coreq_raw = _extract_requirement_text(content, "Coreq")
        if coreq_raw:
            edges = _parse_coreq_expression(coreq_raw, target, chunk_id)
            for edge in edges:
                src = edge["from"]
                if not is_valid_course_code(src):
                    continue
                if src == target:
                    log.warning(f"Skipping self-loop coreq edge {src} -> {target} from chunk {chunk_id}")
                    continue
                if src not in G:
                    G.add_node(src)
                    total_nodes += 1
                G.add_edge(
                    src, target,
                    requirement_type = "corequisite",
                    min_grade        = "",
                    either_or_group  = None,
                    source_chunk_id  = edge["source_chunk_id"],
                )
                total_edges += 1

    log.info(
        f"Graph built: {total_nodes} nodes, {total_edges} edges added. "
        f"{skipped} chunks skipped (no course code found)."
    )

    G.graph["global_policy_requirements"] = global_policy_requirements
    if global_policy_requirements:
        log.info(f"Captured {len(global_policy_requirements)} global policy requirement(s).")

    pruned = _prune_isolated_sparse_nodes(G)
    if pruned:
        log.info(f"Pruned {pruned} isolated sparse nodes.")

    save()