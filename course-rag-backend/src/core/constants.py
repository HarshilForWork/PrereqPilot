import re


ABSTENTION_MSG = (
	"I don't have that information in the provided catalog/policies. "
	"Please check: your academic advisor, the department's official website, "
	"or the current schedule of classes."
)

COURSE_CODE_RE = r"\b[A-Z]{2,4}\s?\d{3,4}[A-Z]?\b"
COURSE_CODE_TOKEN_RE = (
	r"(?:\d{1,2}[A-Z]{0,2}\.[A-Z0-9]{2,}(?:\[J\])?|[A-Z]{1,4}\.[A-Z0-9]{2,}(?:\[J\])?)"
)
COURSE_CODE_PAT = re.compile(
	rf"(?<![A-Z0-9]){COURSE_CODE_TOKEN_RE}(?=$|[^A-Z0-9])",
	re.IGNORECASE,
)
COURSE_CODE_FULL_REGEX = re.compile(rf"^{COURSE_CODE_TOKEN_RE}$", re.IGNORECASE)
CATALOG_YEAR_RE = r"\b(20\d{2}[-–]\d{2,4})\b"


def normalize_course_code(value: str) -> str:
	return value.strip().replace(" ", "").upper()


def is_valid_course_code(value: str) -> bool:
	return bool(COURSE_CODE_FULL_REGEX.fullmatch(normalize_course_code(value)))

PREREQ_KW = [
	"prerequisite",
	"prereq",
	"pre-req",
	"must have completed",
	"prior to",
]

GRADE_KW = [
	"grade of",
	"minimum grade",
	"c or better",
	"b or higher",
	"at least a",
]

CREDIT_KW = [
	"credit hours",
	"credit units",
	"credits",
	"units",
]

COREQ_KW = [
	"co-requisite",
	"corequisite",
	"coreq",
	"taken concurrently",
]

COURSE_CODE_REGEX = re.compile(COURSE_CODE_RE)
CATALOG_YEAR_REGEX = re.compile(CATALOG_YEAR_RE)
