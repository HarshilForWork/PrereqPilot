# Evaluation Report (25 Queries)

## Dataset Split
- Prereq checks: 10
- Multi-hop chain: 5
- Program requirement: 5
- Trick / not-in-docs: 5

Source: `tests/eval/eval_set.json`

## Metrics Summary
- Total queries: 25
- Citation coverage rate: 100.0%
- Eligibility correctness rate: 100.0%
- Abstention accuracy rate: 100.0%

## Metric Definitions
- Citation coverage:
  - Numerator: queries with `expected_citations_present=true` where response had at least one citation.
  - Denominator: all queries with `expected_citations_present=true`.
- Eligibility correctness:
  - Numerator: prereq checks where actual decision matched `expected_decision`.
  - Denominator: all prereq checks with non-null `expected_decision`.
- Abstention accuracy:
  - Numerator: trick queries where response `abstained` matched `expected_abstained=true`.
  - Denominator: all queries with `expected_abstained` set.

## Eligibility Correctness Rubric
A prereq decision is graded correct only when all are true:
- Decision label exactly matches one of: `Eligible`, `Not eligible`, `Need more info`.
- Label equals `expected_decision` in eval set.
- Decision is produced from graph reasoning for the extracted target course code.

## Example Transcript 1: Eligibility + Citations
User: Can I enroll in 6.1060 after 6.1020, 6.1210, and 6.1910?

Assistant:
- Decision: Eligible
- Answer:
  - DECISION: Eligible
  - EVIDENCE: [SOURCE: 6_coursestext.pdf, Section: , Chunk: 6_coursestext.pdf_230_37f280f6]; [SOURCE: 6_coursestext.pdf, Section: , Chunk: 6_coursestext.pdf_130_4090fc82]; [SOURCE: 6_coursestext.pdf, Section: , Chunk: 6_coursestext.pdf_10_565cdc3c]
  - NEXT STEP: Share missing prerequisite details with your advisor if any requirement is unclear.

## Example Transcript 2: Plan + Citations
User: Build a Fall term plan for Course 6-3 with up to 12 credits.

Assistant (excerpt):
- Suggested courses (sample): 1.000, 1.066[J], 1.686[J], 1.C01
- Rationale style: Recommended based on available prerequisites and retrieved catalog context.
- Citations:
  - [SOURCE: 6_coursestext.pdf, Section: , Chunk: 6_coursestext.pdf_132_fd64393d]
  - [SOURCE: 6_coursestext.pdf, Section: , Chunk: 6_coursestext.pdf_258_1d0a0299]
  - [SOURCE: 6_coursestext.pdf, Section: , Chunk: 6_coursestext.pdf_141_c2494452]

## Example Transcript 3: Abstention + Guidance
User: What is the Wi-Fi password in Stata Center right now?

Assistant:
- Abstained: true
- Answer: I don't have that information in the provided catalog/policies. Please check: your academic advisor, the department's official website, or the current schedule of classes.
- Guidance: For instructor/schedule/facility details, check official MIT registrar and department sites.

## Artifacts
- Full machine-readable report: `tests/eval/eval_report.json`
- Eval runner: `tests/eval/run_eval.py`
