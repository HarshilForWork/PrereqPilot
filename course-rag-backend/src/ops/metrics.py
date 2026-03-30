from dataclasses import asdict, dataclass


@dataclass
class Metrics:
	total_requests: int = 0
	prereq_requests: int = 0
	plan_requests: int = 0
	abstentions: int = 0
	verifier_rewrites: int = 0
	citation_hits: int = 0
	avg_latency_ms: float = 0.0
	avg_chunks_retrieved: float = 0.0


_METRICS = Metrics()


def _update_running_average(current_avg: float, current_count: int, new_value: float) -> float:
	if current_count <= 0:
		return new_value
	return ((current_avg * (current_count - 1)) + new_value) / current_count


def snapshot() -> dict:
	return asdict(_METRICS)


def increment_total_requests() -> None:
	_METRICS.total_requests += 1


def increment_prereq_requests() -> None:
	_METRICS.prereq_requests += 1


def increment_plan_requests() -> None:
	_METRICS.plan_requests += 1


def increment_abstentions() -> None:
	_METRICS.abstentions += 1


def increment_verifier_rewrites() -> None:
	_METRICS.verifier_rewrites += 1


def add_citation_hits(count: int) -> None:
	_METRICS.citation_hits += max(0, count)


def observe_latency_ms(latency_ms: float) -> None:
	_METRICS.avg_latency_ms = _update_running_average(
		_METRICS.avg_latency_ms,
		_METRICS.total_requests,
		latency_ms,
	)


def observe_chunks_retrieved(chunks: int) -> None:
	_METRICS.avg_chunks_retrieved = _update_running_average(
		_METRICS.avg_chunks_retrieved,
		_METRICS.total_requests,
		float(chunks),
	)
