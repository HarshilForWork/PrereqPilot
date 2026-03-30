from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.runnables import RunnableLambda, RunnableParallel

try:
	from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional runtime dependency
	ChatGroq = None

from src.agents.intake import IntakeAgent
from src.agents.planner import PlannerAgent
from src.agents.retriever_agent import CatalogRetrieverAgent
from src.agents.verifier import VerifierAgent
from src.core.config import settings
from src.core.constants import ABSTENTION_MSG, COURSE_CODE_PAT
from src.core.exceptions import GraphNodeNotFoundError
from src.core.logger import get_logger
from src.graph.reasoning import check_eligibility, get_available_next_courses

log = get_logger(__name__)

# Keep Crew runtime quiet and avoid best-effort telemetry export timeouts in local/dev runs.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

try:
	from crewai import Agent as CrewAgent
	from crewai import Crew, Process, Task
	from crewai import LLM as CrewLLM
	from crewai.llm import BaseLLM
except Exception:  # pragma: no cover - optional runtime dependency
	CrewAgent = None
	Crew = None
	Process = None
	Task = None
	CrewLLM = None
	BaseLLM = None


class CrewRuntime:
	def __init__(
		self,
		intake_agent: IntakeAgent,
		retriever_agent: CatalogRetrieverAgent,
		planner_agent: PlannerAgent,
		verifier_agent: VerifierAgent,
	) -> None:
		self._intake = intake_agent
		self._retriever = retriever_agent
		self._planner = planner_agent
		self._verifier = verifier_agent

	def _extract_target_course(self, question: str) -> str:
		match = COURSE_CODE_PAT.search(question.upper())
		if not match:
			return ""
		return match.group(0).replace(" ", "").upper()

	def _normalize_decision(self, decision: str, fallback: str = "Need more info") -> str:
		allowed = {"Eligible", "Not eligible", "Need more info"}
		if decision in allowed:
			return decision
		if fallback in allowed:
			return fallback
		return "Need more info"

	def _default_graph_result(self) -> dict[str, Any]:
		return {
			"eligible": False,
			"decision": "Need more info",
			"missing_prereqs": [],
			"missing_coreqs": [],
			"grade_issues": [],
			"gpa_issues": [],
			"prereq_path": [],
			"either_or_options": [],
		}

	def _make_evidence(self, citations: list[dict[str, Any]]) -> str:
		if not citations:
			return "No cited evidence available in retrieved context."
		lines = []
		for citation in citations:
			lines.append(
				"[SOURCE: "
				f"{citation.get('document_name', '')}, "
				f"Section: {citation.get('section_heading', '')}, "
				f"Chunk: {citation.get('chunk_id', '')}]"
			)
		return "\n".join(lines)

	def _next_step(self, graph_result: dict[str, Any]) -> str:
		missing = graph_result.get("missing_prereqs", [])
		if missing:
			return f"Complete missing prerequisites first: {', '.join(missing)}."
		either_or_options = graph_result.get("either_or_options", [])
		if either_or_options:
			groups: list[str] = []
			for options in either_or_options[:3]:
				if not isinstance(options, list):
					continue
				cleaned = [str(option).strip() for option in options if str(option).strip()]
				if cleaned:
					formatted = [f"({option})" if " and " in option.lower() else option for option in cleaned[:4]]
					groups.append(" OR ".join(formatted))
			if groups:
				return f"Complete one option from each prerequisite group: {'; '.join(groups)}."
		gpa_issues = graph_result.get("gpa_issues", [])
		if gpa_issues:
			if any("student level not provided" in str(issue).lower() for issue in gpa_issues):
				return "Share whether you are an undergraduate or graduate student for accurate GPA policy checks."
			if any("not provided" in str(issue).lower() for issue in gpa_issues):
				return "Share your current cumulative GPA for complete eligibility verification."
			return "Review GPA requirement constraints and contact your advisor if needed."
		if graph_result.get("decision") == "Need more info":
			return "Share additional profile details for a precise eligibility check."
		return "You can proceed with enrollment verification through your department/advisor."

	def _build_crewai_groq_llm(self, model_name: str, groq_key: str) -> Any | None:
		if ChatGroq is None or BaseLLM is None:
			return None

		chat_model = ChatGroq(
			model=model_name,
			groq_api_key=groq_key,
			temperature=float(settings.TEMP_REASONING),
			max_tokens=int(settings.LLM_MAX_TOKENS),
		)

		class CrewGroqLLMAdapter(BaseLLM):
			def __init__(self) -> None:
				super().__init__(
					model=model_name,
					temperature=float(settings.TEMP_REASONING),
					api_key=groq_key,
					provider="groq",
				)
				self._chat_model = chat_model

			@staticmethod
			def _serialize_messages(messages: Any) -> str:
				if isinstance(messages, str):
					return messages

				lines: list[str] = []
				for msg in messages or []:
					if isinstance(msg, dict):
						role = str(msg.get("role", "user"))
						content = msg.get("content", "")
					else:
						role = str(getattr(msg, "role", "user"))
						content = getattr(msg, "content", "")

					if isinstance(content, list):
						parts: list[str] = []
						for item in content:
							if isinstance(item, dict):
								parts.append(str(item.get("text", "")))
							else:
								parts.append(str(item))
						content = " ".join(parts).strip()

					lines.append(f"{role}: {content}")
				return "\n".join(lines).strip()

			def call(
				self,
				messages: Any,
				tools: Any = None,
				callbacks: Any = None,
				available_functions: Any = None,
				from_task: Any = None,
				from_agent: Any = None,
				response_model: Any = None,
			) -> str:
				prompt = self._serialize_messages(messages)
				response = self._chat_model.invoke(prompt)
				content = getattr(response, "content", "")
				if isinstance(content, list):
					parts: list[str] = []
					for item in content:
						if isinstance(item, dict):
							parts.append(str(item.get("text", "")))
						else:
							parts.append(str(item))
					return " ".join(parts).strip()
				return str(content or "")

		return CrewGroqLLMAdapter()

	def _normalize_groq_model(self, model_name: str) -> tuple[str, str]:
		normalized = (model_name or "").strip()
		if not normalized:
			normalized = str(settings.LLM_MODEL or "").strip()
		if not normalized:
			normalized = "openai/gpt-oss-120b"
		if normalized.startswith("groq/"):
			return normalized, normalized.split("/", 1)[1]
		return f"groq/{normalized}", normalized

	def _resolve_crewai_llm(self) -> tuple[Any | None, bool]:
		explicit_model = str(os.getenv("CREWAI_MODEL", "") or "").strip()
		configured_model = str(settings.LLM_MODEL or "").strip()
		groq_key = str(settings.GROQ_API_KEY or "").strip()
		if groq_key and not groq_key.startswith("dev-placeholder"):
			preferred_model = explicit_model or configured_model or str(os.getenv("CREWAI_GROQ_MODEL", "") or "").strip()
			crewai_model, adapter_model = self._normalize_groq_model(preferred_model)

			if CrewLLM is not None:
				try:
					return (
						CrewLLM(
							model=crewai_model,
							api_key=groq_key,
							temperature=float(settings.TEMP_REASONING),
							max_tokens=int(settings.LLM_MAX_TOKENS),
						),
						True,
					)
				except Exception as exc:
					log.warning(
						"CrewAI LLM initialization failed for model %s (%s); falling back to adapter.",
						crewai_model,
						exc,
					)

			adapter = self._build_crewai_groq_llm(adapter_model, groq_key)
			if adapter is not None:
				return adapter, True

			log.warning("Groq CrewAI LLM fallback unavailable; langchain-groq/CrewAI BaseLLM not ready.")
			return None, False

		if explicit_model:
			return explicit_model, explicit_model.startswith("groq/")

		openai_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
		if openai_key:
			return str(os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip(), False

		return None, False

	def _create_crewai_crew(self, mode: str) -> Any | None:
		if not bool(settings.CREWAI_ENABLED):
			return None

		if Crew is None or CrewAgent is None or Task is None:
			log.warning("CrewAI package not available; continuing with LangChain stage orchestration.")
			return None

		crewai_llm, is_groq_llm = self._resolve_crewai_llm()
		if not crewai_llm:
			log.warning(
				"CrewAI model/provider not configured; set CREWAI_MODEL or configure GROQ_API_KEY/OPENAI_API_KEY. "
				"Continuing with LangChain stage orchestration."
			)
			return None

		try:
			intake = CrewAgent(
				role="Intake Agent",
				goal="Normalize student profile and identify missing info",
				backstory="Handles student profile normalization and clarification checks.",
				llm=crewai_llm,
				verbose=settings.CREWAI_VERBOSE,
				allow_delegation=False,
			)
			retriever = CrewAgent(
				role="Catalog Retriever Agent",
				goal="Retrieve catalog evidence and citations",
				backstory="Finds relevant catalog chunks and citation metadata.",
				llm=crewai_llm,
				verbose=settings.CREWAI_VERBOSE,
				allow_delegation=False,
			)
			planner = CrewAgent(
				role="Planner Agent",
				goal="Produce policy-grounded student guidance",
				backstory="Generates structured plans and eligibility guidance from evidence.",
				llm=crewai_llm,
				verbose=settings.CREWAI_VERBOSE,
				allow_delegation=False,
			)
			verifier = CrewAgent(
				role="Verifier Agent",
				goal="Audit factual support and decision consistency",
				backstory="Checks citation support and graph reasoning consistency.",
				llm=crewai_llm,
				verbose=settings.CREWAI_VERBOSE,
				allow_delegation=False,
			)

			tasks = [
				Task(
					description=f"Run intake normalization for {mode} query request.",
					expected_output="Normalized profile payload.",
					agent=intake,
				),
				Task(
					description=f"Retrieve relevant catalog context for {mode} query.",
					expected_output="Retrieved chunks with citations.",
					agent=retriever,
				),
				Task(
					description=f"Generate structured planner response for {mode} query.",
					expected_output="Structured planner output.",
					agent=planner,
				),
				Task(
					description="Audit the response for citation and consistency issues.",
					expected_output="Structured verification output.",
					agent=verifier,
				),
			]

			use_memory = bool(settings.CREWAI_MEMORY)
			if is_groq_llm and use_memory:
				log.warning("CrewAI memory disabled for Groq model compatibility.")
				use_memory = False

			process = Process.sequential if Process is not None else "sequential"
			return Crew(
				agents=[intake, retriever, planner, verifier],
				tasks=tasks,
				verbose=settings.CREWAI_VERBOSE,
				memory=use_memory,
				process=process,
			)
		except Exception as exc:
			log.warning(
				"CrewAI initialization failed (%s); continuing with LangChain stage orchestration.",
				exc,
			)
			return None

	async def _touch_crewai(self, mode: str, payload: dict[str, Any]) -> None:
		"""
		Instantiate and kickoff CrewAI orchestration metadata pass.

		The authoritative response is still produced by LangChain-backed stage chains,
		but this keeps CrewAI as the orchestration layer entrypoint when enabled.
		"""
		if not bool(settings.CREWAI_ENABLED):
			return

		crew = self._create_crewai_crew(mode)
		if crew is None:
			return

		def _kickoff() -> None:
			crew.kickoff(inputs={"mode": mode, "payload": payload})

		try:
			await asyncio.wait_for(
				asyncio.to_thread(_kickoff),
				timeout=float(settings.CREWAI_TIMEOUT_SECONDS),
			)
		except Exception:
			log.exception("CrewAI kickoff failed; continuing with LangChain stage orchestration.")

	async def run_prereq(self, payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
		question = str(payload.get("question", "") or "")
		await self._touch_crewai("prereq", payload)

		profile = await self._intake.run(payload, mode="prereq")
		missing_profile_fields = [
			str(field).strip()
			for field in (profile.get("missing_fields", []) or [])
			if str(field).strip()
		]
		needs_profile_clarification = bool(missing_profile_fields)

		target_course = self._extract_target_course(question)
		enforce_gpa_policies = bool(
			profile.get("gpa", None) is not None
			or str(profile.get("student_level", "")).strip()
			or bool(profile.get("include_financial_aid_policies", False))
		)

		async def _graph_task() -> dict[str, Any]:
			if not target_course:
				return self._default_graph_result()
			try:
				return await asyncio.to_thread(
					check_eligibility,
					target_course,
					profile.get("completed_courses", []),
					profile.get("grades", {}),
					student_gpa=profile.get("gpa", None),
					student_level=profile.get("student_level", ""),
					enforce_gpa_policies=enforce_gpa_policies,
					include_financial_aid_policies=bool(profile.get("include_financial_aid_policies", False)),
				)
			except GraphNodeNotFoundError:
				return self._default_graph_result()

		filters = {"catalog_year": profile.get("catalog_year", "")} if profile.get("catalog_year") else None

		async def _vector_task(_: dict[str, Any]) -> dict[str, Any]:
			return await self._retriever.run(question, filters=filters)

		async def _graph_stage_task(_: dict[str, Any]) -> dict[str, Any]:
			return await _graph_task()

		stage_parallel = RunnableParallel(
			vector_ctx=RunnableLambda(_vector_task),
			graph_result=RunnableLambda(_graph_stage_task),
		)
		parallel_result = await stage_parallel.ainvoke({})
		vector_ctx = parallel_result.get("vector_ctx", {})
		graph_result = parallel_result.get("graph_result", self._default_graph_result())
		graph_result_for_prereq = graph_result
		if needs_profile_clarification:
			graph_result_for_prereq = self._default_graph_result()

		planner_result = await self._planner.run(
			profile=profile,
			vector_ctx=vector_ctx,
			graph_result=graph_result_for_prereq,
			mode="prereq",
			question=question,
		)
		effective_decision = self._normalize_decision(
			str(graph_result_for_prereq.get("decision", "Need more info")),
			fallback="Need more info",
		)

		draft = {
			"decision": effective_decision,
			"answer": str(planner_result.get("answer", ABSTENTION_MSG)),
			"evidence": self._make_evidence(planner_result.get("citations", [])),
			"citations": planner_result.get("citations", []),
			"next_step": self._next_step(graph_result_for_prereq),
			"clarifying_questions": (
				planner_result.get("clarifying_questions", [])
				if effective_decision == "Need more info"
				else []
			),
			"graph_result": {
				"eligible": bool(graph_result_for_prereq.get("eligible", False)),
				"missing_prereqs": graph_result_for_prereq.get("missing_prereqs", []),
				"missing_coreqs": graph_result_for_prereq.get("missing_coreqs", []),
				"grade_issues": graph_result_for_prereq.get("grade_issues", []),
				"gpa_issues": graph_result_for_prereq.get("gpa_issues", []),
				"prereq_path": graph_result_for_prereq.get("prereq_path", []),
				"either_or_options": graph_result_for_prereq.get("either_or_options", []),
			},
			"assumptions": planner_result.get("assumptions", []),
			"trace_id": trace_id,
		}

		verified = await self._verifier.run(draft, vector_ctx.get("retrieved_chunks", []), graph_result_for_prereq)
		final_payload = dict(verified["final_response"])
		final_payload["decision"] = self._normalize_decision(
			str(final_payload.get("decision", draft["decision"])),
			fallback=draft["decision"],
		)
		final_payload["trace_id"] = trace_id

		chunks_retrieved = len(vector_ctx.get("retrieved_chunks", []))
		citation_count = len(final_payload.get("citations", []))
		abstained = ABSTENTION_MSG in str(final_payload.get("answer", ""))
		return {
			"response": final_payload,
			"chunks_retrieved": chunks_retrieved,
			"citation_count": citation_count,
			"abstained": abstained,
			"graph_used": True,
			"verifier_passed": bool(verified.get("passed", False)),
			"verifier_issues": verified.get("issues", []),
			"agents_called": [
				"IntakeAgent",
				"CatalogRetrieverAgent",
				"PlannerAgent",
				"VerifierAgent",
				"CrewRuntime",
			],
		}

	async def run_plan(self, payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
		await self._touch_crewai("plan", payload)

		profile = await self._intake.run(payload, mode="plan")

		plan_question = str(payload.get("question", "") or "").strip()
		if not plan_question:
			plan_question = (
				f"Create a {profile.get('target_term', '')} course plan for {profile.get('target_major', '')} "
				f"with at most {profile.get('max_credits', 0)} credits."
			)
		filters = {"catalog_year": profile.get("catalog_year", "")} if profile.get("catalog_year") else None

		async def _vector_task(_: dict[str, Any]) -> dict[str, Any]:
			return await self._retriever.run(plan_question, filters=filters)

		async def _available_courses_task(_: dict[str, Any]) -> list[Any]:
			completed_courses = profile.get("completed_courses", [])
			if not completed_courses:
				return []
			return await asyncio.to_thread(get_available_next_courses, completed_courses)

		stage_parallel = RunnableParallel(
			vector_ctx=RunnableLambda(_vector_task),
			available_courses=RunnableLambda(_available_courses_task),
		)
		parallel_result = await stage_parallel.ainvoke({})
		vector_ctx = parallel_result.get("vector_ctx", {})
		available_courses = parallel_result.get("available_courses", [])

		graph_result = {"available_next_courses": available_courses}
		planner_result = await self._planner.run(
			profile=profile,
			vector_ctx=vector_ctx,
			graph_result=graph_result,
			mode="plan",
			question=plan_question,
		)

		draft = {
			"answer": planner_result.get("answer", ""),
			"plan": planner_result.get("plan", []),
			"citations": planner_result.get("citations", []),
			"clarifying_questions": planner_result.get("clarifying_questions", []),
			"assumptions": planner_result.get("assumptions", []),
			"risks": planner_result.get("risks", []),
		}

		verified = await self._verifier.run(draft, vector_ctx.get("retrieved_chunks", []), graph_result)
		final_payload = dict(verified["final_response"])
		final_payload["trace_id"] = trace_id
		final_payload["total_credits"] = int(
			sum(int(course.get("credits", 0) or 0) for course in final_payload.get("plan", []))
		)

		chunks_retrieved = len(vector_ctx.get("retrieved_chunks", []))
		citation_count = len(final_payload.get("citations", []))
		abstained = ABSTENTION_MSG in str(final_payload.get("answer", ""))
		return {
			"response": final_payload,
			"chunks_retrieved": chunks_retrieved,
			"citation_count": citation_count,
			"abstained": abstained,
			"graph_used": True,
			"verifier_passed": bool(verified.get("passed", False)),
			"verifier_issues": verified.get("issues", []),
			"agents_called": [
				"IntakeAgent",
				"CatalogRetrieverAgent",
				"PlannerAgent",
				"VerifierAgent",
				"CrewRuntime",
			],
		}

	async def run_ask(self, payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
		await self._touch_crewai("ask", payload)

		question = str(payload.get("question", "") or "")
		filters = payload.get("filters")

		profile = await self._intake.run({}, mode="ask")
		vector_ctx = await self._retriever.run(question, filters=filters)
		planner_result = await self._planner.run(
			profile=profile,
			vector_ctx=vector_ctx,
			graph_result={},
			mode="ask",
			question=question,
		)

		draft = {
			"answer": str(planner_result.get("answer", ABSTENTION_MSG)),
			"citations": planner_result.get("citations", []),
			"abstained": ABSTENTION_MSG in str(planner_result.get("answer", "")),
		}
		verified = await self._verifier.run(draft, vector_ctx.get("retrieved_chunks", []), {})
		final_payload = dict(verified["final_response"])
		final_payload["abstained"] = bool(final_payload.get("abstained", False)) or (
			ABSTENTION_MSG in str(final_payload.get("answer", ""))
		)
		final_payload["trace_id"] = trace_id

		chunks_retrieved = len(vector_ctx.get("retrieved_chunks", []))
		citation_count = len(final_payload.get("citations", []))
		abstained = bool(final_payload.get("abstained", False))
		return {
			"response": final_payload,
			"chunks_retrieved": chunks_retrieved,
			"citation_count": citation_count,
			"abstained": abstained,
			"graph_used": False,
			"verifier_passed": bool(verified.get("passed", False)),
			"verifier_issues": verified.get("issues", []),
			"agents_called": [
				"IntakeAgent",
				"CatalogRetrieverAgent",
				"PlannerAgent",
				"VerifierAgent",
				"CrewRuntime",
			],
		}
