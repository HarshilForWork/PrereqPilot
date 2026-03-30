from src.inference.prompts import COURSE_PLAN_PROMPT, PREREQUISITE_CHECK_PROMPT, VERIFIER_PROMPT


def render_prereq_prompt(context: str, question: str, graph_result: str) -> tuple[str, str]:
	messages = PREREQUISITE_CHECK_PROMPT.format_messages(
		context=context,
		question=question,
		graph_result=graph_result,
	)
	return messages[0].content, messages[1].content


def render_plan_prompt(profile: str, context: str, graph_result: str) -> tuple[str, str]:
	messages = COURSE_PLAN_PROMPT.format_messages(
		profile=profile,
		context=context,
		graph_result=graph_result,
	)
	return messages[0].content, messages[1].content


def render_verifier_prompt(draft: str, context: str, graph_result: str) -> tuple[str, str]:
	messages = VERIFIER_PROMPT.format_messages(
		draft=draft,
		context=context,
		graph_result=graph_result,
	)
	return messages[0].content, messages[1].content
