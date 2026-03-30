import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { queryPlan } from '../../services/api';
import CitationSection from '../shared/CitationSection';
import ClarifyingQuestionsUI from '../shared/ClarifyingQuestionsUI';
import ResponseFallback from '../shared/ResponseFallback';

export default function PlanView() {
  const { state, dispatch } = useAppContext();
  const { status, data, error } = state.plan;

  const [completedCourses, setCompletedCourses] = useState('');
  const [targetMajor, setTargetMajor] = useState('');
  const [targetTerm, setTargetTerm] = useState('');
  const [maxCredits, setMaxCredits] = useState('');
  const [question, setQuestion] = useState('');

  const buildRequest = (extraContext?: string): { student_profile: any; question: string } => {
    const baseQuestion = question || "Generate a course plan";
    return {
      question: extraContext
        ? `${baseQuestion}\n\nAdditional context: ${extraContext}`
        : baseQuestion,
      student_profile: {
        completed_courses: completedCourses.split(',').map((c) => c.trim()).filter(Boolean),
        target_major: targetMajor || undefined,
        target_term: targetTerm || undefined,
        max_credits: maxCredits ? parseInt(maxCredits, 10) : undefined,
      },
    };
  };

  const submit = async (extra?: string) => {
    dispatch({ type: 'FETCH_START', mode: 'plan' });
    try {
      const res = await queryPlan(buildRequest(extra));
      dispatch({ type: 'FETCH_SUCCESS_PLAN', data: res });
    } catch (e: unknown) {
      dispatch({ type: 'FETCH_ERROR', mode: 'plan', error: (e as Error).message });
    }
  };

  const retry = () => submit();

  return (
    <div className="mode-view">
      {/* ── Form ── */}
      <form className="query-form" onSubmit={(e) => { e.preventDefault(); submit(); }} aria-label="Course plan form">
        <div className="form-group">
          <label htmlFor="plan-question" className="form-label">What would you like to plan?</label>
          <textarea
            id="plan-question"
            className="form-textarea"
            rows={2}
            placeholder="e.g. Plan for Fall 2026 for Computer Science major"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={status === 'loading'}
          />
        </div>
        <div className="form-group">
          <label htmlFor="plan-courses" className="form-label">
            Completed courses <span className="form-hint">(comma-separated)</span>
          </label>
          <input
            id="plan-courses"
            type="text"
            className="form-input"
            placeholder="CS101, CS201, MATH120"
            value={completedCourses}
            onChange={(e) => setCompletedCourses(e.target.value)}
            disabled={status === 'loading'}
          />
        </div>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="plan-major" className="form-label">Target major</label>
            <input
              id="plan-major"
              type="text"
              className="form-input"
              placeholder="Computer Science"
              value={targetMajor}
              onChange={(e) => setTargetMajor(e.target.value)}
              disabled={status === 'loading'}
            />
          </div>
          <div className="form-group">
            <label htmlFor="plan-term" className="form-label">Target term</label>
            <input
              id="plan-term"
              type="text"
              className="form-input"
              placeholder="Fall 2025"
              value={targetTerm}
              onChange={(e) => setTargetTerm(e.target.value)}
              disabled={status === 'loading'}
            />
          </div>
          <div className="form-group form-group--sm">
            <label htmlFor="plan-credits" className="form-label">Max credits</label>
            <input
              id="plan-credits"
              type="number"
              className="form-input"
              placeholder="18"
              min={1}
              max={30}
              value={maxCredits}
              onChange={(e) => setMaxCredits(e.target.value)}
              disabled={status === 'loading'}
            />
          </div>
        </div>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={status === 'loading'}
          aria-busy={status === 'loading'}
        >
          {status === 'loading' ? 'Planning…' : 'Generate Course Plan'}
        </button>
      </form>

      {/* ── States ── */}
      {status === 'loading' && <ResponseFallback type="loading" />}
      {status === 'error' && <ResponseFallback type="error" message={error ?? undefined} onRetry={retry} />}
      {status === 'idle' && !data && <ResponseFallback type="empty" message="Fill in your profile and generate a plan." />}

      {/* ── Success ── */}
      {data && (
        <article className="response-card" aria-label="Course plan result">
          {/* Clarifying questions first */}
          <ClarifyingQuestionsUI
            questions={data.clarifying_questions ?? []}
          />

          {/* Answer */}
          <section className="response-section">
            <h3>Plan Summary</h3>
            <p className="response-text">{data.answer}</p>
          </section>

          {/* Total credits callout */}
          {data.total_credits != null && (
            <div className="credit-callout" aria-label={`Total credits: ${data.total_credits}`}>
              <span className="credit-number">{data.total_credits}</span>
              <span className="credit-label">Total Credits</span>
            </div>
          )}

          {/* Course plan list */}
          {data.plan && data.plan.length > 0 && (
            <section className="response-section">
              <h3>Recommended Courses</h3>
              <ul className="course-plan-list" role="list">
                {data.plan.map((course) => (
                  <li key={course.course_code} className="course-plan-item" role="listitem">
                    <div className="course-plan-header">
                      <span className="course-code">{course.course_code}</span>
                      <span className="course-name">{course.course_name}</span>
                      <span className="badge badge--blue">{course.credits} cr</span>
                    </div>
                    <p className="course-justification">{course.justification}</p>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Risks */}
          {data.risks && data.risks.length > 0 && (
            <section className="response-section">
              <h3>Risks</h3>
              <ul className="tag-list">
                {data.risks.map((r, i) => (
                  <li key={i} className="tag tag--red">{r}</li>
                ))}
              </ul>
            </section>
          )}

          {/* Assumptions */}
          {data.assumptions && data.assumptions.length > 0 && (
            <section className="response-section">
              <h3>Assumptions</h3>
              <ul className="tag-list">
                {data.assumptions.map((a, i) => (
                  <li key={i} className="tag tag--warning">{a}</li>
                ))}
              </ul>
            </section>
          )}

          <CitationSection citations={data.citations ?? []} />

          <footer className="trace-footer" aria-label="Trace ID">
            trace: {data.trace_id}
          </footer>
        </article>
      )}
    </div>
  );
}
