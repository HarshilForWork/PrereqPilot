import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { queryPlan } from '../../services/api';
import { Button } from '@/components/animate-ui/components/buttons/button';
import CitationSection from '../shared/CitationSection';
import ClarifyingQuestionsUI from '../shared/ClarifyingQuestionsUI';
import ResponseFallback from '../shared/ResponseFallback';

export default function PlanView() {
  const { state, dispatch } = useAppContext();
  const { status, data, error } = state.plan;

  const [question, setQuestion] = useState('');

  const submit = async (extraContext?: string) => {
    if (!question.trim()) return;
    dispatch({ type: 'FETCH_START', mode: 'plan' });
    try {
      const queryText = extraContext 
        ? `${question}\n\nAdditional context: ${extraContext}` 
        : question;
      const res = await queryPlan({ query: queryText });
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
            rows={3}
            placeholder="e.g. Plan for Fall 2026 for Computer Science major after CS101"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={status === 'loading'}
            required
            aria-required="true"
          />
        </div>
        <Button
          type="submit"
          className="btn btn-primary"
          disabled={status === 'loading' || !question.trim()}
          aria-busy={status === 'loading'}
        >
          {status === 'loading' ? 'Planning…' : 'Generate Course Plan'}
        </Button>
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
