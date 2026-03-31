import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { queryPrereq } from '../../services/api';
import { Button } from '@/components/animate-ui/components/buttons/button';
import CitationSection from '../shared/CitationSection';
import ClarifyingQuestionsUI from '../shared/ClarifyingQuestionsUI';
import ResponseFallback from '../shared/ResponseFallback';
import type { PrereqResponse } from '../../types/api';

const DECISION_CLASS: Record<string, string> = {
  Eligible: 'badge badge--green',
  'Not eligible': 'badge badge--red',
  'Need more info': 'badge badge--yellow',
};

export default function PrereqView() {
  const { state, dispatch } = useAppContext();
  const { status, data, error } = state.prereq;

  const [question, setQuestion] = useState('');
  const [courses, setCourses] = useState('');

  const buildRequest = (extra?: string) => ({
    question: extra ? `${question}\n\nAdditional context: ${extra}` : question,
    student_profile: {
      completed_courses: courses.split(',').map((c) => c.trim()).filter(Boolean),
    },
  });

  const submit = async (extra?: string) => {
    if (!question.trim()) return;
    dispatch({ type: 'FETCH_START', mode: 'prereq' });
    try {
      const res = await queryPrereq(buildRequest(extra));
      dispatch({ type: 'FETCH_SUCCESS_PREREQ', data: res });
    } catch (e: unknown) {
      dispatch({ type: 'FETCH_ERROR', mode: 'prereq', error: (e as Error).message });
    }
  };

  const retry = () => submit();

  return (
    <div className="mode-view">
      {/* ── Query Form ── */}
      <form className="query-form" onSubmit={(e) => { e.preventDefault(); submit(); }} aria-label="Prerequisite check form">
        <div className="form-group">
          <label htmlFor="prereq-question" className="form-label">Your question</label>
          <textarea
            id="prereq-question"
            className="form-textarea"
            rows={3}
            placeholder="e.g. Can I take CS301 if I completed CS101 and CS201?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={status === 'loading'}
            required
            aria-required="true"
          />
        </div>
        <div className="form-group">
          <label htmlFor="prereq-courses" className="form-label">
            Completed courses <span className="form-hint">(comma-separated)</span>
          </label>
          <input
            id="prereq-courses"
            type="text"
            className="form-input"
            placeholder="CS101, CS201, MATH120"
            value={courses}
            onChange={(e) => setCourses(e.target.value)}
            disabled={status === 'loading'}
          />
        </div>
        <Button
          type="submit"
          className="btn btn-primary"
          disabled={status === 'loading' || !question.trim()}
          aria-busy={status === 'loading'}
        >
          {status === 'loading' ? 'Checking…' : 'Check Prerequisites'}
        </Button>
      </form>

      {/* ── States ── */}
      {status === 'loading' && <ResponseFallback type="loading" />}
      {status === 'error' && <ResponseFallback type="error" message={error ?? undefined} onRetry={retry} />}
      {status === 'idle' && !data && <ResponseFallback type="empty" />}

      {/* ── Success ── */}
      {data && (
        <article className="response-card" aria-label="Prerequisite check result">
          {/* Clarifying questions first — always prominent */}
          <ClarifyingQuestionsUI
            questions={data.clarifying_questions ?? []}
          />

          {/* Decision badge */}
          <div className="response-header">
            <span className={DECISION_CLASS[data.decision] ?? 'badge badge--yellow'} aria-label="Decision">
              {data.decision}
            </span>
          </div>

          {/* Main answer */}
          <section className="response-section">
            <h3>Answer</h3>
            <p className="response-text">{data.answer}</p>
          </section>

          {/* Evidence */}
          {data.evidence && (
            <section className="response-section">
              <h3>Evidence</h3>
              <p className="response-text">{data.evidence}</p>
            </section>
          )}

          {/* Next step */}
          {data.next_step && (
            <section className="response-section next-step-box">
              <h3>Next Step</h3>
              <p>{data.next_step}</p>
            </section>
          )}

          {/* Graph result */}
          {data.graph_result && (
            <section className="response-section">
              <h3>Prerequisite Chain</h3>
              <GraphResultDisplay result={data} />
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

function GraphResultDisplay({ result }: { result: PrereqResponse }) {
  const gr = result.graph_result;
  return (
    <div className="graph-display">
      <div className={`eligibility-indicator ${gr.eligible ? 'eligible' : 'not-eligible'}`}>
        {gr.eligible ? 'ELIGIBLE BASED ON CATALOG' : 'INELIGIBLE BASED ON CATALOG'}
      </div>
      {gr.prereq_path.length > 0 && (
        <div className="prereq-path">
          <span className="path-label">Path: </span>
          {gr.prereq_path.map((code, i) => (
            <span key={code} className="path-item">
              <span className="course-code">{code}</span>
              {i < gr.prereq_path.length - 1 && <span className="path-arrow"> → </span>}
            </span>
          ))}
        </div>
      )}
      {gr.missing_prereqs.length > 0 && (
        <div className="missing-prereqs">
          <span className="path-label">Missing: </span>
          {gr.missing_prereqs.map((code) => (
            <span key={code} className="course-code course-code--missing">{code}</span>
          ))}
        </div>
      )}
      {gr.either_or_options.length > 0 && (
        <div className="either-or">
          <span className="path-label">Either / Or options:</span>
          {gr.either_or_options.map((group, i) => (
            <span key={i} className="either-or-group">
              {group.map((c, j) => (
                <span key={c}>
                  <span className="course-code">{c}</span>
                  {j < group.length - 1 && <span className="path-arrow"> or </span>}
                </span>
              ))}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
