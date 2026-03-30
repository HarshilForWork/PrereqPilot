import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { queryAsk } from '../../services/api';
import CitationSection from '../shared/CitationSection';
import ClarifyingQuestionsUI from '../shared/ClarifyingQuestionsUI';
import ResponseFallback from '../shared/ResponseFallback';

export default function AskView() {
  const { state, dispatch } = useAppContext();
  const { status, data, error } = state.ask;

  const [question, setQuestion] = useState('');

  const buildRequest = (extra?: string) => ({
    question: extra ? `${question}\n\nAdditional context: ${extra}` : question,
  });

  const submit = async (extra?: string) => {
    if (!question.trim()) return;
    dispatch({ type: 'FETCH_START', mode: 'ask' });
    try {
      const res = await queryAsk(buildRequest(extra));
      dispatch({ type: 'FETCH_SUCCESS_ASK', data: res });
    } catch (e: unknown) {
      dispatch({ type: 'FETCH_ERROR', mode: 'ask', error: (e as Error).message });
    }
  };

  const retry = () => submit();

  return (
    <div className="mode-view">
      {/* ── Form ── */}
      <form className="query-form" onSubmit={(e) => { e.preventDefault(); submit(); }} aria-label="Ask a question form">
        <div className="form-group">
          <label htmlFor="ask-question" className="form-label">Ask anything about your courses</label>
          <textarea
            id="ask-question"
            className="form-textarea"
            rows={3}
            placeholder="e.g. What is the academic policy for course withdrawal?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={status === 'loading'}
            required
            aria-required="true"
          />
        </div>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={status === 'loading' || !question.trim()}
          aria-busy={status === 'loading'}
        >
          {status === 'loading' ? 'Asking…' : 'Ask'}
        </button>
      </form>

      {/* ── States ── */}
      {status === 'loading' && <ResponseFallback type="loading" />}
      {status === 'error' && <ResponseFallback type="error" message={error ?? undefined} onRetry={retry} />}
      {status === 'idle' && !data && <ResponseFallback type="empty" message="Ask any course-related question." />}

      {/* ── Success ── */}
      {data && (
        <article className="response-card" aria-label="Ask mode result">
          {/* Clarifying questions first */}
          <ClarifyingQuestionsUI
            questions={data.clarifying_questions ?? []}
          />

          {/* Abstained fallback */}
          {data.abstained ? (
            <div className="abstain-box" role="alert">
              <span className="abstain-icon">🤔</span>
              <div>
                <p className="abstain-title">Information not available</p>
                <p className="abstain-body">
                  {data.answer ||
                    "I don't have that information in the provided catalog/policies. Please check your academic advisor or the department's official website."}
                </p>
              </div>
            </div>
          ) : (
            <section className="response-section">
              <h3>Answer</h3>
              <p className="response-text">{data.answer}</p>
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
