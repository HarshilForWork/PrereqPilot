interface ClarifyingQuestionsUIProps {
  questions: string[];
}

export default function ClarifyingQuestionsUI({
  questions,
}: ClarifyingQuestionsUIProps) {
  if (!questions || questions.length === 0) return null;

  return (
    <section
      className="clarify-section"
      aria-label="Information needed"
      aria-live="polite"
    >
      <div className="clarify-header">
        <h3 className="clarify-title">Refinement Requested</h3>
      </div>
      <p className="clarify-summary">
        To provide a more accurate response, please include the following details in your query above:
      </p>

      <ul className="clarify-list" role="list">
        {questions.map((q, i) => (
          <li key={i} className="clarify-question" role="listitem">
            <span className="clarify-q-num">•</span> {q}
          </li>
        ))}
      </ul>
    </section>
  );
}
