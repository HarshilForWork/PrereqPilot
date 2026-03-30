interface ResponseFallbackProps {
  message?: string;
  onRetry?: () => void;
  type?: 'empty' | 'error' | 'loading';
}

export default function ResponseFallback({
  message,
  onRetry,
  type = 'empty',
}: ResponseFallbackProps) {
  if (type === 'loading') {
    return (
      <div className="fallback fallback--loading" role="status" aria-live="polite" aria-label="Loading response">
        <div className="spinner" aria-hidden="true" />
        <p>Thinking…</p>
      </div>
    );
  }

  if (type === 'error') {
    return (
      <div className="fallback fallback--error" role="alert" aria-live="assertive">
        <span className="fallback-icon">⚠️</span>
        <p className="fallback-msg">{message ?? 'Something went wrong.'}</p>
        {onRetry && (
          <button className="btn btn-outline" onClick={onRetry} aria-label="Retry request">
            Retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="fallback fallback--empty" aria-label="No results yet">
      <span className="fallback-icon">🎓</span>
      <p className="fallback-msg">{message ?? 'Submit a query to get started.'}</p>
    </div>
  );
}
