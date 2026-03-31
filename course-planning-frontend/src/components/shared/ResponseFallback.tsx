import { Button } from '@/components/animate-ui/components/buttons/button';

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
      <div className="premium-loader" role="status" aria-live="polite" aria-label="Loading response">
        <div className="loader-container">
          <div className="loader-ring" />
          <div className="loader-ring" />
          <div className="loader-ring" />
        </div>
        <p className="loader-text">Analyzing Academic Data</p>
      </div>
    );
  }

  if (type === 'error') {
    return (
      <div className="fallback fallback--error" role="alert" aria-live="assertive">
        <p className="fallback-msg" style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--red)' }}>
           System Error
        </p>
        <p className="fallback-msg">{message ?? 'Something went wrong.'}</p>
        {onRetry && (
          <Button variant="outline" className="btn btn-outline" onClick={onRetry} aria-label="Retry request">
            Retry
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className="fallback fallback--empty" aria-label="No results yet">
      <div className="header-mark" style={{ opacity: 0.3, filter: 'grayscale(1)', marginBottom: '1rem' }}>P</div>
      <p className="fallback-msg">{message ?? 'Enter your course query to begin analysis.'}</p>
    </div>
  );
}
