import { useAppContext } from '../../context/AppContext';

type Mode = 'prereq' | 'plan' | 'ask';

const TABS: { id: Mode; label: string; icon: string; desc: string }[] = [
  { id: 'prereq', label: 'Check Prerequisites', icon: '🔍', desc: 'Am I eligible for a course?' },
  { id: 'plan', label: 'Course Plan', icon: '📋', desc: 'Build my next term schedule' },
  { id: 'ask', label: 'Ask Anything', icon: '💡', desc: 'General course questions' },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { state, dispatch } = useAppContext();

  return (
    <div className="app-layout">
      {/* ── Header ── */}
      <header className="app-header" role="banner">
        <div className="header-inner">
          <div className="header-brand">
            <span className="header-logo" aria-hidden="true">🎓</span>
            <div>
              <h1 className="header-title">Course Planning Assistant</h1>
              <p className="header-subtitle">Agentic RAG — Powered by your academic catalog</p>
            </div>
          </div>
        </div>
      </header>

      {/* ── Tab Navigation ── */}
      <nav className="tab-nav" role="navigation" aria-label="Query modes">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn ${state.activeMode === tab.id ? 'tab-btn--active' : ''}`}
            onClick={() => dispatch({ type: 'SET_MODE', mode: tab.id })}
            aria-selected={state.activeMode === tab.id}
            aria-controls={`panel-${tab.id}`}
            role="tab"
            id={`tab-${tab.id}`}
          >
            <span className="tab-icon" aria-hidden="true">{tab.icon}</span>
            <span className="tab-content">
              <span className="tab-label">{tab.label}</span>
              <span className="tab-desc">{tab.desc}</span>
            </span>
          </button>
        ))}
      </nav>

      {/* ── Main content ── */}
      <main className="app-main" id={`panel-${state.activeMode}`} role="tabpanel" aria-labelledby={`tab-${state.activeMode}`}>
        {children}
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer" role="contentinfo">
        <p>Course Planning Assistant · Agentic RAG Backend</p>
      </footer>
    </div>
  );
}
