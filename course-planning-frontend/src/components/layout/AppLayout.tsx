import { useAppContext } from '../../context/AppContext';
import InfoIcon from '../shared/InfoIcon';
import { GravityStarsBackground } from '@/components/animate-ui/components/backgrounds/gravity-stars';
import { Button } from '@/components/animate-ui/components/buttons/button';

type Mode = 'prereq' | 'plan' | 'ask';

const TABS: { id: Mode; label: string; icon: string; desc: string }[] = [
  { id: 'prereq', label: 'Prerequisites', icon: 'PRE', desc: 'Eligibility check' },
  { id: 'plan', label: 'Course Plan', icon: 'PLN', desc: 'Schedule builder' },
  { id: 'ask', label: 'Consultant', icon: 'ASK', desc: 'General help' },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { state, dispatch } = useAppContext();

  return (
    <div className="app-layout">
      <GravityStarsBackground 
        starsCount={150} 
        starsSize={1.5} 
        starsOpacity={0.6} 
        glowIntensity={30}
        glowAnimation="spring"
        movementSpeed={0.5}
        mouseInfluence={180}
        mouseGravity="attract"
        gravityStrength={120}
        className="fixed inset-0 z-[-1] pointer-events-auto bg-black"
      />
      {/* ── Header ── */}
      <header className="app-header" role="banner">
        <div className="header-inner">
          <div className="header-brand">
            <div className="header-mark">P</div>
            <div>
              <h1 className="header-title">PrereqPilot</h1>
              <p className="header-subtitle">Intelligent Academic Catalog Assistant</p>
            </div>
          </div>
          <InfoIcon />
        </div>
      </header>

      {/* ── Tab Navigation ── */}
      <nav className="tab-nav" role="navigation" aria-label="Query modes">
        {TABS.map((tab) => (
          <Button
            key={tab.id}
            variant="ghost"
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
          </Button>
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
