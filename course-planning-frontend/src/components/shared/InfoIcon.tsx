import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/animate-ui/components/buttons/button';
import './InfoIcon.css';

const ENDPOINTS = [
  {
    title: 'Check Prerequisites',
    endpoint: '/query/prereq',
    description: 'Calculates eligibility based on your completed courses, grades, and corequisite rules using a deterministic DAG and semantic catalog retrieval.',
    use: 'Best for checking if you can take a specific course right now.'
  },
  {
    title: 'Course Plan',
    endpoint: '/query/plan',
    description: 'Generates a recommended course list for a target term based on your history, major requirements, and credit limits.',
    use: 'Best for long-term planning and degree progress.'
  },
  {
    title: 'Ask Anything',
    endpoint: '/query/ask',
    description: 'General-purpose RAG-powered Q&A for academic policies, campus resources, and miscellaneous catalog information.',
    use: 'Best for specific questions like "What is the grading policy?"'
  }
];

export default function InfoIcon() {
  const [isOpen, setIsOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="info-icon-container" ref={popoverRef}>
      <Button 
        variant="ghost"
        className={`info-icon-btn ${isOpen ? 'info-icon-btn--active' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="API Information"
        title="View Endpoint Information"
      >
        i
      </Button>

      {isOpen && (
        <div className="info-popover">
          <div className="info-popover-header">
            <h3>Backend Capabilities</h3>
            <p>This assistant uses three specialized agentic endpoints:</p>
          </div>
          <div className="info-popover-content">
            {ENDPOINTS.map((ep) => (
              <div key={ep.endpoint} className="info-ep-card">
                <div className="info-ep-title">
                  <span className="info-ep-label">{ep.title}</span>
                  <code className="info-ep-route">{ep.endpoint}</code>
                </div>
                <p className="info-ep-desc">{ep.description}</p>
                <div className="info-ep-use">
                  <strong>Use Case:</strong> {ep.use}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
