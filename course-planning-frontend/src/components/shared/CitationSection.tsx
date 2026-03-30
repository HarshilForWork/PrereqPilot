import type { CitationModel } from '../../types/api';

interface CitationSectionProps {
  citations: CitationModel[];
}

export default function CitationSection({ citations }: CitationSectionProps) {
  if (!citations || citations.length === 0) return null;

  return (
    <section className="citation-section" aria-label="Citations">
      <h3 className="citation-heading">
        <span className="citation-icon">📚</span> Sources
      </h3>
      <ul className="citation-list" role="list">
        {citations.map((c) => (
          <li key={c.chunk_id} className="citation-item" role="listitem">
            <span className="citation-doc">{c.document_name}</span>
            {c.section_heading && (
              <>
                <span className="citation-sep"> › </span>
                <span className="citation-section-name">{c.section_heading}</span>
              </>
            )}
            <span className="citation-chunk" aria-label="Chunk ID">
              #{c.chunk_id}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
