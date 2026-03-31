// ─── Shared ────────────────────────────────────────────────────────────────

export interface CitationModel {
  chunk_id: string;
  document_name: string;
  section_heading: string;
}

export type ModeStatus = 'idle' | 'loading' | 'success' | 'error';

// ─── Simplified Request Pattern ─────────────────────────────────────────────
// The backend uses an IntakeAgent to extract profile details (major, credits, 
// completed courses) directly from the natural language query.

export interface QueryRequest {
  query: string;
}

// ─── Prereq ─────────────────────────────────────────────────────────────────

export interface PrereqRequest extends QueryRequest {}

export interface GraphResult {
  eligible: boolean;
  missing_prereqs: string[];
  prereq_path: string[];
  either_or_options: string[][];
}

export interface PrereqResponse {
  decision: 'Eligible' | 'Not eligible' | 'Need more info';
  answer: string;
  evidence: string;
  citations: CitationModel[];
  next_step: string;
  clarifying_questions: string[];
  graph_result: GraphResult;
  assumptions: string[];
  trace_id: string;
}

// ─── Plan ────────────────────────────────────────────────────────────────────

export interface PlanRequest extends QueryRequest {}

export interface PlannedCourse {
  course_code: string;
  course_name: string;
  credits: number;
  justification: string;
  citation?: CitationModel;
}

export interface PlanResponse {
  answer: string;
  plan: PlannedCourse[];
  total_credits: number;
  citations: CitationModel[];
  clarifying_questions: string[];
  risks: string[];
  assumptions: string[];
  trace_id: string;
}

// ─── Ask ─────────────────────────────────────────────────────────────────────

export interface AskRequest extends QueryRequest {}

export interface AskResponse {
  answer: string;
  citations: CitationModel[];
  clarifying_questions: string[];
  abstained: boolean;
  trace_id: string;
}

// ─── Per-mode state slices ────────────────────────────────────────────────────

export interface ModeState<TData> {
  status: ModeStatus;
  data: TData | null;
  error: string | null;
}
