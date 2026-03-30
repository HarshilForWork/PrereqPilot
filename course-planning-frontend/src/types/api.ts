// ─── Shared ────────────────────────────────────────────────────────────────

export interface CitationModel {
  chunk_id: string;
  document_name: string;
  section_heading: string;
}

export type ModeStatus = 'idle' | 'loading' | 'success' | 'error';

// ─── Prereq ─────────────────────────────────────────────────────────────────

export interface StudentProfilePrereq {
  completed_courses: string[];
  grades?: Record<string, string>;
  catalog_year?: string;
}

export interface PrereqRequest {
  question: string;
  student_profile: StudentProfilePrereq;
}

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

export interface StudentProfilePlan {
  completed_courses: string[];
  grades?: Record<string, string>;
  target_major?: string;
  target_term?: string;
  max_credits?: number;
}

export interface PlanRequest {
  student_profile: StudentProfilePlan;
  question: string;
}

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

export interface AskRequest {
  question: string;
  student_profile?: Partial<StudentProfilePrereq>;
}

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
