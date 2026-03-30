import type {
  PrereqRequest,
  PrereqResponse,
  PlanRequest,
  PlanResponse,
  AskRequest,
  AskResponse,
} from '../types/api';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

async function apiPost<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let message = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const err = await res.json();
      message = err?.detail ?? err?.message ?? message;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(message);
  }
  return res.json() as Promise<TRes>;
}

export const queryPrereq = (req: PrereqRequest): Promise<PrereqResponse> =>
  apiPost<PrereqRequest, PrereqResponse>('/query/prereq', req);

export const queryPlan = (req: PlanRequest): Promise<PlanResponse> =>
  apiPost<PlanRequest, PlanResponse>('/query/plan', req);

export const queryAsk = (req: AskRequest): Promise<AskResponse> =>
  apiPost<AskRequest, AskResponse>('/query/ask', req);
