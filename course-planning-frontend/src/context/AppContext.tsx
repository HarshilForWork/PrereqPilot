import React, {
  createContext,
  useContext,
  useReducer,
  type ReactNode,
} from 'react';
import type {
  ModeState,
  PrereqResponse,
  PlanResponse,
  AskResponse,
} from '../types/api';

// ─── State Shape ─────────────────────────────────────────────────────────────

interface AppState {
  prereq: ModeState<PrereqResponse>;
  plan: ModeState<PlanResponse>;
  ask: ModeState<AskResponse>;
  activeMode: 'prereq' | 'plan' | 'ask';
}

const empty = <T,>(): ModeState<T> => ({
  status: 'idle',
  data: null,
  error: null,
});

const initialState: AppState = {
  prereq: empty<PrereqResponse>(),
  plan: empty<PlanResponse>(),
  ask: empty<AskResponse>(),
  activeMode: 'prereq',
};

// ─── Actions ─────────────────────────────────────────────────────────────────

type Mode = 'prereq' | 'plan' | 'ask';

type Action =
  | { type: 'SET_MODE'; mode: Mode }
  | { type: 'FETCH_START'; mode: Mode }
  | { type: 'FETCH_SUCCESS_PREREQ'; data: PrereqResponse }
  | { type: 'FETCH_SUCCESS_PLAN'; data: PlanResponse }
  | { type: 'FETCH_SUCCESS_ASK'; data: AskResponse }
  | { type: 'FETCH_ERROR'; mode: Mode; error: string };

// ─── Reducer ─────────────────────────────────────────────────────────────────

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_MODE':
      return { ...state, activeMode: action.mode };

    case 'FETCH_START':
      return {
        ...state,
        [action.mode]: { ...state[action.mode], status: 'loading', error: null },
      };

    case 'FETCH_SUCCESS_PREREQ':
      return {
        ...state,
        prereq: { status: 'success', data: action.data, error: null },
      };

    case 'FETCH_SUCCESS_PLAN':
      return {
        ...state,
        plan: { status: 'success', data: action.data, error: null },
      };

    case 'FETCH_SUCCESS_ASK':
      return {
        ...state,
        ask: { status: 'success', data: action.data, error: null },
      };

    case 'FETCH_ERROR':
      return {
        ...state,
        [action.mode]: { ...state[action.mode], status: 'error', error: action.error },
      };

    default:
      return state;
  }
}

// ─── Context ─────────────────────────────────────────────────────────────────

interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppContext must be used inside AppProvider');
  return ctx;
}
