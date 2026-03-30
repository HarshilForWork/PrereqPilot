import { useAppContext } from '../../context/AppContext';
import PrereqView from '../modes/PrereqView';
import PlanView from '../modes/PlanView';
import AskView from '../modes/AskView';

/**
 * ModeTabContainer keeps all three mode views mounted at all times
 * (display:none toggling via CSS classes) so state is preserved when
 * the user switches tabs.
 */
export default function ModeTabContainer() {
  const { state } = useAppContext();
  const active = state.activeMode;

  return (
    <>
      <div className={active === 'prereq' ? 'tab-panel tab-panel--active' : 'tab-panel'} aria-hidden={active !== 'prereq'}>
        <PrereqView />
      </div>
      <div className={active === 'plan' ? 'tab-panel tab-panel--active' : 'tab-panel'} aria-hidden={active !== 'plan'}>
        <PlanView />
      </div>
      <div className={active === 'ask' ? 'tab-panel tab-panel--active' : 'tab-panel'} aria-hidden={active !== 'ask'}>
        <AskView />
      </div>
    </>
  );
}
