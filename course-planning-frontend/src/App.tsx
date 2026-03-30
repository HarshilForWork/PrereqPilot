import { AppProvider } from './context/AppContext';
import AppLayout from './components/layout/AppLayout';
import ModeTabContainer from './components/layout/ModeTabContainer';
import './index.css';

export default function App() {
  return (
    <AppProvider>
      <AppLayout>
        <ModeTabContainer />
      </AppLayout>
    </AppProvider>
  );
}
