import React, { useState } from 'react';
import { Layout, type NavTab } from './components/Layout';
import { OverviewPage } from './pages/OverviewPage';
import { TrafficAnalysisPage } from './pages/TrafficAnalysisPage';
import { SafetyViolationsPage } from './pages/SafetyViolationsPage';
import { PlateScannerPage } from './pages/PlateScannerPage';
import { RecordsReportsPage } from './pages/RecordsReportsPage';
import { DiagnosticsPage } from './pages/DiagnosticsPage';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');

  return (
    <Layout currentTab={currentTab} onTabChange={setCurrentTab}>
      {currentTab === 'overview' && <OverviewPage onNavigate={setCurrentTab} />}
      {currentTab === 'traffic' && <TrafficAnalysisPage />}
      {currentTab === 'violations' && <SafetyViolationsPage />}
      {currentTab === 'plates' && <PlateScannerPage />}
      {currentTab === 'records' && <RecordsReportsPage />}
      {currentTab === 'diagnostics' && <DiagnosticsPage />}
    </Layout>
  );
};

export default App;
