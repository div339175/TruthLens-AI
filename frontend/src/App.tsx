import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { VerificationPage } from './pages/VerificationPage';
import { HistoryPage } from './pages/HistoryPage';
import { ProfessorDemoPage } from './pages/ProfessorDemoPage';
import { EvaluationHubPage } from './pages/EvaluationHubPage';
import { ArchitecturePage } from './pages/ArchitecturePage';
import { DatasetPage } from './pages/DatasetPage';
import { checkHealth } from './services/api';
import { HealthInfo } from './types';
import { ShieldCheck } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'verify' | 'history' | 'demo' | 'evaluation' | 'architecture' | 'dataset'>('verify');
  const [health, setHealth] = useState<HealthInfo | null>(null);

  useEffect(() => {
    // Check initial health
    checkHealth()
      .then(setHealth)
      .catch((err) => console.warn('Backend not responding yet:', err));

    const interval = setInterval(() => {
      checkHealth().then(setHealth).catch(() => setHealth(null));
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      {/* Main Tab Content */}
      <main className="flex-1 pb-16">
        {activeTab === 'verify' && <VerificationPage />}
        {activeTab === 'history' && <HistoryPage />}
        {activeTab === 'demo' && <ProfessorDemoPage />}
        {activeTab === 'evaluation' && <EvaluationHubPage />}
        {activeTab === 'architecture' && <ArchitecturePage />}
        {activeTab === 'dataset' && <DatasetPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-sky-400" />
            <span className="font-semibold text-slate-400">TruthLens AI</span>
            <span>— Multimodal Deepfake Verification System</span>
          </div>

          <div className="flex items-center space-x-4">
            <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              University NLP Project • Milestone 1
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
