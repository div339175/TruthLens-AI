import React from 'react';
import { ShieldCheck, Cpu, Database, Activity, History, BarChart3, Layers } from 'lucide-react';
import { HealthInfo } from '../types';

interface NavbarProps {
  activeTab: 'verify' | 'history' | 'benchmark' | 'evaluation' | 'architecture' | 'dataset';
  setActiveTab: (tab: 'verify' | 'history' | 'benchmark' | 'evaluation' | 'architecture' | 'dataset') => void;
  health: HealthInfo | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, health }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('verify')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-white tracking-tight">TruthLens AI</span>
                <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">
                  MVP 1.0
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">Multimodal NLP for Deepfake Verification</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            <button
              onClick={() => setActiveTab('verify')}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'verify'
                  ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              Verify Content
            </button>
            <button
              onClick={() => setActiveTab('benchmark')}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
                activeTab === 'benchmark'
                  ? 'bg-indigo-500/15 text-indigo-400 border border-indigo-500/30 font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Layers className="w-4 h-4 text-indigo-400" />
              <span>External Benchmark</span>
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
                activeTab === 'history'
                  ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <History className="w-4 h-4" />
              <span>History</span>
            </button>
            <button
              onClick={() => setActiveTab('evaluation')}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
                activeTab === 'evaluation'
                  ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Evaluation Hub</span>
            </button>
            <button
              onClick={() => setActiveTab('architecture')}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
                activeTab === 'architecture'
                  ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Cpu className="w-4 h-4" />
              <span className="hidden md:inline">Architecture</span>
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
                activeTab === 'dataset'
                  ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Database className="w-4 h-4" />
              <span className="hidden md:inline">Dataset</span>
            </button>
          </nav>

          {/* System Hardware & Status Pill */}
          <div className="hidden lg:flex items-center space-x-2 text-xs">
            {health ? (
              <>
                <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span>Pipeline: {health.execution_mode.toUpperCase()}</span>
                </div>
                <div className="flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700 text-slate-300">
                  <Cpu className="w-3.5 h-3.5 text-sky-400" />
                  <span>Device: {health.device.toUpperCase()}</span>
                </div>
              </>
            ) : (
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-400">
                <Activity className="w-3.5 h-3.5 animate-spin" />
                <span>Connecting...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
