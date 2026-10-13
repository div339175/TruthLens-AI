import React from 'react';
import { CheckCircle2, AlertTriangle, HelpCircle, Shield, Cpu, Clock } from 'lucide-react';
import { VerificationReport } from '../types';

interface PredictionCardProps {
  report: VerificationReport;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ report }) => {
  const isReal = report.prediction === 'REAL';
  const isFake = report.prediction === 'FAKE';

  const badgeColor = isReal
    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
    : isFake
    ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
    : 'bg-amber-500/10 text-amber-400 border-amber-500/30';

  const Icon = isReal ? CheckCircle2 : isFake ? AlertTriangle : HelpCircle;

  return (
    <div className="glass-panel rounded-2xl p-6 sm:p-8 shadow-xl border border-slate-800">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <div
            className={`w-14 h-14 rounded-2xl flex items-center justify-center border ${
              isReal
                ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400'
                : isFake
                ? 'bg-rose-500/20 border-rose-500/30 text-rose-400'
                : 'bg-amber-500/20 border-amber-500/30 text-amber-400'
            }`}
          >
            <Icon className="w-8 h-8" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Verification Verdict</span>
              <span className={`px-2 py-0.5 rounded text-[11px] font-bold border ${badgeColor}`}>
                {report.execution_mode.toUpperCase()}
              </span>
            </div>
            <h2 className="text-3xl font-extrabold text-white tracking-tight">{report.prediction}</h2>
          </div>
        </div>

        {/* Confidence Gauge */}
        <div className="text-right">
          <span className="text-xs text-slate-400 font-medium">Model Confidence</span>
          <div className="text-3xl font-extrabold text-sky-400">
            {(report.confidence * 100).toFixed(1)}%
          </div>
          <span className="text-[11px] text-slate-500">
            {report.uncertain ? 'Threshold (65%) not met' : 'Calibrated Softmax'}
          </span>
        </div>
      </div>

      {/* Probability Bars */}
      <div className="mt-6 space-y-3">
        <div className="flex justify-between text-xs font-medium text-slate-300">
          <span>Modality Evidence Distribution</span>
          <div className="space-x-4">
            <span className="text-emerald-400">Authentic (Real): {(report.real_probability * 100).toFixed(1)}%</span>
            <span className="text-rose-400">Manipulated (Fake): {(report.fake_probability * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden flex">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
            style={{ width: `${Math.max(report.real_probability * 100, 2)}%` }}
            title={`Real: ${(report.real_probability * 100).toFixed(1)}%`}
          />
          <div
            className="h-full bg-gradient-to-r from-rose-500 to-rose-400 transition-all duration-500"
            style={{ width: `${Math.max(report.fake_probability * 100, 2)}%` }}
            title={`Fake: ${(report.fake_probability * 100).toFixed(1)}%`}
          />
        </div>
      </div>

      {/* Meta Grid */}
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <span className="text-slate-400 block mb-1">Manipulation Category</span>
          <span className="text-white font-medium">{report.manipulation_type}</span>
        </div>
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <span className="text-slate-400 block mb-1">Modalities Analyzed</span>
          <div className="flex gap-1.5 flex-wrap">
            {report.modalities_used.map((mod) => (
              <span key={mod} className="px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-medium">
                {mod.toUpperCase()}
              </span>
            ))}
          </div>
        </div>
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <span className="text-slate-400 block mb-1">Architecture & Fusion</span>
          <span className="text-white font-medium flex items-center space-x-1">
            <Cpu className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <span className="truncate">{report.fusion_type} ({report.device_used})</span>
          </span>
        </div>
      </div>

      {/* Human-readable Explanation */}
      <div className="mt-6 p-4 rounded-xl bg-slate-900/80 border border-slate-800">
        <div className="flex items-center space-x-2 mb-2">
          <Shield className="w-4 h-4 text-sky-400" />
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Synthesized Verification Report
          </h4>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">{report.explanation}</p>
        <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-500">
          <span>Model: {report.model_version}</span>
          <span className="flex items-center space-x-1">
            <Clock className="w-3 h-3" />
            <span>{new Date(report.timestamp).toLocaleTimeString()} UTC</span>
          </span>
        </div>
      </div>
    </div>
  );
};
