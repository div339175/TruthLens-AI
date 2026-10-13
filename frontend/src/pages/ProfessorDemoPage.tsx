import React, { useState } from 'react';
import { GraduationCap, CheckCircle2, ShieldCheck, ArrowRight, Info } from 'lucide-react';
import { verifyMultimodal } from '../services/api';
import { VerificationReport } from '../types';
import { PredictionCard } from '../components/PredictionCard';
import { EvidenceSection } from '../components/EvidenceSection';

const DEMO_CASES = [
  {
    id: 'case_real',
    title: 'Authentic Official Briefing',
    category: 'REAL',
    text: 'Verified official press conference on sustainable environmental policy and industrial regulations.',
    description: 'Authentic news broadcast text with consistent factual language and no synthetic artifacts.',
  },
  {
    id: 'case_fake',
    title: 'AI Deepfake Synthetic Speech',
    category: 'FAKE',
    text: 'Leaked AI-generated speech showing high-profile celebrity endorsing unverified investment token.',
    description: 'Suspicious textual markers and generative synthesis indicators.',
  },
  {
    id: 'case_uncertain',
    title: 'Ambiguous / Low Signal Post',
    category: 'UNCERTAIN',
    text: 'Unconfirmed rumor regarding sudden executive departures at regional logistics hub.',
    description: 'Demonstrates uncertainty policy when confidence does not exceed 65% threshold.',
  },
];

export const ProfessorDemoPage: React.FC = () => {
  const [selectedCase, setSelectedCase] = useState<typeof DEMO_CASES[0]>(DEMO_CASES[0]);
  const [customText, setCustomText] = useState<string>(DEMO_CASES[0].text);
  const [loading, setLoading] = useState<boolean>(false);
  const [report, setReport] = useState<VerificationReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSelectCase = (c: typeof DEMO_CASES[0]) => {
    setSelectedCase(c);
    setCustomText(c.text);
    setReport(null);
    setError(null);
  };

  const handleRunDemoVerification = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await verifyMultimodal(customText);
      setReport(res);
    } catch (err: any) {
      setError(err.message || 'Demo verification error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      {/* University Demo Header */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-indigo-500/30 bg-gradient-to-br from-slate-900/90 via-slate-900/60 to-indigo-950/30 shadow-xl">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 text-xs font-bold uppercase tracking-wider flex items-center space-x-1.5">
                <GraduationCap className="w-4 h-4" />
                <span>University Evaluation Demonstration</span>
              </span>
              <span className="text-xs text-slate-400 font-medium">Milestone 1 Working Prototype</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">TRUTHLENS AI</h1>
            <p className="text-sm text-slate-300 max-w-2xl leading-relaxed">
              Multimodal Deepfake Verification and Explainable Content Verification.
              This system assesses multimodal evidence across text and vision streams to provide explainable reports.
            </p>
          </div>

          <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 text-xs space-y-1">
            <div className="text-slate-400">Current Implementation Stage:</div>
            <div className="font-bold text-sky-400">Text + Image Baseline</div>
            <div className="text-[11px] text-emerald-400 flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Real PyTorch Model Pipeline</span>
            </div>
          </div>
        </div>

        {/* Evaluation Policy Notice */}
        <div className="mt-6 pt-5 border-t border-slate-800 flex items-center space-x-2 text-xs text-slate-400">
          <Info className="w-4 h-4 text-sky-400 shrink-0" />
          <span>
            <strong>Transparency Notice:</strong> All metrics, predictions, token saliency, and confidence scores are computed by actual model passes without hardcoded mock values.
          </span>
        </div>
      </div>

      {/* Preset Test Scenarios */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Select Professor Demonstration Scenario
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {DEMO_CASES.map((c) => {
            const isSelected = selectedCase.id === c.id;
            return (
              <button
                key={c.id}
                type="button"
                onClick={() => handleSelectCase(c)}
                className={`p-4 rounded-xl text-left border transition-all ${
                  isSelected
                    ? 'bg-sky-500/10 border-sky-500/40 shadow-lg shadow-sky-500/10'
                    : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-sm text-white">{c.title}</span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                      c.category === 'REAL'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : c.category === 'FAKE'
                        ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                        : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    }`}
                  >
                    {c.category}
                  </span>
                </div>
                <p className="text-xs text-slate-400 line-clamp-2">{c.description}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Interactive Input Box */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <label className="text-sm font-semibold text-slate-200">
            Selected Content for Evaluation:
          </label>
          <span className="text-xs text-slate-500">{customText.length} characters</span>
        </div>

        <textarea
          value={customText}
          onChange={(e) => setCustomText(e.target.value)}
          rows={4}
          className="w-full rounded-xl bg-slate-900 border border-slate-800 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 p-4 text-sm text-slate-100 resize-none"
        />

        <div className="flex justify-end">
          <button
            type="button"
            onClick={handleRunDemoVerification}
            disabled={loading}
            className="px-6 py-3 rounded-xl text-sm font-bold text-white bg-gradient-to-r from-indigo-500 to-sky-600 hover:from-indigo-400 hover:to-sky-500 shadow-lg shadow-indigo-500/20 flex items-center space-x-2 transition"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Running Pipeline Forward Pass...</span>
              </>
            ) : (
              <>
                <span>Verify Content with TruthLens</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {/* Result Display */}
      {report && (
        <div className="space-y-6 animate-fadeIn">
          <div className="flex items-center space-x-2 text-sm text-slate-400">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Inspection Complete — Real Execution Result</span>
          </div>
          <PredictionCard report={report} />
          <EvidenceSection evidence={report.evidence} />
        </div>
      )}
    </div>
  );
};
