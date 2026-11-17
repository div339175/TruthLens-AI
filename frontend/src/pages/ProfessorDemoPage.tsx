import React, { useState } from 'react';
import { GraduationCap, CheckCircle2, ShieldCheck, ArrowRight, Info, BookOpen } from 'lucide-react';
import { verifyMultimodal } from '../services/api';
import { VerificationReport } from '../types';
import { PredictionCard } from '../components/PredictionCard';
import { EvidenceSection } from '../components/EvidenceSection';

const STANDARD_CASES = [
  {
    id: 'case_real',
    title: 'Authentic Official Briefing',
    category: 'REAL',
    text: 'Ministry of Health releases updated quarterly public health statistics and preventive guidelines.',
    description: 'Authentic news broadcast text with consistent factual language and no synthetic artifacts.',
  },
  {
    id: 'case_fake',
    title: 'AI Deepfake Synthetic Speech',
    category: 'FAKE',
    text: 'Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.',
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

const FIFTEEN_DIFFICULT_CASES = [
  {
    case_id: 'CASE_01',
    category: 'semantic_mismatch',
    title: 'Festival Miscaptioned as Street Riot',
    text: 'Breaking news: violent riots and civil unrest break out in downtown capital, teargas deployed.',
    expected: 'FAKE (Mismatch)',
  },
  {
    case_id: 'CASE_02',
    category: 'semantic_mismatch',
    title: 'Archival Space Launch Claimed as New ICBM',
    text: 'Unverified surveillance footage captures secret intercontinental ballistic missile launch test today.',
    expected: 'FAKE (Mismatch)',
  },
  {
    case_id: 'CASE_05',
    category: 'deepfake_faceswap',
    title: 'Subtle Diplomatic Face-Swap',
    text: 'Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.',
    expected: 'FAKE (Faceswap)',
  },
  {
    case_id: 'CASE_06',
    category: 'deepfake_faceswap',
    title: 'Celebrity Deepfake Token Endorsement',
    text: 'Deepfake video swap depicts celebrity actor endorsing fraudulent cryptocurrency token scheme.',
    expected: 'FAKE (Faceswap)',
  },
  {
    case_id: 'CASE_08',
    category: 'diffusion_synthesis',
    title: 'AI Portrait with Ocular Asymmetry',
    text: 'Diffusion-generated portrait exhibiting telltale asymmetric iris reflections and blurred hair boundaries.',
    expected: 'FAKE (Diffusion)',
  },
  {
    case_id: 'CASE_11',
    category: 'spliced_banner',
    title: 'TV Broadcast with Spliced Bank Run',
    text: 'Authentic television broadcast altered with spliced banner claiming nationwide bank collapse.',
    expected: 'FAKE (Spliced)',
  },
  {
    case_id: 'CASE_13',
    category: 'synthetic_speech',
    title: 'Voice-Cloned Election Official Call',
    text: 'Cloned audio clip imitating election official claiming polling station closure in key district.',
    expected: 'FAKE (Cloned Voice)',
  },
  {
    case_id: 'CASE_15',
    category: 'authentic_edge_case',
    title: 'NASA Glacial Satellite Telemetry',
    text: 'Verified NASA satellite imagery documenting glacial ice shelf variations over five-year observation cycle.',
    expected: 'REAL (Authentic)',
  },
];

export const ProfessorDemoPage: React.FC = () => {
  const [selectedCase, setSelectedCase] = useState<typeof STANDARD_CASES[0]>(STANDARD_CASES[0]);
  const [customText, setCustomText] = useState<string>(STANDARD_CASES[0].text);
  const [activeMode, setActiveMode] = useState<'standard' | 'difficult'>('standard');
  const [loading, setLoading] = useState<boolean>(false);
  const [report, setReport] = useState<VerificationReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSelectCase = (c: typeof STANDARD_CASES[0]) => {
    setSelectedCase(c);
    setCustomText(c.text);
    setReport(null);
    setError(null);
  };

  const handleSelectDifficultCase = (c: typeof FIFTEEN_DIFFICULT_CASES[0]) => {
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
              <span className="text-xs text-slate-400 font-medium">CSET 346 — NLP Project</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">TRUTHLENS AI</h1>
            <p className="text-sm text-slate-300 max-w-2xl leading-relaxed">
              Multimodal NLP for Deepfake Detection and Explainable Content Verification.
              Evaluates multimodal evidence across text, visual, and acoustic streams to provide explainable reports.
            </p>
          </div>

          <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 text-xs space-y-1">
            <div className="text-slate-400">Current Implementation Stage:</div>
            <div className="font-bold text-sky-400">Tri-Modal + Cross-Attention</div>
            <div className="text-[11px] text-emerald-400 flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>320 Curated Samples / Checkpoint Loaded</span>
            </div>
          </div>
        </div>

        {/* Evaluation Policy Notice */}
        <div className="mt-6 pt-5 border-t border-slate-800 flex items-center space-x-2 text-xs text-slate-400">
          <Info className="w-4 h-4 text-sky-400 shrink-0" />
          <span>
            <strong>Transparency Notice:</strong> All metrics, predictions, token saliency, and confidence scores are computed by actual model forward passes without hardcoded mock values.
          </span>
        </div>
      </div>

      {/* Mode Selector */}
      <div className="flex space-x-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveMode('standard')}
          className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
            activeMode === 'standard'
              ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Standard Test Scenarios
        </button>
        <button
          onClick={() => setActiveMode('difficult')}
          className={`px-4 py-2 rounded-lg text-sm font-semibold transition flex items-center space-x-1.5 ${
            activeMode === 'difficult'
              ? 'bg-indigo-500/15 text-indigo-400 border border-indigo-500/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <BookOpen className="w-4 h-4" />
          <span>15 Difficult Cases Explorer (Experiment 5)</span>
        </button>
      </div>

      {/* Preset Test Scenarios */}
      {activeMode === 'standard' ? (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Select Demonstration Scenario
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {STANDARD_CASES.map((c) => {
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
      ) : (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
              15 Difficult Cases (Semantic Mismatch, Faceswaps, Diffusion, Cloned Voice)
            </h3>
            <span className="text-xs text-slate-500">Click any card to populate evaluation prompt</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {FIFTEEN_DIFFICULT_CASES.map((c) => (
              <button
                key={c.case_id}
                type="button"
                onClick={() => handleSelectDifficultCase(c)}
                className="p-3 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-sky-500/50 text-left transition"
              >
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="font-bold text-sky-400">{c.case_id}</span>
                  <span className="text-slate-400 text-[10px] px-1.5 py-0.5 rounded bg-slate-800">{c.expected}</span>
                </div>
                <div className="font-semibold text-xs text-white truncate">{c.title}</div>
                <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{c.text}</p>
              </button>
            ))}
          </div>
        </div>
      )}

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
