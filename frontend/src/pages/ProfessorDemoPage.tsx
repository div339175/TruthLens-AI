import React, { useState } from 'react';
import {
  GraduationCap,
  CheckCircle2,
  ShieldCheck,
  ArrowRight,
  BookOpen,
  Image as ImageIcon,
  Layers,
  Sparkles,
} from 'lucide-react';
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
    image_file: 'real_official_press_02.png',
    description: 'Authentic news broadcast text with consistent factual language and no synthetic artifacts.',
  },
  {
    id: 'case_fake',
    title: 'AI Deepfake Synthetic Face Swap',
    category: 'FAKE',
    text: 'Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.',
    image_file: 'fake_deepfake_faceswap_01.png',
    description: 'Generative blending boundary artifacts coupled with inflammatory textual claims.',
  },
  {
    id: 'case_uncertain',
    title: 'Ambiguous / Low Signal Post',
    category: 'UNCERTAIN',
    text: 'Unconfirmed rumor regarding sudden executive departures at regional logistics hub.',
    image_file: null,
    description: 'Demonstrates uncertainty policy when model confidence does not exceed 65% threshold.',
  },
];

const FIFTEEN_DIFFICULT_CASES = [
  {
    case_id: 'CASE_01',
    category: 'semantic_mismatch',
    title: 'Festival Miscaptioned as Street Riot',
    text: 'Breaking news: violent riots and civil unrest break out in downtown capital, teargas deployed.',
    image_file: 'real_official_press_01.png',
    expected: 'FAKE (Mismatch)',
    rationale: 'Clean authentic imagery paired with fabricated violent conflict claims.',
  },
  {
    case_id: 'CASE_02',
    category: 'semantic_mismatch',
    title: 'Archival Space Launch Claimed as New ICBM',
    text: 'Unverified surveillance footage captures secret intercontinental ballistic missile launch test today.',
    image_file: 'real_science_nature_01.png',
    expected: 'FAKE (Mismatch)',
    rationale: 'Authentic space exploration footage repurposed with modern ballistic warfare headline.',
  },
  {
    case_id: 'CASE_03',
    category: 'semantic_mismatch',
    title: 'Wildfire Drill Repurposed as Disaster',
    text: 'Emergency declared as uncontrolled chemical fire engulfs municipal residential sectors.',
    image_file: 'real_science_nature_02.png',
    expected: 'FAKE (Mismatch)',
    rationale: 'Routine training exercise photo captioned as active catastrophic disaster.',
  },
  {
    case_id: 'CASE_04',
    category: 'semantic_mismatch',
    title: 'Hospital Opening Claimed as Military Biolab',
    text: 'Leaked photograph exposes clandestine bioweapon research laboratory operating under commercial cover.',
    image_file: 'real_verified_journalism_01.png',
    expected: 'FAKE (Mismatch)',
    rationale: 'Legitimate medical ribbon-cutting captioned as conspiratorial biolab.',
  },
  {
    case_id: 'CASE_05',
    category: 'deepfake_faceswap',
    title: 'Subtle Diplomatic Face-Swap',
    text: 'Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.',
    image_file: 'fake_deepfake_faceswap_01.png',
    expected: 'FAKE (Faceswap)',
    rationale: 'Feathered boundary blending requiring fine spatial frequency analysis.',
  },
  {
    case_id: 'CASE_06',
    category: 'deepfake_faceswap',
    title: 'Celebrity Deepfake Token Endorsement',
    text: 'Deepfake video swap depicts celebrity actor endorsing fraudulent cryptocurrency token scheme.',
    image_file: 'fake_deepfake_faceswap_02.png',
    expected: 'FAKE (Faceswap)',
    rationale: 'High-fidelity generative facial replacement matching lighting conditions.',
  },
  {
    case_id: 'CASE_07',
    category: 'deepfake_faceswap',
    title: 'Synthesized Politician Debate',
    text: 'Synthesized face-swap interview shows opposition politician making fabricated confessions.',
    image_file: 'fake_deepfake_faceswap_03.png',
    expected: 'FAKE (Faceswap)',
    rationale: 'Subtle facial expression manipulation with localized landmark artifacts.',
  },
  {
    case_id: 'CASE_08',
    category: 'diffusion_synthesis',
    title: 'AI Portrait with Ocular Asymmetry',
    text: 'Diffusion-generated portrait exhibiting telltale asymmetric iris reflections and blurred hair boundaries.',
    image_file: 'fake_diffusion_synthesis_01.png',
    expected: 'FAKE (Diffusion)',
    rationale: 'High-frequency generative artifacts in ocular geometry and hair edges.',
  },
  {
    case_id: 'CASE_09',
    category: 'diffusion_synthesis',
    title: 'Synthesized Disaster Scene',
    text: 'Synthetic diffusion image portrays fabricated catastrophic structural collapse of metropolitan bridge.',
    image_file: 'fake_diffusion_synthesis_02.png',
    expected: 'FAKE (Diffusion)',
    rationale: 'Diffusion structural warping in bridge cables and unphysical water reflections.',
  },
  {
    case_id: 'CASE_10',
    category: 'diffusion_synthesis',
    title: 'AI Military Hardware Prototype',
    text: 'Generative AI composite portrays non-existent stealth aircraft prototype parked on foreign military airfield.',
    image_file: 'fake_diffusion_synthesis_03.png',
    expected: 'FAKE (Diffusion)',
    rationale: 'Photorealistic aircraft composite with background perspective inconsistencies.',
  },
  {
    case_id: 'CASE_11',
    category: 'spliced_banner',
    title: 'TV Broadcast with Spliced Bank Run',
    text: 'Authentic television broadcast altered with spliced banner claiming nationwide bank collapse.',
    image_file: 'fake_spliced_banner_01.png',
    expected: 'FAKE (Spliced)',
    rationale: 'Sharply spliced news chyron overlay creating sharp edge gradients.',
  },
  {
    case_id: 'CASE_12',
    category: 'spliced_banner',
    title: 'Cloned Military Convoy Vehicles',
    text: 'Digital image shows troop convoy with cloned military vehicles duplicated to exaggerate deployment scale.',
    image_file: 'fake_spliced_banner_02.png',
    expected: 'FAKE (Spliced)',
    rationale: 'Copy-move patch duplication preserving vehicle lighting and perspective.',
  },
  {
    case_id: 'CASE_13',
    category: 'synthetic_speech',
    title: 'Voice-Cloned Election Official Call',
    text: 'Cloned audio clip imitating election official claiming polling station closure in key district.',
    image_file: 'fake_synthetic_speech_01.png',
    expected: 'FAKE (Voice/Multi)',
    rationale: 'Acoustic vocoder artifacts paired with fabricated administrative announcement.',
  },
  {
    case_id: 'CASE_14',
    category: 'synthetic_speech',
    title: 'Synthesized Authorization Call',
    text: 'Voice-cloned phone call impersonating executive officer authorizing fraudulent wire transfer.',
    image_file: 'fake_synthetic_speech_02.png',
    expected: 'FAKE (Voice/Multi)',
    rationale: 'Pitch-flattened vocal harmonics and urgent business diction.',
  },
  {
    case_id: 'CASE_15',
    category: 'authentic_edge_case',
    title: 'NASA Glacial Satellite Telemetry',
    text: 'Verified NASA satellite imagery documenting glacial ice shelf variations over five-year observation cycle.',
    image_file: 'real_science_nature_03.png',
    expected: 'REAL (Authentic)',
    rationale: 'Complex irregular satellite textures that uncalibrated models often false-positive.',
  },
];

export const ProfessorDemoPage: React.FC = () => {
  const [selectedTitle, setSelectedTitle] = useState<string>(STANDARD_CASES[0].title);
  const [customText, setCustomText] = useState<string>(STANDARD_CASES[0].text);
  const [selectedImageFile, setSelectedImageFile] = useState<string | null>(STANDARD_CASES[0].image_file);
  const [activeMode, setActiveMode] = useState<'standard' | 'difficult'>('standard');
  const [loading, setLoading] = useState<boolean>(false);
  const [report, setReport] = useState<VerificationReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSelectCase = (c: typeof STANDARD_CASES[0]) => {
    setSelectedTitle(c.title);
    setCustomText(c.text);
    setSelectedImageFile(c.image_file);
    setReport(null);
    setError(null);
  };

  const handleSelectDifficultCase = (c: typeof FIFTEEN_DIFFICULT_CASES[0]) => {
    setSelectedTitle(c.title);
    setCustomText(c.text);
    setSelectedImageFile(c.image_file);
    setReport(null);
    setError(null);
  };

  const handleRunDemoVerification = async () => {
    setLoading(true);
    setError(null);

    try {
      let imageFile: File | undefined = undefined;

      // When an image file is paired with the case, fetch the sample media and pass to pipeline
      if (selectedImageFile) {
        try {
          const imgRes = await fetch(`/api/samples/${selectedImageFile}`);
          if (imgRes.ok) {
            const blob = await imgRes.blob();
            imageFile = new File([blob], selectedImageFile, { type: 'image/png' });
          }
        } catch (fetchErr) {
          console.warn('Could not load sample image; falling back to text-only:', fetchErr);
        }
      }

      // Live verification sending BOTH text and image to backend
      const res = await verifyMultimodal(
        customText,
        imageFile,
        undefined,
        undefined,
        `Demo: ${selectedTitle}`
      );
      setReport(res);
    } catch (err: any) {
      setError(err.message || 'Demo verification error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
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
              Multimodal deepfake and misinformation verification. Demonstrates joint text (DistilBERT) and vision (ResNet18) analysis with Bidirectional Cross-Attention and Layer4 Grad-CAM explainability.
            </p>
          </div>

          <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 text-xs space-y-1">
            <div className="text-slate-400">Pipeline Mode:</div>
            <div className="font-bold text-sky-400">Bidirectional Cross-Attention</div>
            <div className="text-[11px] text-emerald-400 flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Full Model Pipeline Active</span>
            </div>
          </div>
        </div>
      </div>

      {/* Mode Switcher */}
      <div className="flex items-center space-x-2 border-b border-slate-800 pb-4">
        <button
          onClick={() => {
            setActiveMode('standard');
            handleSelectCase(STANDARD_CASES[0]);
          }}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center space-x-1.5 ${
            activeMode === 'standard'
              ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30 shadow-md'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          <span>Standard Demonstration Cases</span>
        </button>

        <button
          onClick={() => {
            setActiveMode('difficult');
            handleSelectDifficultCase(FIFTEEN_DIFFICULT_CASES[0]);
          }}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center space-x-1.5 ${
            activeMode === 'difficult'
              ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 shadow-md'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <BookOpen className="w-4 h-4" />
          <span>15 Difficult Cases (CSET 346 Error Analysis)</span>
        </button>
      </div>

      {/* Case Selector Cards */}
      {activeMode === 'standard' ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
              Standard Multimodal Benchmark Presets:
            </h3>
            <span className="text-xs text-slate-500">Click to load paired multimodal inputs</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {STANDARD_CASES.map((c) => {
              const isSelected = selectedTitle === c.title;
              return (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => handleSelectCase(c)}
                  className={`p-4 rounded-xl border text-left transition ${
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
                  {c.image_file && (
                    <span className="text-[10px] text-sky-400 flex items-center space-x-1 mt-2">
                      <ImageIcon className="w-3 h-3" />
                      <span>Paired Image: {c.image_file}</span>
                    </span>
                  )}
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
            <span className="text-xs text-slate-500">Click any card to load multimodal pair</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {FIFTEEN_DIFFICULT_CASES.map((c) => {
              const isSelected = selectedTitle === c.title;
              return (
                <button
                  key={c.case_id}
                  type="button"
                  onClick={() => handleSelectDifficultCase(c)}
                  className={`p-3 rounded-xl border text-left transition ${
                    isSelected
                      ? 'bg-indigo-500/15 border-indigo-500/50 shadow-md'
                      : 'bg-slate-900/70 border-slate-800 hover:border-sky-500/50'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] mb-1">
                    <span className="font-bold text-sky-400">{c.case_id}</span>
                    <span className="text-slate-400 text-[10px] px-1.5 py-0.5 rounded bg-slate-800">
                      {c.expected}
                    </span>
                  </div>
                  <div className="font-semibold text-xs text-white truncate">{c.title}</div>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{c.rationale}</p>
                  <div className="flex items-center justify-between mt-2 pt-1.5 border-t border-slate-800 text-[10px] text-slate-500">
                    <span className="capitalize">{c.category.replace('_', ' ')}</span>
                    <span className="text-sky-400 flex items-center space-x-1">
                      <ImageIcon className="w-3 h-3" />
                      <span>Has Image</span>
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Paired Multimodal Input Box */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              Active Evaluation Sample
            </span>
            <h3 className="text-base font-bold text-white">{selectedTitle}</h3>
          </div>

          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded-md bg-sky-500/10 text-sky-400 border border-sky-500/20 text-xs font-semibold flex items-center space-x-1">
              <Layers className="w-3.5 h-3.5" />
              <span>{selectedImageFile ? 'Modality: Text + Image (Cross-Attention)' : 'Modality: Text-Only'}</span>
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Text Claim Area */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-400">Textual Claim / Headline:</label>
            <textarea
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              rows={6}
              className="w-full rounded-xl bg-slate-900 border border-slate-800 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 p-4 text-xs text-slate-100 resize-none leading-relaxed"
            />
          </div>

          {/* Paired Image Area */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-400">Paired Dataset Image:</label>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 h-40 flex items-center justify-center overflow-hidden">
              {selectedImageFile ? (
                <div className="text-center">
                  <img
                    src={`/api/samples/${selectedImageFile}`}
                    alt="Paired Multimodal Evidence"
                    className="max-h-32 object-contain rounded-lg mx-auto shadow"
                  />
                  <span className="text-[10px] text-slate-400 block mt-1 font-mono">{selectedImageFile}</span>
                </div>
              ) : (
                <div className="text-center text-slate-500 text-xs">
                  <span>No image attached (Text-only evaluation)</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex justify-end pt-2">
          <button
            type="button"
            onClick={handleRunDemoVerification}
            disabled={loading}
            className="px-6 py-3 rounded-xl text-sm font-bold text-white bg-gradient-to-r from-indigo-500 to-sky-600 hover:from-indigo-400 hover:to-sky-500 shadow-lg shadow-indigo-500/20 flex items-center space-x-2 transition"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Running Multimodal Cross-Attention...</span>
              </>
            ) : (
              <>
                <span>Verify Multimodal Pair with TruthLens</span>
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
          <div className="flex items-center justify-between text-xs text-slate-400">
            <div className="flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Inspection Complete — Real Model Pipeline Execution</span>
            </div>
            {report.id && (
              <span className="font-mono text-emerald-400">Archived ID: #{report.id}</span>
            )}
          </div>
          <PredictionCard report={report} />
          <EvidenceSection
            evidence={report.evidence}
            originalImagePreview={selectedImageFile ? `/api/samples/${selectedImageFile}` : null}
          />
        </div>
      )}
    </div>
  );
};
