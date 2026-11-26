import React, { useState, useEffect } from 'react';
import {
  FileText,
  Image as ImageIcon,
  Video,
  Music,
  X,
  Sparkles,
  AlertCircle,
  Database,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Layers,
  Play,
  RotateCcw,
  Eye,
  ChevronLeft,
  ChevronRight,
  Globe,
  ExternalLink,
} from 'lucide-react';
import {
  verifyMultimodal,
  getFakedditBenchmarkResults,
  getFakedditSamples,
  getFakedditSampleDetail,
} from '../services/api';
import {
  VerificationReport,
  FakedditBenchmarkResults,
  FakedditSample,
} from '../types';
import { PredictionCard } from '../components/PredictionCard';
import { EvidenceSection } from '../components/EvidenceSection';

interface DemoCase {
  id: string;
  name: string;
  badge: 'REAL' | 'FAKE' | 'UNCERTAIN';
  category: string;
  text: string;
  image_file?: string;
  video_file?: string;
  audio_file?: string;
  description: string;
}

const NINE_DEMO_CASES: DemoCase[] = [
  {
    id: 'case_1',
    name: 'Case 1: Authentic Official Press Release (Easy REAL)',
    badge: 'REAL',
    category: 'Authentic Multi-modal',
    text: 'Ministry of Health releases updated quarterly public health statistics and preventive guidelines.',
    image_file: 'real_official_press_02.png',
    description: 'Consistent factual text with authentic, unmanipulated official press conference imagery.',
  },
  {
    id: 'case_2',
    name: 'Case 2: Synthetic Face Swap Manipulation (Easy FAKE)',
    badge: 'FAKE',
    category: 'Deepfake Face-Swap',
    text: 'Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.',
    image_file: 'fake_deepfake_faceswap_01.png',
    description: 'Generative blending boundary artifacts coupled with inflammatory textual claims.',
  },
  {
    id: 'case_3',
    name: 'Case 3: Authentic Imagery with Fabricated Caption (Semantic Mismatch)',
    badge: 'FAKE',
    category: 'Cross-Modal Mismatch',
    text: 'Breaking news: violent riots and civil unrest break out in downtown capital, teargas deployed.',
    image_file: 'real_official_press_01.png',
    description: 'Authentic press photo out-of-context with fabricated violent conflict claims.',
  },
  {
    id: 'case_4',
    name: 'Case 4: High-Fidelity Facial Reenactment (Face Manipulation)',
    badge: 'FAKE',
    category: 'Facial Landmark Tampering',
    text: 'Synthesized face-swap interview shows opposition politician making fabricated confessions.',
    image_file: 'fake_deepfake_faceswap_03.png',
    description: 'High-fidelity face-swap with subtle facial landmark warping.',
  },
  {
    id: 'case_5',
    name: 'Case 5: AI-Generated Disaster Scene (Diffusion Synthesis)',
    badge: 'FAKE',
    category: 'Latent Diffusion Generation',
    text: 'Synthetic diffusion image portrays fabricated catastrophic structural collapse of metropolitan bridge.',
    image_file: 'fake_diffusion_synthesis_02.png',
    description: 'Generative diffusion structural warping in bridge cables and unphysical reflections.',
  },
  {
    id: 'case_6',
    name: 'Case 6: Cross-Modal Context Inconsistency (Text-Image Conflict)',
    badge: 'FAKE',
    category: 'Semantic Misattribution',
    text: 'Emergency declared as uncontrolled chemical fire engulfs municipal residential sectors.',
    image_file: 'real_science_nature_02.png',
    description: 'Routine training exercise photo captioned as active catastrophic disaster.',
  },
  {
    id: 'case_7',
    name: 'Case 7: Temporal Video Manipulation (Video Deepfake)',
    badge: 'FAKE',
    category: 'Multi-Frame Video Manipulation',
    text: 'Surveillance recording captures suspicious unauthorized midnight access to secure government facility.',
    video_file: 'video_fake_001.mp4',
    description: 'Temporal frame-level manipulation evaluated across 8 uniformly sampled frames via ResNet18 feature-level pooling.',
  },
  {
    id: 'case_8',
    name: 'Case 8: Cloned Speech Impersonation (Audio Voice Clone)',
    badge: 'FAKE',
    category: 'Acoustic Voice Synthesis',
    text: 'Voice-cloned phone call impersonating executive officer authorizing fraudulent wire transfer.',
    audio_file: 'audio_fake_001.wav',
    description: 'Synthetic speech vocoder artifacts detected via 128-dimensional spectral acoustic features.',
  },
  {
    id: 'case_9',
    name: 'Case 9: Low-Confidence Ambiguous Claim (UNCERTAIN)',
    badge: 'UNCERTAIN',
    category: 'Uncertainty Policy Fallback',
    text: 'Unconfirmed rumor regarding sudden executive departures at regional logistics hub.',
    description: 'Softmax probability falls below operational 65% threshold, routing cleanly to human auditor.',
  },
];

interface ExternalDemoPreset {
  id: string;
  name: string;
  groundTruth: 'REAL' | 'FAKE';
  prediction: 'REAL' | 'FAKE' | 'UNCERTAIN';
  status: 'CORRECT' | 'INCORRECT' | 'UNCERTAIN';
  category: string;
  taxonomy: string;
  text: string;
  confidence: number;
  description: string;
}

const EXTERNAL_DEMO_PRESETS: ExternalDemoPreset[] = [
  {
    id: 'fakeddit_fake_001',
    name: 'Game of Thrones Emmy Award Claim',
    groundTruth: 'FAKE',
    prediction: 'UNCERTAIN',
    status: 'UNCERTAIN',
    category: 'Imposter Content',
    taxonomy: 'imposter_content',
    text: 'this one episode of game of thrones just won emmys',
    confidence: 0.634,
    description: 'Subtle imposter claim with authentic visual. Model confidence (63.4%) falls below 65% threshold, safely routing to human audit.',
  },
  {
    id: 'fakeddit_real_001',
    name: 'Palestinian Baby Breastfed by Israeli Nurse',
    groundTruth: 'REAL',
    prediction: 'UNCERTAIN',
    status: 'UNCERTAIN',
    category: 'Authentic Human News',
    taxonomy: 'authentic',
    text: 'israeli nurse breastfeeds palestinian baby after mom is critically injured',
    confidence: 0.542,
    description: 'Genuine historical report tested zero-shot. Demonstrates safe uncertain routing without negative false convictions.',
  },
  {
    id: 'fakeddit_fake_002',
    name: 'Supportive Online Message Board Claim',
    groundTruth: 'FAKE',
    prediction: 'FAKE',
    status: 'CORRECT',
    category: 'Satire / Parody Misinformation',
    taxonomy: 'satire_parody',
    text: 'hentai message board features surprisingly closeknit supportive community',
    confidence: 0.813,
    description: 'Parodic text-image mismatch detected with 81.3% fake confidence via bidirectional cross-attention.',
  },
  {
    id: 'fakeddit_real_003',
    name: 'Latika Chakravarty Business Realization',
    groundTruth: 'REAL',
    prediction: 'FAKE',
    status: 'INCORRECT',
    category: 'Domain Shift False Positive',
    taxonomy: 'authentic',
    text: 'starting her own business at latika chakravarty is proof that there is no expiry date for realizing your dreams',
    confidence: 0.825,
    description: 'Authentic news misclassified FAKE due to conversational social vocabulary shift compared to formal press releases.',
  },
  {
    id: 'fakeddit_fake_004',
    name: 'Lincoln Gettysburg Address Circa',
    groundTruth: 'FAKE',
    prediction: 'FAKE',
    status: 'CORRECT',
    category: 'Misleading Temporal Content',
    taxonomy: 'misleading_content',
    text: 'lincoln delivers the gettysburg address circa',
    confidence: 0.887,
    description: 'Historical claim paired with discordant modern photo flagged cleanly with 88.7% confidence.',
  },
  {
    id: 'fakeddit_fake_011',
    name: 'Grocery Store Cuddles Cat',
    groundTruth: 'FAKE',
    prediction: 'FAKE',
    status: 'CORRECT',
    category: 'Manipulated Visual Media',
    taxonomy: 'manipulated_content',
    text: 'he didnt choose the grocery store without this furr ball demanding cuddles',
    confidence: 0.769,
    description: 'Visual manipulation artifacts identified by ResNet-18 spatial feature maps.',
  },
];

interface VerificationPageProps {
  onNavigateToBenchmark?: () => void;
}

export const VerificationPage: React.FC<VerificationPageProps> = ({ onNavigateToBenchmark }) => {
  const [suiteMode, setSuiteMode] = useState<'in_domain' | 'external'>('in_domain');

  const [caseName, setCaseName] = useState<string>('');
  const [text, setText] = useState<string>('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioName, setAudioName] = useState<string | null>(null);

  const [selectedCaseId, setSelectedCaseId] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<VerificationReport | null>(null);
  const [demoNotice, setDemoNotice] = useState<string | null>(null);

  // External Fakeddit benchmark inspection state
  const [externalResults, setExternalResults] = useState<FakedditBenchmarkResults | null>(null);
  const [externalSamples, setExternalSamples] = useState<FakedditSample[]>([]);
  const [externalFilter, setExternalFilter] = useState<'all' | 'correct' | 'incorrect' | 'uncertain'>('all');
  const [externalIndex, setExternalIndex] = useState<number>(0);
  const [externalSampleDetail, setExternalSampleDetail] = useState<FakedditSample | null>(null);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [isExternalLoading, setIsExternalLoading] = useState<boolean>(false);

  // Lazy load external benchmark samples when switching to external suite
  useEffect(() => {
    if (suiteMode === 'external' && externalSamples.length === 0) {
      loadExternalData('all');
    }
  }, [suiteMode]);

  const loadExternalData = async (filter: 'all' | 'correct' | 'incorrect' | 'uncertain' = 'all') => {
    setIsExternalLoading(true);
    try {
      const [resData, samplesData] = await Promise.all([
        getFakedditBenchmarkResults(),
        getFakedditSamples(filter),
      ]);
      setExternalResults(resData);
      setExternalSamples(samplesData.samples);
      setExternalIndex(0);
      setShowHeatmap(false);
      if (samplesData.samples.length > 0) {
        const detail = await getFakedditSampleDetail(samplesData.samples[0].sample_id);
        setExternalSampleDetail(detail.sample);
      } else {
        setExternalSampleDetail(null);
      }
    } catch (err) {
      console.warn('Could not load external benchmark samples:', err);
    } finally {
      setIsExternalLoading(false);
    }
  };

  const handleExternalFilterChange = async (filter: 'all' | 'correct' | 'incorrect' | 'uncertain') => {
    setExternalFilter(filter);
    setIsExternalLoading(true);
    setShowHeatmap(false);
    try {
      const samplesData = await getFakedditSamples(filter);
      setExternalSamples(samplesData.samples);
      setExternalIndex(0);
      if (samplesData.samples.length > 0) {
        const detail = await getFakedditSampleDetail(samplesData.samples[0].sample_id);
        setExternalSampleDetail(detail.sample);
      } else {
        setExternalSampleDetail(null);
      }
    } catch (err: any) {
      console.warn('Error filtering external samples:', err);
    } finally {
      setIsExternalLoading(false);
    }
  };

  const handleExternalPrev = async () => {
    if (externalSamples.length === 0) return;
    const newIdx = externalIndex > 0 ? externalIndex - 1 : externalSamples.length - 1;
    setExternalIndex(newIdx);
    setShowHeatmap(false);
    try {
      const detail = await getFakedditSampleDetail(externalSamples[newIdx].sample_id);
      setExternalSampleDetail(detail.sample);
    } catch (e) {
      console.warn(e);
    }
  };

  const handleExternalNext = async () => {
    if (externalSamples.length === 0) return;
    const newIdx = externalIndex < externalSamples.length - 1 ? externalIndex + 1 : 0;
    setExternalIndex(newIdx);
    setShowHeatmap(false);
    try {
      const detail = await getFakedditSampleDetail(externalSamples[newIdx].sample_id);
      setExternalSampleDetail(detail.sample);
    } catch (e) {
      console.warn(e);
    }
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setVideoFile(null);
      setVideoPreview(null);
      setError(null);
    }
  };

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setImageFile(null);
      setImagePreview(null);
      setError(null);
    }
  };

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAudioFile(file);
      setAudioName(file.name);
      setError(null);
    }
  };

  const removeVisualMedia = () => {
    setImageFile(null);
    setImagePreview(null);
    setVideoFile(null);
    setVideoPreview(null);
  };

  const removeAudioMedia = () => {
    setAudioFile(null);
    setAudioName(null);
  };

  const handleSelectDemoCase = async (caseId: string) => {
    const c = NINE_DEMO_CASES.find((item) => item.id === caseId);
    if (!c) return;

    setSelectedCaseId(c.id);
    setCaseName(c.name);
    setText(c.text);
    removeVisualMedia();
    removeAudioMedia();
    setReport(null);
    setError(null);
    setDemoNotice(`Loaded In-Domain: "${c.name}" — ${c.description}`);

    try {
      if (c.image_file) {
        const res = await fetch(`/api/samples/${c.image_file}`);
        if (res.ok) {
          const blob = await res.blob();
          const file = new File([blob], c.image_file, { type: 'image/png' });
          setImageFile(file);
          setImagePreview(URL.createObjectURL(blob));
        }
      } else if (c.video_file) {
        const res = await fetch(`/api/samples/${c.video_file}`);
        if (res.ok) {
          const blob = await res.blob();
          const file = new File([blob], c.video_file, { type: 'video/mp4' });
          setVideoFile(file);
          setVideoPreview(URL.createObjectURL(blob));
        }
      }

      if (c.audio_file) {
        const res = await fetch(`/api/samples/${c.audio_file}`);
        if (res.ok) {
          const blob = await res.blob();
          const file = new File([blob], c.audio_file, { type: 'audio/wav' });
          setAudioFile(file);
          setAudioName(c.audio_file);
        }
      }
    } catch (err: any) {
      console.warn('Could not load sample asset from /api/samples:', err);
    }
  };

  const handleLoadAndVerifyExternalSample = async (
    sample: {
      sample_id?: string;
      id?: string;
      text: string;
      ground_truth?: string;
      groundTruth?: string;
      manipulation_type?: string;
      taxonomy?: string;
    },
    autoRun = true
  ) => {
    const sampleId = sample.sample_id || sample.id || '';
    const groundTruth = sample.ground_truth || sample.groundTruth || 'N/A';
    const taxonomy = sample.manipulation_type || sample.taxonomy || 'unseen';

    setSelectedCaseId(sampleId);
    setCaseName(`External Benchmark • ${sampleId}`);
    setText(sample.text);
    removeVisualMedia();
    removeAudioMedia();
    setReport(null);
    setError(null);
    setDemoNotice(
      `Loaded External Fakeddit Sample [${sampleId}] — Ground Truth: ${groundTruth}, Taxonomy: ${taxonomy}`
    );

    try {
      const res = await fetch(`/api/benchmark/fakeddit/image/${sampleId}`);
      if (res.ok) {
        const blob = await res.blob();
        const file = new File([blob], `${sampleId}.jpg`, { type: 'image/jpeg' });
        setImageFile(file);
        setImagePreview(URL.createObjectURL(blob));

        if (autoRun) {
          setLoading(true);
          try {
            const verifyRes = await verifyMultimodal(
              sample.text.trim(),
              file,
              undefined,
              undefined,
              `External Fakeddit • ${sampleId}`
            );
            setReport(verifyRes);
            setTimeout(() => {
              const el = document.getElementById('verification-result-section');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
            }, 100);
          } catch (vErr: any) {
            setError(vErr?.message || 'Verification could not be completed.');
          } finally {
            setLoading(false);
          }
        }
      }
    } catch (err: any) {
      console.warn('Could not load external sample image:', err);
    }
  };

  const handleVerify = async () => {
    if (!text.trim() && !imageFile && !videoFile && !audioFile) {
      setError('Please provide at least text, an image, a video, or an audio clip for verification.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await verifyMultimodal(
        text.trim() ? text.trim() : undefined,
        imageFile || undefined,
        videoFile || undefined,
        audioFile || undefined,
        caseName.trim() ? caseName.trim() : undefined
      );
      setReport(res);
      setTimeout(() => {
        const el = document.getElementById('verification-result-section');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err: any) {
      const userMessage = err?.message || 'Verification could not be completed. Please verify that backend models are operational.';
      setError(userMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setCaseName('');
    setText('');
    setSelectedCaseId('');
    removeVisualMedia();
    removeAudioMedia();
    setReport(null);
    setError(null);
    setDemoNotice(null);
  };

  const activeExternalSample =
    externalSampleDetail || (externalSamples.length > 0 ? externalSamples[externalIndex] : null);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Intro banner */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 mb-1.5">
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-sky-500/10 text-sky-400 border border-sky-500/20 uppercase tracking-wider">
              Multimodal Inference & Generalization Evaluation
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Multimodal Content Verification</h1>
          <p className="text-slate-400 text-sm mt-1">
            Detect deepfakes and verify digital claims across text, imagery, video, and audio using calibrated cross-attention fusion.
          </p>
        </div>
        {onNavigateToBenchmark && (
          <button
            onClick={onNavigateToBenchmark}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-indigo-500/15 hover:bg-indigo-500/25 border border-indigo-500/30 text-indigo-300 text-xs font-semibold transition shadow"
          >
            <Layers className="w-4 h-4 text-indigo-400" />
            <span>Fakeddit External Benchmark (Full Analytics & Matrix)</span>
          </button>
        )}
      </div>

      {/* Multimodal Demonstration & External Benchmark Testing Suite */}
      <div className="glass-panel rounded-2xl p-5 sm:p-6 border border-slate-800 bg-slate-900/50 space-y-5">
        {/* Suite Header with Mode Tabs */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-2.5">
            <Sparkles className="w-5 h-5 text-sky-400" />
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                Demonstration & Generalization Testing Suite
              </h2>
              <p className="text-[11px] text-slate-400">
                Pre-load curated benchmarks directly into the live verification pipeline
              </p>
            </div>
          </div>

          {/* Mode Switcher Tabs */}
          <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
            <button
              type="button"
              onClick={() => setSuiteMode('in_domain')}
              className={`px-3.5 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition ${
                suiteMode === 'in_domain'
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>In-Domain Course Suite (9)</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setSuiteMode('external');
                if (externalSamples.length === 0) {
                  loadExternalData('all');
                }
              }}
              className={`px-3.5 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition ${
                suiteMode === 'external'
                  ? 'bg-indigo-500/25 text-indigo-300 border border-indigo-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Globe className="w-3.5 h-3.5 text-indigo-400" />
              <span>External Fakeddit Suite (60)</span>
              <span className="text-[9px] px-1.5 py-0.2 rounded bg-indigo-500/30 text-indigo-200 uppercase font-mono font-bold">
                Unseen
              </span>
            </button>
          </div>
        </div>

        {/* MODE 1: In-Domain Course Suite */}
        {suiteMode === 'in_domain' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Trained Distribution: 9 Multimodal Course Benchmarks (160 Real / 160 Fake)</span>
              <span className="text-[11px] text-slate-500 hidden sm:inline">Click any card to pre-load text & media</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {NINE_DEMO_CASES.map((c) => {
                const isSelected = selectedCaseId === c.id;
                const badgeColor =
                  c.badge === 'REAL'
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                    : c.badge === 'FAKE'
                    ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                    : 'bg-amber-500/10 text-amber-400 border-amber-500/20';

                return (
                  <button
                    key={c.id}
                    type="button"
                    onClick={() => handleSelectDemoCase(c.id)}
                    className={`p-2.5 rounded-xl text-left border transition flex items-start justify-between space-x-2 ${
                      isSelected
                        ? 'bg-sky-500/15 border-sky-500/50 text-white ring-1 ring-sky-500'
                        : 'bg-slate-900/80 hover:bg-slate-800/80 border-slate-800 text-slate-300'
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="flex items-center space-x-1.5 mb-1">
                        <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded border ${badgeColor}`}>
                          {c.badge}
                        </span>
                        <span className="text-[10px] text-slate-400 font-medium truncate">{c.category}</span>
                      </div>
                      <div className="text-xs font-semibold truncate text-slate-200">
                        {c.name.split(':')[1]?.trim() || c.name}
                      </div>
                    </div>
                    <Play className={`w-3.5 h-3.5 shrink-0 mt-2 ${isSelected ? 'text-sky-400' : 'text-slate-600'}`} />
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* MODE 2: External Fakeddit Suite */}
        {suiteMode === 'external' && (
          <div className="space-y-6">
            {/* External Suite Banner */}
            <div className="p-3.5 rounded-xl bg-indigo-950/30 border border-indigo-500/20 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-indigo-300">
              <div className="flex items-center space-x-2.5">
                <Globe className="w-4 h-4 text-indigo-400 shrink-0" />
                <span>
                  <strong>Unseen Generalization Testing:</strong> Evaluating frozen model on 60 multimodal Reddit claims (30 REAL / 30 FAKE) from UC Santa Barbara's Fakeddit corpus (LREC 2020). Zero-shot evaluation.
                </span>
              </div>
              {onNavigateToBenchmark && (
                <button
                  type="button"
                  onClick={onNavigateToBenchmark}
                  className="inline-flex items-center space-x-1 px-3 py-1 rounded-lg bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-200 text-[11px] font-medium shrink-0 border border-indigo-500/30 transition"
                >
                  <span>Full Confusion Matrix & Paper Info</span>
                  <ExternalLink className="w-3 h-3 ml-1" />
                </button>
              )}
            </div>

            {/* Part 2A: Curated External Presets (1-Click Run) */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300 flex items-center space-x-1.5">
                  <Play className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Featured External Presets (1-Click Live Test)</span>
                </span>
                <span className="text-[11px] text-slate-500">Click to automatically run live model inference</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {EXTERNAL_DEMO_PRESETS.map((p) => {
                  const isSelected = selectedCaseId === p.id;
                  const gtColor =
                    p.groundTruth === 'REAL'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : 'bg-rose-500/10 text-rose-400 border-rose-500/20';

                  const predColor =
                    p.prediction === 'REAL'
                      ? 'text-emerald-400'
                      : p.prediction === 'FAKE'
                      ? 'text-rose-400'
                      : 'text-amber-400';

                  return (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => handleLoadAndVerifyExternalSample(p, true)}
                      className={`p-2.5 rounded-xl text-left border transition flex items-start justify-between space-x-2 ${
                        isSelected
                          ? 'bg-indigo-500/20 border-indigo-500/60 text-white ring-1 ring-indigo-500'
                          : 'bg-slate-900/80 hover:bg-slate-800/80 border-slate-800 text-slate-300'
                      }`}
                    >
                      <div className="min-w-0 space-y-1">
                        <div className="flex items-center space-x-1.5">
                          <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded border ${gtColor}`}>
                            GT: {p.groundTruth}
                          </span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded bg-slate-800 border border-slate-700 ${predColor}`}>
                            Model: {p.prediction}
                          </span>
                          <span className="text-[10px] text-slate-400 truncate">{p.category}</span>
                        </div>
                        <div className="text-xs font-semibold truncate text-slate-200">{p.name}</div>
                        <p className="text-[10px] text-slate-400 truncate">"{p.text}"</p>
                      </div>
                      <div className="shrink-0 flex flex-col items-center mt-2">
                        <Play className={`w-3.5 h-3.5 ${isSelected ? 'text-indigo-400' : 'text-slate-500'}`} />
                        <span className="text-[8px] text-indigo-400 font-mono mt-0.5">TEST</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Part 2B: Full Interactive Sample-by-Sample Inspection Tool */}
            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 sm:p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
                <div>
                  <div className="flex items-center space-x-2">
                    <Eye className="w-4 h-4 text-sky-400" />
                    <h3 className="text-sm font-bold text-white">Sample-by-Sample Inspection</h3>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Step through individual unseen Fakeddit multimodal pairs to evaluate cross-attention reasoning and Grad-CAM.
                  </p>
                </div>

                {/* Filter Pills */}
                <div className="flex items-center space-x-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
                  <button
                    type="button"
                    onClick={() => handleExternalFilterChange('all')}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition ${
                      externalFilter === 'all'
                        ? 'bg-sky-500 text-white shadow'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    All ({externalResults?.evaluation_samples || 60})
                  </button>
                  <button
                    type="button"
                    onClick={() => handleExternalFilterChange('correct')}
                    className={`px-2.5 py-1 rounded-lg font-semibold flex items-center space-x-1 transition ${
                      externalFilter === 'correct'
                        ? 'bg-emerald-500 text-white shadow'
                        : 'text-slate-400 hover:text-emerald-400'
                    }`}
                  >
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Correct ({externalResults?.counts.correct || 14})</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleExternalFilterChange('incorrect')}
                    className={`px-2.5 py-1 rounded-lg font-semibold flex items-center space-x-1 transition ${
                      externalFilter === 'incorrect'
                        ? 'bg-rose-500 text-white shadow'
                        : 'text-slate-400 hover:text-rose-400'
                    }`}
                  >
                    <XCircle className="w-3 h-3" />
                    <span>Incorrect ({externalResults?.counts.incorrect || 19})</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleExternalFilterChange('uncertain')}
                    className={`px-2.5 py-1 rounded-lg font-semibold flex items-center space-x-1 transition ${
                      externalFilter === 'uncertain'
                        ? 'bg-amber-500 text-slate-950 font-bold shadow'
                        : 'text-slate-400 hover:text-amber-400'
                    }`}
                  >
                    <AlertTriangle className="w-3 h-3" />
                    <span>Uncertain ({externalResults?.counts.uncertain || 27})</span>
                  </button>
                </div>
              </div>

              {/* Carousel Navigation Bar */}
              <div className="flex items-center justify-between bg-slate-900/90 p-2.5 rounded-xl border border-slate-800 text-xs">
                <button
                  type="button"
                  onClick={handleExternalPrev}
                  disabled={externalSamples.length === 0}
                  className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold transition disabled:opacity-40"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous</span>
                </button>

                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-slate-300">
                    Sample {externalSamples.length > 0 ? externalIndex + 1 : 0} of {externalSamples.length}
                  </span>
                  {activeExternalSample && (
                    <span className="text-slate-500 font-mono text-[11px]">
                      ({activeExternalSample.sample_id})
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  onClick={handleExternalNext}
                  disabled={externalSamples.length === 0}
                  className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold transition disabled:opacity-40"
                >
                  <span>Next</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              {/* Active Sample Inspector Display */}
              {isExternalLoading ? (
                <div className="flex flex-col items-center justify-center py-12 space-y-2">
                  <RotateCcw className="w-6 h-6 text-sky-400 animate-spin" />
                  <span className="text-xs text-slate-400">Loading benchmark sample...</span>
                </div>
              ) : activeExternalSample ? (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                  {/* Left Column: Visual Media (5 cols) */}
                  <div className="lg:col-span-5 space-y-3">
                    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden relative group">
                      <div className="aspect-square w-full bg-slate-950 flex items-center justify-center relative">
                        {showHeatmap && activeExternalSample.heatmap_base64 ? (
                          <img
                            src={activeExternalSample.heatmap_base64}
                            alt="Grad-CAM Saliency Heatmap"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <img
                            src={activeExternalSample.image_url}
                            alt="Fakeddit external benchmark media"
                            className="w-full h-full object-cover"
                          />
                        )}
                      </div>

                      {/* Heatmap overlay toggle badge */}
                      {activeExternalSample.heatmap_base64 && (
                        <div className="absolute top-2.5 right-2.5">
                          <button
                            type="button"
                            onClick={() => setShowHeatmap(!showHeatmap)}
                            className={`px-2.5 py-1 rounded-lg text-xs font-bold shadow-lg transition flex items-center space-x-1.5 ${
                              showHeatmap
                                ? 'bg-rose-500 text-white border border-rose-400'
                                : 'bg-slate-900/90 text-slate-200 border border-slate-700 hover:bg-slate-800'
                            }`}
                          >
                            <Layers className="w-3 h-3" />
                            <span>{showHeatmap ? 'Grad-CAM Active' : 'View Grad-CAM'}</span>
                          </button>
                        </div>
                      )}

                      <div className="p-2.5 bg-slate-900/90 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                        <div className="flex items-center space-x-1.5">
                          <ImageIcon className="w-3.5 h-3.5 text-sky-400" />
                          <span>ResNet-18 (224×224 RGB)</span>
                        </div>
                        <span className="font-mono text-[10px] text-slate-400 uppercase font-semibold">
                          Taxonomy: {activeExternalSample.manipulation_type}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Right Column: Reasoning & Actions (7 cols) */}
                  <div className="lg:col-span-7 space-y-4">
                    {/* Caption in Quote Box */}
                    <div className="space-y-1.5">
                      <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-400">
                        <FileText className="w-3.5 h-3.5 text-indigo-400" />
                        <span>Text Claim / Social Media Caption</span>
                      </div>
                      <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-100 font-medium text-sm leading-relaxed">
                        "{activeExternalSample.text}"
                      </div>
                    </div>

                    {/* Status Badges */}
                    <div className="grid grid-cols-3 gap-2">
                      <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                        <span className="text-[10px] text-slate-400 block font-medium">Ground Truth</span>
                        <span
                          className={`text-xs font-extrabold mt-1 inline-block px-2 py-0.5 rounded-md ${
                            activeExternalSample.ground_truth === 'REAL'
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                              : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                          }`}
                        >
                          {activeExternalSample.ground_truth}
                        </span>
                      </div>

                      <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                        <span className="text-[10px] text-slate-400 block font-medium">TruthLens Prediction</span>
                        <span
                          className={`text-xs font-extrabold mt-1 inline-block px-2 py-0.5 rounded-md ${
                            activeExternalSample.prediction === 'REAL'
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                              : activeExternalSample.prediction === 'FAKE'
                              ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                              : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          }`}
                        >
                          {activeExternalSample.prediction}
                        </span>
                      </div>

                      <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                        <span className="text-[10px] text-slate-400 block font-medium">Verdict Status</span>
                        <span
                          className={`text-xs font-bold mt-1 inline-flex items-center space-x-1 px-2 py-0.5 rounded-md ${
                            activeExternalSample.status === 'CORRECT'
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              : activeExternalSample.status === 'UNCERTAIN'
                              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                              : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                          }`}
                        >
                          {activeExternalSample.status === 'CORRECT' && <CheckCircle2 className="w-3 h-3" />}
                          {activeExternalSample.status === 'UNCERTAIN' && <AlertTriangle className="w-3 h-3" />}
                          {activeExternalSample.status === 'INCORRECT' && <XCircle className="w-3 h-3" />}
                          <span>{activeExternalSample.status}</span>
                        </span>
                      </div>
                    </div>

                    {/* Confidence Meter */}
                    <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
                      <div className="flex items-center justify-between text-xs font-semibold">
                        <span className="text-slate-400">
                          Model Confidence: {(activeExternalSample.confidence * 100).toFixed(1)}%
                        </span>
                        <div className="space-x-3 text-slate-300">
                          <span className="text-emerald-400">
                            REAL: {(activeExternalSample.real_probability * 100).toFixed(1)}%
                          </span>
                          <span className="text-rose-400">
                            FAKE: {(activeExternalSample.fake_probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden flex">
                        <div
                          className="bg-emerald-500 h-full transition-all duration-300"
                          style={{ width: `${activeExternalSample.real_probability * 100}%` }}
                        />
                        <div
                          className="bg-rose-500 h-full transition-all duration-300"
                          style={{ width: `${activeExternalSample.fake_probability * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Token Saliency Chips */}
                    {activeExternalSample.token_saliency && activeExternalSample.token_saliency.length > 0 && (
                      <div className="p-2.5 bg-slate-900/80 border border-slate-800 rounded-xl space-y-1">
                        <span className="text-[10px] font-semibold text-slate-400 block">
                          Token Gradient Saliency (Influential Words):
                        </span>
                        <div className="flex flex-wrap gap-1">
                          {activeExternalSample.token_saliency.map((t, i) => (
                            <span
                              key={i}
                              className="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-slate-800 text-sky-300 border border-slate-700"
                            >
                              {t.token} <span className="text-[9px] text-slate-500">({t.score.toFixed(2)})</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Reasoning explanation */}
                    {activeExternalSample.explanation && (
                      <p className="text-xs text-slate-400 italic bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/80">
                        "{activeExternalSample.explanation}"
                      </p>
                    )}

                    {/* Action CTA Buttons */}
                    <div className="pt-2 flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={() => handleLoadAndVerifyExternalSample(activeExternalSample, true)}
                        className="px-4 py-2 rounded-xl text-xs font-semibold text-white bg-gradient-to-r from-indigo-500 to-sky-600 hover:from-indigo-400 hover:to-sky-500 shadow-md shadow-indigo-500/20 flex items-center space-x-1.5 transition"
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>⚡ Test in Live Verification Pipeline (Run Model Now)</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleLoadAndVerifyExternalSample(activeExternalSample, false)}
                        className="px-3.5 py-2 rounded-xl text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 flex items-center space-x-1.5 transition"
                      >
                        <FileText className="w-3.5 h-3.5" />
                        <span>Pre-load into Form Only (Edit Caption)</span>
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-slate-500 text-xs">
                  No samples match the selected filter.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Demo Notification / Active Preset Banner */}
        {demoNotice && (
          <div className="mt-2 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-sky-300 bg-sky-950/20 px-3 py-2 rounded-lg border border-sky-900/30">
            <span className="truncate">{demoNotice}</span>
            <button
              onClick={() => setDemoNotice(null)}
              className="text-slate-400 hover:text-white shrink-0 ml-2"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-center space-x-3 text-sm">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Input Form Box */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-6">
        {/* External Sample Active Pill Indicator */}
        {selectedCaseId.startsWith('fakeddit') && (
          <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2">
              <Globe className="w-4 h-4 text-indigo-400 shrink-0" />
              <span>
                <strong>External Generalization Test:</strong> Sample <code>{selectedCaseId}</code> from r/Fakeddit (LREC 2020). Evaluated against frozen model trained on course dataset.
              </span>
            </div>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-200 border border-indigo-500/30 font-semibold">
              Zero-Shot Test
            </span>
          </div>
        )}

        {/* Verification Case Name Input */}
        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1.5">
            Verification / Case Name (Optional)
          </label>
          <input
            type="text"
            value={caseName}
            onChange={(e) => setCaseName(e.target.value)}
            placeholder="e.g. Breaking News Broadcast Verification #01"
            className="w-full px-4 py-2.5 rounded-xl bg-slate-900 border border-slate-800 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 text-sm text-slate-200 placeholder:text-slate-500"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Text Claim Input */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                <FileText className="w-4 h-4 text-sky-400" />
                <span>Text Claim or Caption</span>
              </label>
              <span className="text-xs text-slate-500">{text.length}/5000 chars</span>
            </div>

            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste caption, headline, or claim to evaluate (e.g. 'Breaking: Official government broadcast confirms unprecedented policy shift')..."
              rows={8}
              className="w-full rounded-xl bg-slate-900 border border-slate-800 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 p-4 text-sm text-slate-100 placeholder-slate-500 resize-none transition"
            />
          </div>

          {/* Right Column: Independent Visual & Audio Media Upload */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                <Layers className="w-4 h-4 text-emerald-400" />
                <span>Media Attachments (Visual + Audio)</span>
              </label>
              <span className="text-xs text-slate-500">Supports simultaneous media</span>
            </div>

            {/* Visual Media Slot (Image OR Video) */}
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-300 flex items-center space-x-1.5">
                  <ImageIcon className="w-3.5 h-3.5 text-sky-400" />
                  <span>Visual Modality (Image or Video)</span>
                </span>
                {(imagePreview || videoPreview) && (
                  <button
                    type="button"
                    onClick={removeVisualMedia}
                    className="text-slate-400 hover:text-rose-400 flex items-center space-x-1 transition"
                  >
                    <X className="w-3.5 h-3.5" />
                    <span>Remove</span>
                  </button>
                )}
              </div>

              {!imagePreview && !videoPreview ? (
                <div className="grid grid-cols-2 gap-2 pt-1">
                  <label className="border border-dashed border-slate-800 hover:border-sky-500/50 rounded-lg p-3 flex flex-col items-center justify-center cursor-pointer bg-slate-950/40 hover:bg-slate-900/80 transition group h-24">
                    <ImageIcon className="w-5 h-5 text-sky-400 group-hover:scale-110 transition-transform mb-1" />
                    <span className="text-xs text-slate-300 font-medium">Add Image</span>
                    <span className="text-[10px] text-slate-500">PNG / JPG</span>
                    <input
                      type="file"
                      accept="image/png, image/jpeg, image/webp"
                      onChange={handleImageChange}
                      className="hidden"
                    />
                  </label>

                  <label className="border border-dashed border-slate-800 hover:border-purple-500/50 rounded-lg p-3 flex flex-col items-center justify-center cursor-pointer bg-slate-950/40 hover:bg-slate-900/80 transition group h-24">
                    <Video className="w-5 h-5 text-purple-400 group-hover:scale-110 transition-transform mb-1" />
                    <span className="text-xs text-slate-300 font-medium">Add Video</span>
                    <span className="text-[10px] text-slate-500">MP4 (8 Frames)</span>
                    <input
                      type="file"
                      accept="video/mp4, video/quicktime, video/avi"
                      onChange={handleVideoChange}
                      className="hidden"
                    />
                  </label>
                </div>
              ) : (
                <div className="relative rounded-lg border border-slate-800 bg-slate-950/80 p-2 flex items-center justify-center h-28 overflow-hidden">
                  {imagePreview && (
                    <img
                      src={imagePreview}
                      alt="Visual Preview"
                      className="max-h-24 object-contain rounded"
                    />
                  )}
                  {videoPreview && (
                    <video
                      src={videoPreview}
                      controls
                      className="max-h-24 object-contain rounded"
                    />
                  )}
                </div>
              )}
            </div>

            {/* Audio Media Slot (Independent of Visual) */}
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-300 flex items-center space-x-1.5">
                  <Music className="w-3.5 h-3.5 text-amber-400" />
                  <span>Acoustic Modality (Speech / Vocoder)</span>
                </span>
                {audioName && (
                  <button
                    type="button"
                    onClick={removeAudioMedia}
                    className="text-slate-400 hover:text-rose-400 flex items-center space-x-1 transition"
                  >
                    <X className="w-3.5 h-3.5" />
                    <span>Remove</span>
                  </button>
                )}
              </div>

              {!audioName ? (
                <label className="border border-dashed border-slate-800 hover:border-amber-500/50 rounded-lg p-3 flex flex-col items-center justify-center cursor-pointer bg-slate-950/40 hover:bg-slate-900/80 transition group h-20">
                  <Music className="w-5 h-5 text-amber-400 group-hover:scale-110 transition-transform mb-1" />
                  <span className="text-xs text-slate-300 font-medium">Add Audio File</span>
                  <span className="text-[10px] text-slate-500">WAV / MP3 (128-dim Spectral Features)</span>
                  <input
                    type="file"
                    accept="audio/wav, audio/mpeg, audio/mp3, audio/ogg"
                    onChange={handleAudioChange}
                    className="hidden"
                  />
                </label>
              ) : (
                <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-3 flex items-center justify-between">
                  <div className="flex items-center space-x-2.5 truncate">
                    <Music className="w-5 h-5 text-amber-400 shrink-0" />
                    <span className="text-xs font-medium text-slate-200 truncate">{audioName}</span>
                  </div>
                  <span className="text-[10px] text-amber-400 font-semibold px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 shrink-0">
                    STFT Ready
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Loading Multi-Stage Indicator */}
        {loading && (
          <div className="p-4 rounded-xl bg-sky-950/40 border border-sky-500/30 text-sky-200 space-y-3 animate-pulse">
            <div className="flex items-center space-x-3">
              <div className="w-5 h-5 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />
              <span className="font-semibold text-sm">Executing Multimodal Verification Pipeline...</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-300">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800">1. DistilBERT Tokenization</div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800">2. ResNet18 / STFT Acoustic</div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800">3. Cross-Modal Attention Fusion</div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800">4. Calibrated Decision (T=1.0)</div>
            </div>
          </div>
        )}

        {/* CTA Action Row */}
        <div className="pt-4 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
          <div className="text-xs text-slate-400">
            Selected Production Architecture: <span className="text-sky-400 font-medium">Config A (Baseline Cross-Attention + Calibrated Softmax)</span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              type="button"
              onClick={handleClear}
              className="px-4 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition flex items-center space-x-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
            <button
              type="button"
              onClick={handleVerify}
              disabled={loading}
              className="px-6 py-2.5 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 shadow-lg shadow-sky-500/25 disabled:opacity-50 flex items-center space-x-2 transition"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Verify Multimodal Content</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Verification Output */}
      {report && (
        <div id="verification-result-section" className="space-y-6 animate-fadeIn">
          {/* Persistence Confirmation Badge */}
          {report.id && (
            <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 flex items-center justify-between text-xs">
              <div className="flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>
                  <strong>Saved to Persistent Database:</strong> Case record #{report.id} ({report.case_name || 'Verification'}) safely stored in SQLite.
                </span>
              </div>
              <div className="flex items-center space-x-1.5 font-mono text-[11px] text-emerald-400 bg-emerald-950/60 px-2.5 py-1 rounded-md border border-emerald-500/30">
                <Database className="w-3.5 h-3.5" />
                <span>ID: {report.id}</span>
              </div>
            </div>
          )}

          <PredictionCard report={report} />
          <EvidenceSection evidence={report.evidence} originalImagePreview={imagePreview} />
        </div>
      )}
    </div>
  );
};

export default VerificationPage;
