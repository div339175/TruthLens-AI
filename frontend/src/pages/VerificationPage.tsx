import React, { useState } from 'react';
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
  Layers,
  Play,
  RotateCcw,
} from 'lucide-react';
import { verifyMultimodal } from '../services/api';
import { VerificationReport } from '../types';
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

export const VerificationPage: React.FC = () => {
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

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      // Video and image are mutually exclusive visual formats
      setVideoFile(null);
      setVideoPreview(null);
      // Keep audioFile intact so user can verify Text + Image + Audio!
      setError(null);
    }
  };

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      // Video and image are mutually exclusive visual formats
      setImageFile(null);
      setImagePreview(null);
      // Keep audioFile intact so user can verify Text + Video + Audio!
      setError(null);
    }
  };

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAudioFile(file);
      setAudioName(file.name);
      // Audio does NOT clear image or video, enabling multimodal tri-modal verification
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
    setDemoNotice(`Loaded: "${c.name}" — ${c.description}`);

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

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Intro banner */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Multimodal Content Verification</h1>
          <p className="text-slate-400 text-sm mt-1">
            Detect deepfakes and verify digital claims across text, imagery, video, and audio using calibrated cross-attention fusion.
          </p>
        </div>
      </div>

      {/* 9 Curated Demo Showcase Bar */}
      <div className="glass-panel rounded-2xl p-5 border border-slate-800 bg-slate-900/50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-sky-400" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Curated Demonstration Suite (9 Representative Benchmarks)
            </h2>
          </div>
          <span className="text-[11px] text-slate-500 hidden sm:inline">
            Click any preset to pre-load text & sample media
          </span>
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
                  <div className="text-xs font-semibold truncate text-slate-200">{c.name.split(':')[1]?.trim() || c.name}</div>
                </div>
                <Play className={`w-3.5 h-3.5 shrink-0 mt-2 ${isSelected ? 'text-sky-400' : 'text-slate-600'}`} />
              </button>
            );
          })}
        </div>

        {demoNotice && (
          <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-sky-300 bg-sky-950/20 px-3 py-2 rounded-lg border border-sky-900/30">
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
        <div className="space-y-6 animate-fadeIn">
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