import React, { useState } from 'react';
import { FileText, Image as ImageIcon, Video, Music, X, Sparkles, AlertCircle, Database, CheckCircle2 } from 'lucide-react';
import { verifyMultimodal } from '../services/api';
import { VerificationReport } from '../types';
import { PredictionCard } from '../components/PredictionCard';
import { EvidenceSection } from '../components/EvidenceSection';

export const VerificationPage: React.FC = () => {
  const [caseName, setCaseName] = useState<string>('');
  const [text, setText] = useState<string>('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioName, setAudioName] = useState<string | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<VerificationReport | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setVideoFile(null);
      setVideoPreview(null);
      setAudioFile(null);
      setAudioName(null);
    }
  };

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setImageFile(null);
      setImagePreview(null);
      setAudioFile(null);
      setAudioName(null);
    }
  };

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAudioFile(file);
      setAudioName(file.name);
      setImageFile(null);
      setImagePreview(null);
      setVideoFile(null);
      setVideoPreview(null);
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
      setError(err.message || 'An error occurred during verification.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setCaseName('');
    setText('');
    setImageFile(null);
    setImagePreview(null);
    setVideoFile(null);
    setVideoPreview(null);
    setAudioFile(null);
    setAudioName(null);
    setReport(null);
    setError(null);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Intro banner */}
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Multimodal Content Verification</h1>
        <p className="text-slate-400 text-sm mt-1">
          Verify digital claims across text, imagery, video, and audio using cross-modal deep learning and explainable evidence extraction.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-center space-x-3 text-sm">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Input Form Box */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-6">
        {/* Optional Case Name Input */}
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
              rows={7}
              className="w-full rounded-xl bg-slate-900 border border-slate-800 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 p-4 text-sm text-slate-100 placeholder-slate-500 resize-none transition"
            />

            {/* Quick Preset buttons */}
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="text-slate-500">Presets:</span>
              <button
                type="button"
                onClick={() => setText('Ministry of Health releases updated quarterly public health statistics and preventive guidelines.')}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              >
                Authentic Official
              </button>
              <button
                type="button"
                onClick={() => setText('Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.')}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              >
                Deepfake Faceswap
              </button>
              <button
                type="button"
                onClick={() => setText('Cloned audio clip imitating election official claiming polling station closure in key district.')}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              >
                Cloned Voice
              </button>
            </div>
          </div>

          {/* Right Column: Visual & Audio Media Upload */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                <ImageIcon className="w-4 h-4 text-emerald-400" />
                <span>Non-Textual Evidence (Image, Video, Audio)</span>
              </label>
              <span className="text-xs text-slate-500">PNG, JPG, MP4, WAV, MP3</span>
            </div>

            {/* Upload Area / Previews */}
            {!imagePreview && !videoPreview && !audioName ? (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                {/* Image Picker */}
                <label className="border-2 border-dashed border-slate-800 hover:border-sky-500/50 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer bg-slate-900/40 hover:bg-slate-900/80 transition group h-52">
                  <div className="w-10 h-10 rounded-xl bg-sky-500/10 flex items-center justify-center text-sky-400 group-hover:scale-110 transition-transform mb-2">
                    <ImageIcon className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-semibold text-slate-300">Upload Image</span>
                  <span className="text-[10px] text-slate-500 mt-1 text-center">ResNet18 & CAM</span>
                  <input
                    type="file"
                    accept="image/png, image/jpeg, image/webp"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                </label>

                {/* Video Picker */}
                <label className="border-2 border-dashed border-slate-800 hover:border-purple-500/50 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer bg-slate-900/40 hover:bg-slate-900/80 transition group h-52">
                  <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center text-purple-400 group-hover:scale-110 transition-transform mb-2">
                    <Video className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-semibold text-slate-300">Upload Video</span>
                  <span className="text-[10px] text-slate-500 mt-1 text-center">8 Frames Sample</span>
                  <input
                    type="file"
                    accept="video/mp4, video/quicktime, video/avi"
                    onChange={handleVideoChange}
                    className="hidden"
                  />
                </label>

                {/* Audio Picker */}
                <label className="border-2 border-dashed border-slate-800 hover:border-amber-500/50 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer bg-slate-900/40 hover:bg-slate-900/80 transition group h-52">
                  <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center text-amber-400 group-hover:scale-110 transition-transform mb-2">
                    <Music className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-semibold text-slate-300">Upload Audio</span>
                  <span className="text-[10px] text-slate-500 mt-1 text-center">Spectral Vocoder</span>
                  <input
                    type="file"
                    accept="audio/wav, audio/mpeg, audio/mp3, audio/ogg"
                    onChange={handleAudioChange}
                    className="hidden"
                  />
                </label>
              </div>
            ) : (
              <div className="relative rounded-xl border border-slate-800 bg-slate-900/60 p-3 h-52 flex items-center justify-center overflow-hidden">
                <button
                  type="button"
                  onClick={() => {
                    setImageFile(null);
                    setImagePreview(null);
                    setVideoFile(null);
                    setVideoPreview(null);
                    setAudioFile(null);
                    setAudioName(null);
                  }}
                  className="absolute top-3 right-3 p-1.5 rounded-full bg-slate-800/80 hover:bg-rose-500/80 text-white transition z-10"
                  title="Remove media"
                >
                  <X className="w-4 h-4" />
                </button>

                {imagePreview && (
                  <img
                    src={imagePreview}
                    alt="Upload Preview"
                    className="max-h-44 object-contain rounded-lg"
                  />
                )}
                {videoPreview && (
                  <video
                    src={videoPreview}
                    controls
                    className="max-h-44 object-contain rounded-lg"
                  />
                )}
                {audioName && (
                  <div className="text-center p-4">
                    <Music className="w-12 h-12 text-amber-400 mx-auto mb-2 animate-pulse" />
                    <span className="text-sm font-semibold text-slate-200 block truncate max-w-xs">{audioName}</span>
                    <span className="text-xs text-slate-400">Audio ready for acoustic feature extraction</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* CTA Action Row */}
        <div className="mt-8 pt-6 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
          <div className="text-xs text-slate-400">
            Active Architecture: <span className="text-sky-400 font-medium">Text (DistilBERT) + Vision (ResNet18) + Audio (Spectral) + Bidirectional Cross-Attention</span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              type="button"
              onClick={handleClear}
              className="px-4 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition"
            >
              Reset
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
                  <span>Running Multimodal Pipeline...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Verify Content</span>
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
