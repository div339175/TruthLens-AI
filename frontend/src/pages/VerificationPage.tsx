import React, { useState } from 'react';
import { FileText, Image as ImageIcon, Video, X, Sparkles, AlertCircle } from 'lucide-react';
import { verifyMultimodal } from '../services/api';
import { VerificationReport } from '../types';
import { PredictionCard } from '../components/PredictionCard';
import { EvidenceSection } from '../components/EvidenceSection';

export const VerificationPage: React.FC = () => {
  const [text, setText] = useState<string>('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<VerificationReport | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      // Reset video if image is chosen
      setVideoFile(null);
      setVideoPreview(null);
    }
  };

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      // Reset image if video is chosen
      setImageFile(null);
      setImagePreview(null);
    }
  };

  const handleVerify = async () => {
    if (!text.trim() && !imageFile && !videoFile) {
      setError('Please provide at least text, an image, or a video for verification.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await verifyMultimodal(
        text.trim() ? text.trim() : undefined,
        imageFile || undefined,
        videoFile || undefined
      );
      setReport(res);
    } catch (err: any) {
      setError(err.message || 'An error occurred during verification.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setImageFile(null);
    setImagePreview(null);
    setVideoFile(null);
    setVideoPreview(null);
    setReport(null);
    setError(null);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Intro banner */}
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Multimodal Content Verification</h1>
        <p className="text-slate-400 text-sm mt-1">
          Verify digital claims across text, imagery, and video using cross-modal deep learning and explainable evidence extraction.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-center space-x-3 text-sm">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Input Form Box */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800">
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
                onClick={() => setText('Verified satellite observation confirms Antarctic ice sheet stability.')}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              >
                Authentic Claim
              </button>
              <button
                type="button"
                onClick={() => setText('Shocking leaked synthetic video shows prominent political leader in fabricated controversy.')}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              >
                Deepfake Claim
              </button>
            </div>
          </div>

          {/* Right Column: Visual Media Upload */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                <ImageIcon className="w-4 h-4 text-emerald-400" />
                <span>Visual Evidence (Image / Video)</span>
              </label>
              <span className="text-xs text-slate-500">PNG, JPG, MP4, MOV</span>
            </div>

            {/* Upload Area / Previews */}
            {!imagePreview && !videoPreview ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {/* Image Picker */}
                <label className="border-2 border-dashed border-slate-800 hover:border-sky-500/50 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer bg-slate-900/40 hover:bg-slate-900/80 transition group h-52">
                  <div className="w-12 h-12 rounded-xl bg-sky-500/10 flex items-center justify-center text-sky-400 group-hover:scale-110 transition-transform mb-3">
                    <ImageIcon className="w-6 h-6" />
                  </div>
                  <span className="text-xs font-semibold text-slate-300">Upload Image</span>
                  <span className="text-[11px] text-slate-500 mt-1 text-center">ResNet18 & Grad-CAM</span>
                  <input
                    type="file"
                    accept="image/png, image/jpeg, image/webp"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                </label>

                {/* Video Picker */}
                <label className="border-2 border-dashed border-slate-800 hover:border-purple-500/50 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer bg-slate-900/40 hover:bg-slate-900/80 transition group h-52">
                  <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center text-purple-400 group-hover:scale-110 transition-transform mb-3">
                    <Video className="w-6 h-6" />
                  </div>
                  <span className="text-xs font-semibold text-slate-300">Upload Video</span>
                  <span className="text-[11px] text-slate-500 mt-1 text-center">8 Uniform Frames</span>
                  <input
                    type="file"
                    accept="video/mp4, video/quicktime, video/avi"
                    onChange={handleVideoChange}
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
              </div>
            )}
          </div>
        </div>

        {/* CTA Action Row */}
        <div className="mt-8 pt-6 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
          <div className="text-xs text-slate-400">
            Pipeline: <span className="text-sky-400 font-medium">Text (DistilBERT) + Vision (ResNet18) + Cross-Attention</span>
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
          <PredictionCard report={report} />
          <EvidenceSection evidence={report.evidence} originalImagePreview={imagePreview} />
        </div>
      )}
    </div>
  );
};
