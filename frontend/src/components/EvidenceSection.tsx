import React, { useState } from 'react';
import { Layers, FileText, Image as ImageIcon, Video, Music, GitMerge, Eye } from 'lucide-react';
import { ModalityEvidence } from '../types';

interface EvidenceSectionProps {
  evidence: ModalityEvidence[];
  originalImagePreview?: string | null;
}

export const EvidenceSection: React.FC<EvidenceSectionProps> = ({ evidence, originalImagePreview }) => {
  const [showHeatmap, setShowHeatmap] = useState(true);

  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="glass-panel rounded-2xl p-6 sm:p-8 mt-6 border border-slate-800">
      <div className="flex items-center space-x-2 pb-5 border-b border-slate-800">
        <Layers className="w-5 h-5 text-sky-400" />
        <h3 className="text-lg font-bold text-white tracking-tight">Multimodal Evidence Breakdown</h3>
      </div>

      <div className="mt-6 space-y-6">
        {evidence.map((ev, idx) => (
          <div key={idx} className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80">
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                {ev.modality === 'text' && <FileText className="w-4 h-4 text-sky-400" />}
                {ev.modality === 'visual' && <ImageIcon className="w-4 h-4 text-emerald-400" />}
                {ev.modality === 'video' && <Video className="w-4 h-4 text-purple-400" />}
                {ev.modality === 'audio' && <Music className="w-4 h-4 text-amber-400" />}
                {ev.modality === 'cross-modal' && <GitMerge className="w-4 h-4 text-indigo-400" />}
                <span className="font-semibold text-sm text-white capitalize">{ev.modality} Modality Evidence</span>
              </div>
              <span
                className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${
                  ev.type === 'model-derived'
                    ? 'bg-sky-500/10 text-sky-400 border-sky-500/20'
                    : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                }`}
              >
                {ev.type.toUpperCase()}
              </span>
            </div>

            {/* Description */}
            <p className="text-xs text-slate-300 leading-relaxed mb-4">{ev.description}</p>

            {/* Modality: Text Token Saliency */}
            {ev.modality === 'text' && ev.token_importance && ev.token_importance.length > 0 && (
              <div className="mt-3 p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                <span className="text-[11px] text-slate-400 block mb-2 font-medium">
                  Salient Tokens (Gradient Attribution Score):
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {ev.token_importance.map((tok, tIdx) => {
                    const score = tok.score;
                    // Color based on attribution magnitude
                    const colorStyle =
                      score > 0.8
                        ? 'bg-rose-500/20 border-rose-500/40 text-rose-300 font-bold'
                        : score > 0.5
                        ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                        : 'bg-slate-800 border-slate-700 text-slate-300';

                    return (
                      <span
                        key={tIdx}
                        className={`text-xs px-2.5 py-1 rounded-md border flex items-center space-x-1.5 ${colorStyle}`}
                      >
                        <span>{tok.token}</span>
                        <span className="text-[10px] opacity-75">{(score * 100).toFixed(0)}%</span>
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Modality: Visual Grad-CAM Overlay */}
            {ev.modality === 'visual' && ev.heatmap_available && ev.heatmap_base64 && (
              <div className="mt-3 p-4 rounded-lg bg-slate-950/70 border border-slate-800">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-medium text-slate-300">
                    Grad-CAM Spatial Manipulation Heatmap (Layer4 Activation)
                  </span>
                  {originalImagePreview && (
                    <button
                      onClick={() => setShowHeatmap(!showHeatmap)}
                      className="text-[11px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center space-x-1 transition-colors"
                    >
                      <Eye className="w-3 h-3 text-sky-400" />
                      <span>{showHeatmap ? 'View Original' : 'View Heatmap'}</span>
                    </button>
                  )}
                </div>

                <div className="flex justify-center items-center p-2 rounded bg-black/40 max-h-80 overflow-hidden">
                  <img
                    src={showHeatmap ? ev.heatmap_base64 : (originalImagePreview || ev.heatmap_base64)}
                    alt="Visual Explainability Heatmap"
                    className="max-h-72 rounded-lg object-contain border border-slate-800 shadow-md"
                  />
                </div>
                <div className="mt-2 text-[11px] text-slate-500 text-center">
                  Warmer colors (Red/Yellow) highlight regions with disproportionate activation in the visual feature space.
                </div>
              </div>
            )}

            {/* Modality: Video metadata */}
            {ev.modality === 'video' && ev.details && (
              <div className="mt-2 text-xs text-slate-400 bg-slate-950/70 p-3 rounded-lg border border-slate-800 flex items-center justify-between">
                <span>Sampled Frames Count: <strong className="text-white">{ev.details.sampled_frames_count || 8}</strong></span>
                <span>Temporal Sampling: <strong className="text-white">Uniform linspace</strong></span>
              </div>
            )}

            {/* Modality: Audio metadata */}
            {ev.modality === 'audio' && ev.details && (
              <div className="mt-2 text-xs text-slate-400 bg-slate-950/70 p-3 rounded-lg border border-slate-800 grid grid-cols-1 sm:grid-cols-3 gap-2">
                <div>
                  <span className="block text-[11px] text-slate-500">Clip Duration:</span>
                  <strong className="text-white">{Number(ev.details.duration_sec || 0).toFixed(2)}s</strong>
                </div>
                <div>
                  <span className="block text-[11px] text-slate-500">Zero-Crossing Rate:</span>
                  <strong className="text-amber-400 font-mono">{Number(ev.details.zero_crossing_rate || 0).toFixed(4)}</strong>
                </div>
                <div>
                  <span className="block text-[11px] text-slate-500">High-Freq Ratio:</span>
                  <strong className="text-amber-400 font-mono">{Number(ev.details.high_freq_ratio || 0).toFixed(4)}</strong>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
