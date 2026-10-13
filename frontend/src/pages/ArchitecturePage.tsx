import React, { useEffect, useState } from 'react';
import { Cpu, Layers, GitMerge, FileText, Image as ImageIcon, Video, Music } from 'lucide-react';
import { getModelInfo } from '../services/api';
import { ModelInfo } from '../types';

export const ArchitecturePage: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(console.error);
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">System Architecture & Pipeline</h1>
        <p className="text-slate-400 text-sm mt-1">
          Technical specifications of TruthLens AI multimodal deepfake verification architecture.
        </p>
      </div>

      {/* Model Specs Banner */}
      {modelInfo && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Text Backbone</span>
            <span className="text-sm font-bold text-sky-400 truncate block">{modelInfo.text_backbone}</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Vision Backbone</span>
            <span className="text-sm font-bold text-emerald-400 truncate block">{modelInfo.vision_backbone}</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Active Fusion</span>
            <span className="text-sm font-bold text-indigo-400 truncate block">{modelInfo.fusion_architecture}</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Uncertainty Cutoff</span>
            <span className="text-sm font-bold text-amber-400 truncate block">{(modelInfo.uncertainty_threshold * 100).toFixed(0)}% Threshold</span>
          </div>
        </div>
      )}

      {/* Architecture Flowchart Cards */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-6">
        <h3 className="text-lg font-bold text-white flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-sky-400" />
          <span>Multimodal Verification Pipeline Flow</span>
        </h3>

        <div className="space-y-4 text-xs">
          {/* Step 1: Input Modalities */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
            <span className="font-bold text-slate-300 text-sm mb-2 block">1. Input Ingestion Layer</span>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center space-x-2">
                <FileText className="w-4 h-4 text-sky-400" />
                <span>Text / Caption</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center space-x-2">
                <ImageIcon className="w-4 h-4 text-emerald-400" />
                <span>Image Frame</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center space-x-2">
                <Video className="w-4 h-4 text-purple-400" />
                <span>Video (8 Frames)</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center space-x-2 opacity-60">
                <Music className="w-4 h-4 text-slate-400" />
                <span>Audio (Planned)</span>
              </div>
            </div>
          </div>

          {/* Step 2: Encoders */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
            <span className="font-bold text-slate-300 text-sm mb-2 block">2. Dual Modality Encoders</span>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="font-semibold text-sky-400">NLP: DistilBERT (768-dim)</span>
                <p className="text-slate-400 text-[11px]">
                  Extracts token sequence representations and [CLS] pooled context embedding, projected to 256-dim shared latent space.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="font-semibold text-emerald-400">Vision: ResNet18 (512-dim)</span>
                <p className="text-slate-400 text-[11px]">
                  Extracts 7x7 spatial convolutional maps (conv_stem layer4) and global average pooled vector, projected to 256-dim space.
                </p>
              </div>
            </div>
          </div>

          {/* Step 3: Fusion Engine */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
            <span className="font-bold text-slate-300 text-sm mb-2 block">3. Multimodal Fusion Engine</span>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <div className="flex items-center space-x-1.5 text-indigo-400 font-semibold">
                  <GitMerge className="w-4 h-4" />
                  <span>Bidirectional Cross-Attention (Active)</span>
                </div>
                <p className="text-slate-400 text-[11px]">
                  Text queries attend to visual patches (Text → Vision) and visual queries attend to text tokens (Vision → Text) with residual connections.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <div className="flex items-center space-x-1.5 text-slate-300 font-semibold">
                  <Layers className="w-4 h-4 text-sky-400" />
                  <span>Feature Concatenation Baseline</span>
                </div>
                <p className="text-slate-400 text-[11px]">
                  Direct concatenation [256 + 256 = 512] passing through a 2-layer MLP projection.
                </p>
              </div>
            </div>
          </div>

          {/* Step 4: Decision & Explainability */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
            <span className="font-bold text-slate-300 text-sm mb-2 block">4. Decision Policy & Explainability Engine</span>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="font-semibold text-white block mb-1">Classification Head</span>
                <span className="text-slate-400 text-[11px]">
                  Softmax binary probabilities over REAL (0) vs FAKE (1).
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="font-semibold text-amber-400 block mb-1">Uncertainty Policy</span>
                <span className="text-slate-400 text-[11px]">
                  Returns UNCERTAIN whenever max(real_prob, fake_prob) &lt; 0.65 threshold.
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="font-semibold text-sky-400 block mb-1">Explainability Engine</span>
                <span className="text-slate-400 text-[11px]">
                  Token gradient saliency + ResNet18 Layer4 Grad-CAM heatmaps.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
