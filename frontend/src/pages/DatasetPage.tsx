import React, { useEffect, useState } from 'react';
import { Database, CheckCircle2, AlertCircle, FileSpreadsheet } from 'lucide-react';
import { getDatasetStats } from '../services/api';
import { DatasetStats } from '../types';

export const DatasetPage: React.FC = () => {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    getDatasetStats()
      .then(setStats)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Dataset Repository & Statistics</h1>
        <p className="text-slate-400 text-sm mt-1">
          Actual verified dataset split statistics and academic benchmark acquisition guidelines.
        </p>
      </div>

      {/* Live Dataset Statistics Card */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800">
        <div className="flex items-center justify-between pb-6 border-b border-slate-800 flex-wrap gap-4">
          <div className="flex items-center space-x-2">
            <FileSpreadsheet className="w-5 h-5 text-sky-400" />
            <h3 className="text-lg font-bold text-white">Active Metadata Statistics (ml/data/metadata/dataset.csv)</h3>
          </div>
          {stats?.status === 'initialized' ? (
            <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-semibold flex items-center space-x-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Verified Metadata File</span>
            </span>
          ) : (
            <span className="px-3 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs font-semibold flex items-center space-x-1.5">
              <AlertCircle className="w-3.5 h-3.5" />
              <span>Dataset Not Initialized</span>
            </span>
          )}
        </div>

        {loading ? (
          <div className="py-12 text-center text-slate-400 text-sm">
            <div className="w-6 h-6 border-2 border-sky-400 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <span>Inspecting metadata file...</span>
          </div>
        ) : stats?.status === 'initialized' ? (
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 block mb-1">Total Samples</span>
              <span className="text-2xl font-extrabold text-white">{stats.total_samples}</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 block mb-1">Authentic (REAL)</span>
              <span className="text-2xl font-extrabold text-emerald-400">{stats.real_count}</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 block mb-1">Manipulated (FAKE)</span>
              <span className="text-2xl font-extrabold text-rose-400">{stats.fake_count}</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 block mb-1">Train Split</span>
              <span className="text-2xl font-extrabold text-sky-400">{stats.train_count}</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 block mb-1">Val Split</span>
              <span className="text-2xl font-extrabold text-indigo-400">{stats.val_count}</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 block mb-1">Test Split</span>
              <span className="text-2xl font-extrabold text-purple-400">{stats.test_count}</span>
            </div>
          </div>
        ) : (
          <div className="py-8 text-center text-slate-400 text-sm">
            <p className="text-amber-300 font-medium mb-1">Dataset not initialized.</p>
            <p className="text-xs text-slate-500">
              Run <code className="text-sky-400 bg-slate-900 px-1.5 py-0.5 rounded">python scripts/download_dataset.py --sample</code> to initialize the development dataset.
            </p>
          </div>
        )}
      </div>

      {/* Candidate Benchmark Datasets */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center space-x-2">
          <Database className="w-5 h-5 text-indigo-400" />
          <span>Supported Benchmark Datasets for University Evaluation</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-panel p-5 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-bold text-white text-sm">FaceForensics++ (FF++)</h4>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">Video Manipulation</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Standard benchmark for Deepfakes, Face2Face, FaceSwap, and NeuralTextures manipulations across multiple compression rates (c23, c40).
            </p>
            <div className="text-[11px] text-slate-500 pt-2 border-t border-slate-800/80">
              Command: <code>python scripts/download_dataset.py --dataset faceforensics</code>
            </div>
          </div>

          <div className="glass-panel p-5 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-bold text-white text-sm">Fakeddit</h4>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Multimodal Text+Image</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Over 1M multimodal Reddit posts with submission text, titles, image URLs, and fine-grained 6-way manipulation labels.
            </p>
            <div className="text-[11px] text-slate-500 pt-2 border-t border-slate-800/80">
              Command: <code>python scripts/download_dataset.py --dataset fakeddit</code>
            </div>
          </div>

          <div className="glass-panel p-5 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-bold text-white text-sm">FakeAVCeleb</h4>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">Audio-Visual Multimodal</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Synthesized celebrity videos with deepfake video swaps and cloned synthetic voices for cross-modal audiovisual inconsistency.
            </p>
            <div className="text-[11px] text-slate-500 pt-2 border-t border-slate-800/80">
              Command: <code>python scripts/download_dataset.py --dataset fakeavceleb</code>
            </div>
          </div>

          <div className="glass-panel p-5 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-bold text-white text-sm">DFDC (Deepfake Detection Challenge)</h4>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">Large-Scale Benchmark</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Kaggle/AWS dataset with over 100,000 video clips featuring diverse actors, backgrounds, lighting conditions, and audio synthesis.
            </p>
            <div className="text-[11px] text-slate-500 pt-2 border-t border-slate-800/80">
              Command: <code>python scripts/download_dataset.py --dataset dfdc</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
