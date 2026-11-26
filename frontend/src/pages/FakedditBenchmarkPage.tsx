import React, { useState, useEffect } from 'react';
import {
  Play,
  RotateCcw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Eye,
  FileText,
  Image as ImageIcon,
  BarChart3,
  Sparkles,
  Layers,
} from 'lucide-react';
import {
  getFakedditBenchmarkResults,
  runFakedditBenchmark,
  getFakedditSamples,
  getFakedditSampleDetail,
} from '../services/api';
import { FakedditBenchmarkResults, FakedditSample } from '../types';

export const FakedditBenchmarkPage: React.FC = () => {
  const [results, setResults] = useState<FakedditBenchmarkResults | null>(null);
  const [samples, setSamples] = useState<FakedditSample[]>([]);
  const [activeFilter, setActiveFilter] = useState<'all' | 'correct' | 'incorrect' | 'uncertain'>('all');
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [currentSampleDetail, setCurrentSampleDetail] = useState<FakedditSample | null>(null);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isRunningBenchmark, setIsRunningBenchmark] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Load initial benchmark results and samples
  useEffect(() => {
    loadData('all');
  }, []);

  const loadData = async (filter: 'all' | 'correct' | 'incorrect' | 'uncertain') => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const [resData, samplesData] = await Promise.all([
        getFakedditBenchmarkResults(),
        getFakedditSamples(filter),
      ]);
      setResults(resData);
      setSamples(samplesData.samples);
      setCurrentIndex(0);
      if (samplesData.samples.length > 0) {
        loadDetail(samplesData.samples[0].sample_id);
      } else {
        setCurrentSampleDetail(null);
      }
    } catch (err: any) {
      console.error('Error loading Fakeddit data:', err);
      setErrorMsg(err.message || 'Failed to load Fakeddit benchmark data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFilterChange = async (filter: 'all' | 'correct' | 'incorrect' | 'uncertain') => {
    setActiveFilter(filter);
    setIsLoading(true);
    try {
      const samplesData = await getFakedditSamples(filter);
      setSamples(samplesData.samples);
      setCurrentIndex(0);
      setShowHeatmap(false);
      if (samplesData.samples.length > 0) {
        loadDetail(samplesData.samples[0].sample_id);
      } else {
        setCurrentSampleDetail(null);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to filter samples');
    } finally {
      setIsLoading(false);
    }
  };

  const loadDetail = async (sampleId: string) => {
    try {
      const detail = await getFakedditSampleDetail(sampleId);
      setCurrentSampleDetail(detail.sample);
    } catch (err) {
      console.warn('Could not load detailed sample:', err);
    }
  };

  const handleRunFullBenchmark = async () => {
    setIsRunningBenchmark(true);
    setErrorMsg(null);
    try {
      await runFakedditBenchmark();
      await loadData(activeFilter);
    } catch (err: any) {
      setErrorMsg(err.message || 'Benchmark run failed');
    } finally {
      setIsRunningBenchmark(false);
    }
  };

  const handlePrev = () => {
    if (samples.length === 0) return;
    const newIdx = currentIndex > 0 ? currentIndex - 1 : samples.length - 1;
    setCurrentIndex(newIdx);
    setShowHeatmap(false);
    loadDetail(samples[newIdx].sample_id);
  };

  const handleNext = () => {
    if (samples.length === 0) return;
    const newIdx = currentIndex < samples.length - 1 ? currentIndex + 1 : 0;
    setCurrentIndex(newIdx);
    setShowHeatmap(false);
    loadDetail(samples[newIdx].sample_id);
  };

  const jumpToCase = (targetStatus: 'CORRECT' | 'INCORRECT' | 'UNCERTAIN') => {
    handleFilterChange(targetStatus.toLowerCase() as any);
  };

  const activeSample = currentSampleDetail || (samples.length > 0 ? samples[currentIndex] : null);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/20 rounded-2xl p-6 sm:p-8 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 uppercase tracking-wider">
                Research Evaluation • Unseen External Dataset
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                LREC 2020 Peer-Reviewed
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Fakeddit External Generalization Benchmark
            </h1>
            <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
              Evaluating TruthLens AI’s frozen <strong>DistilBERT + ResNet-18 Cross-Attention</strong> architecture on 
              <strong> 60 unseen multimodal samples</strong> (30 REAL / 30 FAKE) from the UC Santa Barbara Fakeddit corpus. 
              Tests real-world domain generalization, semantic mismatches, and calibrated uncertainty handling.
            </p>
            <div className="pt-2 flex flex-wrap items-center gap-3 text-xs text-slate-400">
              <span className="inline-flex items-center space-x-1.5 bg-slate-800/60 px-3 py-1 rounded-md border border-slate-700/60">
                <span className="font-semibold text-slate-200">Training Source:</span>
                <span>TruthLens Course Dataset (320 samples)</span>
              </span>
              <span className="inline-flex items-center space-x-1.5 bg-rose-500/10 text-rose-300 px-3 py-1 rounded-md border border-rose-500/20">
                <span className="font-semibold">Fakeddit Used in Training:</span>
                <span className="font-bold underline">NO (Zero-Shot)</span>
              </span>
              <span className="inline-flex items-center space-x-1.5 bg-emerald-500/10 text-emerald-300 px-3 py-1 rounded-md border border-emerald-500/20">
                <span className="font-semibold">External Test:</span>
                <span className="font-bold">YES (60 Samples)</span>
              </span>
            </div>
          </div>

          {/* Action Button */}
          <div className="flex flex-col sm:flex-row lg:flex-col gap-3 shrink-0">
            <button
              onClick={handleRunFullBenchmark}
              disabled={isRunningBenchmark}
              className="inline-flex items-center justify-center space-x-2 px-5 py-3 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 transition-all disabled:opacity-50"
            >
              {isRunningBenchmark ? (
                <>
                  <RotateCcw className="w-4 h-4 animate-spin" />
                  <span>Evaluating 60 Samples...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Run Full Benchmark</span>
                </>
              )}
            </button>

            {/* Quick Demo Shortcuts */}
            <div className="flex items-center gap-1.5 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800 text-[11px]">
              <span className="text-slate-400 font-semibold px-2">Demo:</span>
              <button
                onClick={() => jumpToCase('CORRECT')}
                className="px-2 py-1 rounded bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-400 font-medium transition"
              >
                ✓ Correct
              </button>
              <button
                onClick={() => jumpToCase('INCORRECT')}
                className="px-2 py-1 rounded bg-rose-500/15 hover:bg-rose-500/25 text-rose-400 font-medium transition"
              >
                ✗ Incorrect
              </button>
              <button
                onClick={() => jumpToCase('UNCERTAIN')}
                className="px-2 py-1 rounded bg-amber-500/15 hover:bg-amber-500/25 text-amber-400 font-medium transition"
              >
                ⚠ Uncertain
              </button>
            </div>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Benchmark Summary Metrics */}
      {results && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4">
            <span className="text-xs text-slate-400 font-medium block">Total Samples</span>
            <span className="text-2xl font-bold text-white mt-1 block">{results.evaluation_samples}</span>
            <span className="text-[11px] text-slate-500 mt-1 block">30 REAL / 30 FAKE</span>
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4">
            <span className="text-xs text-slate-400 font-medium block">Strict Accuracy</span>
            <span className="text-2xl font-bold text-sky-400 mt-1 block">{(results.accuracy * 100).toFixed(1)}%</span>
            <span className="text-[11px] text-slate-500 mt-1 block">Resolved: {(results.resolved_accuracy * 100).toFixed(1)}%</span>
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4">
            <span className="text-xs text-slate-400 font-medium block">Precision (Fake)</span>
            <span className="text-2xl font-bold text-indigo-400 mt-1 block">{(results.precision * 100).toFixed(1)}%</span>
            <span className="text-[11px] text-slate-500 mt-1 block">F1: {(results.f1_score * 100).toFixed(1)}%</span>
          </div>

          <div className="bg-slate-900/70 border border-emerald-500/20 bg-emerald-500/5 rounded-xl p-4">
            <div className="flex items-center space-x-1 text-emerald-400 text-xs font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Correct</span>
            </div>
            <span className="text-2xl font-bold text-emerald-400 mt-1 block">{results.counts.correct}</span>
            <span className="text-[11px] text-emerald-500/80 mt-1 block">
              {((results.counts.correct / results.evaluation_samples) * 100).toFixed(1)}% of total
            </span>
          </div>

          <div className="bg-slate-900/70 border border-rose-500/20 bg-rose-500/5 rounded-xl p-4">
            <div className="flex items-center space-x-1 text-rose-400 text-xs font-medium">
              <XCircle className="w-3.5 h-3.5" />
              <span>Incorrect</span>
            </div>
            <span className="text-2xl font-bold text-rose-400 mt-1 block">{results.counts.incorrect}</span>
            <span className="text-[11px] text-rose-500/80 mt-1 block">
              {((results.counts.incorrect / results.evaluation_samples) * 100).toFixed(1)}% of total
            </span>
          </div>

          <div className="bg-slate-900/70 border border-amber-500/20 bg-amber-500/5 rounded-xl p-4">
            <div className="flex items-center space-x-1 text-amber-400 text-xs font-medium">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Uncertain</span>
            </div>
            <span className="text-2xl font-bold text-amber-400 mt-1 block">{results.counts.uncertain}</span>
            <span className="text-[11px] text-amber-500/80 mt-1 block">
              {((results.counts.uncertain / results.evaluation_samples) * 100).toFixed(1)}% fallback
            </span>
          </div>
        </div>
      )}

      {/* Confusion Matrix Card */}
      {results && results.confusion_matrix && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
            <div>
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-5 h-5 text-indigo-400" />
                <h2 className="text-base font-bold text-white">External Confusion Matrix</h2>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-500/20 text-indigo-300 uppercase">
                  External / Unseen Data
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Binary performance mapping across 60 external instances ($y=0$ Real, $y=1$ Fake).
              </p>
            </div>
            <div className="text-xs text-slate-400 bg-slate-800/60 px-3 py-1.5 rounded-lg border border-slate-700/60 flex items-center space-x-4">
              <span><strong>Recall (Fake):</strong> {(results.recall * 100).toFixed(1)}%</span>
              <span><strong>F1-Score:</strong> {results.f1_score.toFixed(4)}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border border-slate-800 rounded-xl overflow-hidden text-sm">
              <table className="w-full text-center">
                <thead>
                  <tr className="bg-slate-800/80 text-slate-300 text-xs uppercase font-semibold">
                    <th className="p-3 text-left">Ground Truth</th>
                    <th className="p-3 bg-slate-800/40">Predicted REAL (0)</th>
                    <th className="p-3 bg-slate-800/40">Predicted FAKE (1)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-200">
                  <tr>
                    <td className="p-3 text-left font-semibold text-slate-300 bg-slate-800/20">True REAL</td>
                    <td className="p-3 bg-emerald-500/10 text-emerald-300 font-bold">
                      {results.confusion_matrix.true_negatives} (TN)
                    </td>
                    <td className="p-3 bg-rose-500/10 text-rose-300 font-bold">
                      {results.confusion_matrix.false_positives} (FP)
                    </td>
                  </tr>
                  <tr>
                    <td className="p-3 text-left font-semibold text-slate-300 bg-slate-800/20">True FAKE</td>
                    <td className="p-3 bg-rose-500/10 text-rose-300 font-bold">
                      {results.confusion_matrix.false_negatives} (FN)
                    </td>
                    <td className="p-3 bg-emerald-500/10 text-emerald-300 font-bold">
                      {results.confusion_matrix.true_positives} (TP)
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-center text-xs space-y-2 text-slate-400">
              <p className="font-semibold text-slate-300">Key Scientific Takeaways:</p>
              <p>• <strong>High Recall on External Fakes ({ (results.recall * 100).toFixed(1) }%):</strong> The model successfully identified {results.confusion_matrix.true_positives} out of 30 deceptive social media claims.</p>
              <p>• <strong>Domain Shift on External Real News:</strong> Casual, non-governmental Reddit posts exhibit domain shift from the 320 formal course announcements, illustrating the value of uncertainty routing.</p>
              <p>• <strong>Uncertainty Policy:</strong> {results.counts.uncertain} samples safely fell below the 65% certainty threshold, preventing reckless false convictions.</p>
            </div>
          </div>
        </div>
      )}

      {/* Interactive Sample-by-Sample Demo Viewer */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="flex items-center space-x-2">
              <Eye className="w-5 h-5 text-sky-400" />
              <h2 className="text-lg font-bold text-white">Sample-by-Sample Inspection</h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Step through individual unseen Fakeddit multimodal pairs to evaluate cross-attention reasoning and Grad-CAM.
            </p>
          </div>

          {/* Three Demo Filters */}
          <div className="flex items-center space-x-1.5 bg-slate-950/80 p-1.5 rounded-xl border border-slate-800 text-xs">
            <button
              onClick={() => handleFilterChange('all')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                activeFilter === 'all'
                  ? 'bg-sky-500 text-white shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              All (60)
            </button>
            <button
              onClick={() => handleFilterChange('correct')}
              className={`px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1 transition ${
                activeFilter === 'correct'
                  ? 'bg-emerald-500 text-white shadow'
                  : 'text-slate-400 hover:text-emerald-400'
              }`}
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Correct ({results?.counts.correct || 0})</span>
            </button>
            <button
              onClick={() => handleFilterChange('incorrect')}
              className={`px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1 transition ${
                activeFilter === 'incorrect'
                  ? 'bg-rose-500 text-white shadow'
                  : 'text-slate-400 hover:text-rose-400'
              }`}
            >
              <XCircle className="w-3.5 h-3.5" />
              <span>Incorrect ({results?.counts.incorrect || 0})</span>
            </button>
            <button
              onClick={() => handleFilterChange('uncertain')}
              className={`px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1 transition ${
                activeFilter === 'uncertain'
                  ? 'bg-amber-500 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-amber-400'
              }`}
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Uncertain ({results?.counts.uncertain || 0})</span>
            </button>
          </div>
        </div>

        {/* Carousel Navigation Bar */}
        <div className="flex items-center justify-between bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
          <button
            onClick={handlePrev}
            disabled={samples.length === 0}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition disabled:opacity-40"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Previous</span>
          </button>

          <div className="flex items-center space-x-2 text-xs">
            <span className="font-semibold text-slate-300">
              Sample {samples.length > 0 ? currentIndex + 1 : 0} of {samples.length}
            </span>
            {activeSample && (
              <span className="text-slate-500 font-mono text-[11px]">
                ({activeSample.sample_id})
              </span>
            )}
          </div>

          <button
            onClick={handleNext}
            disabled={samples.length === 0}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition disabled:opacity-40"
          >
            <span>Next</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Active Sample Display Card */}
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-16 space-y-3">
            <RotateCcw className="w-8 h-8 text-sky-400 animate-spin" />
            <span className="text-xs text-slate-400">Loading benchmark samples...</span>
          </div>
        ) : activeSample ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Visual Column (5 cols) */}
            <div className="lg:col-span-5 space-y-4">
              <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden relative group">
                <div className="aspect-square w-full bg-slate-900 flex items-center justify-center relative">
                  {showHeatmap && activeSample.heatmap_base64 ? (
                    <img
                      src={activeSample.heatmap_base64}
                      alt="Grad-CAM Saliency Heatmap"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <img
                      src={activeSample.image_url}
                      alt="Fakeddit external benchmark media"
                      className="w-full h-full object-cover"
                    />
                  )}
                </div>

                {/* Heatmap overlay toggle badge */}
                {activeSample.heatmap_base64 && (
                  <div className="absolute top-3 right-3">
                    <button
                      onClick={() => setShowHeatmap(!showHeatmap)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold shadow-lg transition flex items-center space-x-1.5 ${
                        showHeatmap
                          ? 'bg-rose-500 text-white border border-rose-400'
                          : 'bg-slate-900/90 text-slate-200 border border-slate-700 hover:bg-slate-800'
                      }`}
                    >
                      <Layers className="w-3.5 h-3.5" />
                      <span>{showHeatmap ? 'Grad-CAM Heatmap' : 'View Grad-CAM'}</span>
                    </button>
                  </div>
                )}

                <div className="p-3 bg-slate-900/90 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                  <div className="flex items-center space-x-1.5">
                    <ImageIcon className="w-3.5 h-3.5 text-sky-400" />
                    <span>ResNet-18 (224x224 RGB)</span>
                  </div>
                  <span className="font-mono text-[10px] text-slate-500 uppercase">
                    Taxonomy: {activeSample.manipulation_type}
                  </span>
                </div>
              </div>
            </div>

            {/* Content & Evaluation Column (7 cols) */}
            <div className="lg:col-span-7 space-y-6">
              {/* Text Claim / Caption */}
              <div className="space-y-2">
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-400">
                  <FileText className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Text Claim / Social Media Caption</span>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-slate-100 font-medium text-base leading-relaxed">
                  "{activeSample.text}"
                </div>
              </div>

              {/* Status and Decision Badges */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[11px] text-slate-400 block font-medium">Ground Truth</span>
                  <span
                    className={`text-sm font-extrabold mt-1 inline-block px-2.5 py-0.5 rounded-md ${
                      activeSample.ground_truth === 'REAL'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    }`}
                  >
                    {activeSample.ground_truth}
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[11px] text-slate-400 block font-medium">TruthLens Prediction</span>
                  <span
                    className={`text-sm font-extrabold mt-1 inline-block px-2.5 py-0.5 rounded-md ${
                      activeSample.prediction === 'REAL'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : activeSample.prediction === 'FAKE'
                        ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                        : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    }`}
                  >
                    {activeSample.prediction}
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[11px] text-slate-400 block font-medium">Verdict Status</span>
                  <span
                    className={`text-xs font-bold mt-1 inline-flex items-center space-x-1 px-2.5 py-1 rounded-md ${
                      activeSample.status === 'CORRECT'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : activeSample.status === 'UNCERTAIN'
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                        : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                    }`}
                  >
                    {activeSample.status === 'CORRECT' && <CheckCircle2 className="w-3.5 h-3.5" />}
                    {activeSample.status === 'UNCERTAIN' && <AlertTriangle className="w-3.5 h-3.5" />}
                    {activeSample.status === 'INCORRECT' && <XCircle className="w-3.5 h-3.5" />}
                    <span>{activeSample.status}</span>
                  </span>
                </div>
              </div>

              {/* Confidence & Probability Meter */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold">
                  <span className="text-slate-400">Model Confidence: {(activeSample.confidence * 100).toFixed(1)}%</span>
                  <div className="space-x-3 text-slate-300">
                    <span className="text-emerald-400">REAL: {(activeSample.real_probability * 100).toFixed(1)}%</span>
                    <span className="text-rose-400">FAKE: {(activeSample.fake_probability * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden flex">
                  <div
                    className="bg-emerald-500 h-full transition-all duration-300"
                    style={{ width: `${activeSample.real_probability * 100}%` }}
                  />
                  <div
                    className="bg-rose-500 h-full transition-all duration-300"
                    style={{ width: `${activeSample.fake_probability * 100}%` }}
                  />
                </div>
              </div>

              {/* Multimodal Evidence & Explainability */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-400">
                  <Sparkles className="w-3.5 h-3.5 text-sky-400" />
                  <span>Multimodal Evidence & Cross-Modal Reasoning</span>
                </div>

                {/* Text Token Saliency */}
                {activeSample.token_saliency && activeSample.token_saliency.length > 0 && (
                  <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl space-y-1.5">
                    <span className="text-[11px] font-semibold text-slate-400 block">
                      Token Gradient Saliency (Influential Words):
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {activeSample.token_saliency.map((t, i) => (
                        <span
                          key={i}
                          className="px-2 py-0.5 rounded text-xs font-mono font-medium bg-slate-800 text-sky-300 border border-slate-700"
                        >
                          {t.token} <span className="text-[10px] text-slate-500">({t.score.toFixed(2)})</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Evidence Descriptions */}
                {activeSample.evidence_descriptions && activeSample.evidence_descriptions.length > 0 && (
                  <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl space-y-1.5 text-xs text-slate-300">
                    <span className="text-[11px] font-semibold text-slate-400 block">
                      Cross-Modal Attention Analysis:
                    </span>
                    <ul className="list-disc list-inside space-y-1 text-slate-300">
                      {activeSample.evidence_descriptions.map((desc, i) => (
                        <li key={i}>{desc}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Explanation text */}
                {activeSample.explanation && (
                  <p className="text-xs text-slate-400 italic bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/60">
                    "{activeSample.explanation}"
                  </p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-12 text-slate-500 text-sm">
            No samples match the selected filter.
          </div>
        )}
      </div>
    </div>
  );
};

export default FakedditBenchmarkPage;
