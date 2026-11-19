import React, { useEffect, useState } from 'react';
import {
  History,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Eye,
  Trash2,
  RefreshCw,
  Search,
  Filter,
  X,
  FileText,
  Clock,
} from 'lucide-react';
import { getVerifications, getVerificationById, deleteVerification } from '../services/api';
import { VerificationSummary, VerificationReport } from '../types';
import { PredictionCard } from '../components/PredictionCard';
import { EvidenceSection } from '../components/EvidenceSection';

export const HistoryPage: React.FC = () => {
  const [items, setItems] = useState<VerificationSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [search, setSearch] = useState<string>('');
  const [verdictFilter, setVerdictFilter] = useState<'ALL' | 'REAL' | 'FAKE' | 'UNCERTAIN'>('ALL');

  // Modal Report inspection
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeReport, setActiveReport] = useState<VerificationReport | null>(null);
  const [modalLoading, setModalLoading] = useState<boolean>(false);
  const [modalError, setModalError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getVerifications(100, 0);
      setItems(res.items);
    } catch (err: any) {
      setError(err.message || 'Failed to load verification history.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleOpenReport = async (id: string) => {
    setSelectedId(id);
    setActiveReport(null);
    setModalLoading(true);
    setModalError(null);

    try {
      const report = await getVerificationById(id);
      setActiveReport(report);
    } catch (err: any) {
      setModalError(err.message || `Failed to fetch report for verification #${id}`);
    } finally {
      setModalLoading(false);
    }
  };

  const handleCloseModal = () => {
    setSelectedId(null);
    setActiveReport(null);
    setModalError(null);
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Delete this verification record from persistent history?')) return;

    try {
      await deleteVerification(id);
      setItems((prev) => prev.filter((item) => item.id !== id));
      if (selectedId === id) {
        handleCloseModal();
      }
    } catch (err: any) {
      alert(`Could not delete verification: ${err.message}`);
    }
  };

  // Filtered list
  const filteredItems = items.filter((item) => {
    const matchesSearch =
      item.case_name.toLowerCase().includes(search.toLowerCase()) ||
      (item.text_snippet && item.text_snippet.toLowerCase().includes(search.toLowerCase())) ||
      item.id.toLowerCase().includes(search.toLowerCase());

    const matchesVerdict = verdictFilter === 'ALL' || item.prediction === verdictFilter;
    return matchesSearch && matchesVerdict;
  });

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-2 rounded-xl bg-sky-500/10 text-sky-400 border border-sky-500/20">
              <History className="w-6 h-6" />
            </span>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">Verification History</h1>
          </div>
          <p className="text-slate-400 text-sm mt-1">
            Persistent archive of all multimodal verifications stored in SQLite. Reopen any report to view Grad-CAM overlays and token attributions.
          </p>
        </div>

        <button
          onClick={fetchHistory}
          disabled={loading}
          className="px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 flex items-center space-x-2 text-xs font-semibold transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-sky-400' : ''}`} />
          <span>Refresh Records</span>
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-panel rounded-2xl p-4 sm:p-5 border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Search Input */}
        <div className="relative w-full sm:w-96">
          <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by case title, snippet, or ID..."
            className="w-full pl-10 pr-4 py-2 rounded-xl bg-slate-900 border border-slate-800 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 text-xs text-slate-200 placeholder:text-slate-500"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-3 top-2.5 text-slate-500 hover:text-slate-300"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Verdict Filter Buttons */}
        <div className="flex items-center space-x-1.5 w-full sm:w-auto overflow-x-auto pb-1 sm:pb-0">
          <span className="text-xs text-slate-500 font-medium mr-1 flex items-center">
            <Filter className="w-3 h-3 mr-1" />
            Filter:
          </span>
          {(['ALL', 'REAL', 'FAKE', 'UNCERTAIN'] as const).map((v) => {
            const isActive = verdictFilter === v;
            return (
              <button
                key={v}
                onClick={() => setVerdictFilter(v)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                  isActive
                    ? v === 'REAL'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                      : v === 'FAKE'
                      ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                      : v === 'UNCERTAIN'
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                      : 'bg-sky-500/20 text-sky-400 border border-sky-500/40'
                    : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                }`}
              >
                {v}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Table or Card List */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="py-20 text-center text-slate-400 text-sm space-y-3">
          <div className="w-8 h-8 border-2 border-sky-400 border-t-transparent rounded-full animate-spin mx-auto" />
          <span>Retrieving persistent verification records...</span>
        </div>
      ) : filteredItems.length === 0 ? (
        <div className="glass-panel rounded-2xl p-12 text-center border border-slate-800 space-y-3">
          <History className="w-10 h-10 text-slate-600 mx-auto mb-2" />
          <h3 className="text-base font-bold text-slate-200">No Verifications Found</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            {search || verdictFilter !== 'ALL'
              ? 'No verification records match your active search and filter criteria.'
              : 'You have not performed any content verifications yet. Visit the "Verify Content" tab to analyze multimodal claims.'}
          </p>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <tr>
                  <th className="py-3.5 px-4">Case / Verification</th>
                  <th className="py-3.5 px-4">Verdict</th>
                  <th className="py-3.5 px-4">Confidence</th>
                  <th className="py-3.5 px-4">Modalities</th>
                  <th className="py-3.5 px-4">Timestamp</th>
                  <th className="py-3.5 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {filteredItems.map((item) => {
                  const isReal = item.prediction === 'REAL';
                  const isFake = item.prediction === 'FAKE';
                  const badgeColor = isReal
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    : isFake
                    ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                    : 'bg-amber-500/10 text-amber-400 border-amber-500/30';
                  const Icon = isReal ? CheckCircle2 : isFake ? AlertTriangle : HelpCircle;

                  const dateFormatted = item.created_at
                    ? new Date(item.created_at).toLocaleString(undefined, {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })
                    : 'N/A';

                  return (
                    <tr
                      key={item.id}
                      onClick={() => handleOpenReport(item.id)}
                      className="hover:bg-slate-800/40 transition cursor-pointer group"
                    >
                      {/* Case Name & Snippet */}
                      <td className="py-3.5 px-4 max-w-xs sm:max-w-md">
                        <div className="font-semibold text-white group-hover:text-sky-400 transition truncate">
                          {item.case_name}
                        </div>
                        {item.text_snippet && (
                          <div className="text-[11px] text-slate-500 truncate mt-0.5">
                            {item.text_snippet}
                          </div>
                        )}
                        <span className="text-[10px] text-slate-600 font-mono">ID: {item.id}</span>
                      </td>

                      {/* Verdict Badge */}
                      <td className="py-3.5 px-4 whitespace-nowrap">
                        <span className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold border ${badgeColor}`}>
                          <Icon className="w-3.5 h-3.5" />
                          <span>{item.prediction}</span>
                        </span>
                      </td>

                      {/* Confidence */}
                      <td className="py-3.5 px-4 whitespace-nowrap">
                        <span className="font-extrabold text-sm text-sky-400">
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                      </td>

                      {/* Modalities */}
                      <td className="py-3.5 px-4">
                        <div className="flex gap-1 flex-wrap">
                          {item.modalities_used.map((mod) => (
                            <span
                              key={mod}
                              className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 text-[10px] uppercase font-medium"
                            >
                              {mod}
                            </span>
                          ))}
                        </div>
                      </td>

                      {/* Timestamp */}
                      <td className="py-3.5 px-4 whitespace-nowrap text-slate-500 flex items-center space-x-1">
                        <Clock className="w-3 h-3 text-slate-600 shrink-0" />
                        <span>{dateFormatted}</span>
                      </td>

                      {/* Action Buttons */}
                      <td className="py-3.5 px-4 whitespace-nowrap text-right">
                        <div className="flex items-center justify-end space-x-2">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleOpenReport(item.id);
                            }}
                            className="px-2.5 py-1.5 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 border border-sky-500/20 flex items-center space-x-1 font-semibold text-xs transition"
                            title="Open full saved verification report"
                          >
                            <Eye className="w-3.5 h-3.5" />
                            <span>View Report</span>
                          </button>
                          <button
                            type="button"
                            onClick={(e) => handleDelete(item.id, e)}
                            className="p-1.5 rounded-lg hover:bg-rose-500/20 text-slate-500 hover:text-rose-400 transition"
                            title="Delete record"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Complete Report Reopen Modal */}
      {selectedId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fadeIn">
          <div className="glass-panel max-w-4xl w-full max-h-[90vh] overflow-y-auto rounded-3xl p-6 sm:p-8 border border-slate-700 shadow-2xl relative space-y-6">
            {/* Modal Header Bar */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  <FileText className="w-5 h-5" />
                </span>
                <div>
                  <h3 className="text-base font-bold text-white">Archived Verification Report</h3>
                  <span className="text-[11px] text-slate-400 font-mono">Record ID: {selectedId}</span>
                </div>
              </div>

              <button
                onClick={handleCloseModal}
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            {modalLoading ? (
              <div className="py-24 text-center text-slate-400 text-sm space-y-3">
                <div className="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin mx-auto" />
                <span>Reloading report from SQLite archive...</span>
              </div>
            ) : modalError ? (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm">
                {modalError}
              </div>
            ) : activeReport ? (
              <div className="space-y-6">
                <PredictionCard report={activeReport} />
                <EvidenceSection evidence={activeReport.evidence} />
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};
