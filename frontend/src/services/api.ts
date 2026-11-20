import {
  VerificationReport,
  ModelInfo,
  DatasetStats,
  HealthInfo,
  VerificationListResponse,
} from '../types';

const API_BASE = '/api';

export async function checkHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) throw new Error(`Model info fetch failed: ${res.statusText}`);
  return res.json();
}

export async function getDatasetStats(): Promise<DatasetStats> {
  const res = await fetch(`${API_BASE}/dataset/stats`);
  if (!res.ok) throw new Error(`Dataset stats fetch failed: ${res.statusText}`);
  return res.json();
}

export async function getVerifications(limit = 100, offset = 0): Promise<VerificationListResponse> {
  const res = await fetch(`${API_BASE}/verifications?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to fetch history: ${res.statusText}`);
  return res.json();
}

export async function getVerificationById(id: string): Promise<VerificationReport> {
  const res = await fetch(`${API_BASE}/verifications/${id}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to fetch verification ${id}`);
  }
  return res.json();
}

export async function deleteVerification(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/verifications/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete verification ${id}`);
}

export async function verifyText(text: string, caseName?: string): Promise<VerificationReport> {
  const res = await fetch(`${API_BASE}/verify/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, case_name: caseName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Text verification failed');
  }
  return res.json();
}

export async function verifyImage(imageFile: File): Promise<VerificationReport> {
  const formData = new FormData();
  formData.append('image', imageFile);

  const res = await fetch(`${API_BASE}/verify/image`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Image verification failed');
  }
  return res.json();
}

export async function verifyAudio(audioFile: File): Promise<VerificationReport> {
  const formData = new FormData();
  formData.append('audio', audioFile);

  const res = await fetch(`${API_BASE}/verify/audio`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Audio verification failed');
  }
  return res.json();
}

export async function verifyMultimodal(
  text?: string,
  imageFile?: File,
  videoFile?: File,
  audioFile?: File,
  caseName?: string
): Promise<VerificationReport> {
  const formData = new FormData();
  if (text && text.trim()) {
    formData.append('text', text.trim());
  }
  if (imageFile) {
    formData.append('image', imageFile);
  }
  if (videoFile) {
    formData.append('video', videoFile);
  }
  if (audioFile) {
    formData.append('audio', audioFile);
  }
  if (caseName && caseName.trim()) {
    formData.append('case_name', caseName.trim());
  }

  const res = await fetch(`${API_BASE}/verify/multimodal`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Multimodal verification failed');
  }
  return res.json();
}

export async function getEvaluationOverview(): Promise<any> {
  const res = await fetch(`${API_BASE}/evaluation/overview`);
  if (!res.ok) throw new Error(`Failed to fetch evaluation overview: ${res.statusText}`);
  return res.json();
}

export async function getEvaluationAll(): Promise<any> {
  const res = await fetch(`${API_BASE}/evaluation/all`);
  if (!res.ok) throw new Error(`Failed to fetch evaluation hub data: ${res.statusText}`);
  return res.json();
}
