import { VerificationReport, ModelInfo, DatasetStats, HealthInfo } from '../types';

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

export async function verifyText(text: string): Promise<VerificationReport> {
  const res = await fetch(`${API_BASE}/verify/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
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

export async function verifyMultimodal(
  text?: string,
  imageFile?: File,
  videoFile?: File
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
