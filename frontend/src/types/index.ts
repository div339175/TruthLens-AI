export interface TokenImportance {
  token: string;
  score: number;
}

export interface ModalityEvidence {
  modality: 'text' | 'visual' | 'video' | 'cross-modal';
  type: 'model-derived' | 'heuristic';
  description: string;
  details?: Record<string, any>;
  token_importance?: TokenImportance[];
  heatmap_available?: boolean;
  heatmap_base64?: string;
}

export interface VerificationReport {
  prediction: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;
  real_probability: number;
  fake_probability: number;
  uncertain: boolean;
  modalities_used: string[];
  manipulation_type: string;
  evidence: ModalityEvidence[];
  explanation: string;
  model_version: string;
  fusion_type: string;
  device_used: string;
  execution_mode: 'real' | 'demo';
  timestamp: string;
}

export interface ModelInfo {
  project_name: string;
  version: string;
  execution_mode: string;
  text_backbone: string;
  vision_backbone: string;
  fusion_architecture: string;
  device: string;
  uncertainty_threshold: number;
  is_trained_checkpoint: boolean;
  checkpoint_path?: string;
  architecture_summary: Record<string, any>;
}

export interface DatasetStats {
  status: string;
  total_samples: number;
  real_count: number;
  fake_count: number;
  uncertain_count: number;
  train_count: number;
  val_count: number;
  test_count: number;
  sources: Record<string, number>;
  message?: string;
}

export interface HealthInfo {
  status: string;
  project: string;
  version: string;
  device: string;
  execution_mode: string;
  torch_version: string;
  models_loaded: boolean;
}
