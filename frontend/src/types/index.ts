export interface TokenImportance {
  token: string;
  score: number;
}

export interface ModalityEvidence {
  modality: 'text' | 'visual' | 'video' | 'audio' | 'cross-modal';
  type: 'model-derived' | 'heuristic';
  description: string;
  details?: Record<string, any>;
  token_importance?: TokenImportance[];
  heatmap_available?: boolean;
  heatmap_base64?: string;
}

export interface VerificationReport {
  id?: string;
  case_name?: string;
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

export interface VerificationSummary {
  id: string;
  case_name: string;
  created_at: string;
  prediction: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;
  real_probability: number;
  fake_probability: number;
  uncertain: boolean;
  modalities_used: string[];
  manipulation_type: string;
  text_snippet?: string;
  has_image: boolean;
  has_video: boolean;
  has_audio: boolean;
  execution_mode: string;
}

export interface VerificationListResponse {
  total: number;
  items: VerificationSummary[];
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

export interface EvaluationOverviewMetrics {
  in_domain_accuracy: number;
  in_domain_f1: number;
  in_domain_source: string;
  fakeddit_multimodal_accuracy: number;
  fakeddit_text_adapted_accuracy: number;
  fakeddit_source: string;
  semantic_mismatch_accuracy: number;
  semantic_mismatch_c2_accuracy: number;
  semantic_mismatch_source: string;
  audio_roc_auc: number;
  audio_source: string;
  video_roc_auc: number;
  video_source: string;
  calibrated_ece: number;
  calibrated_ece_source: string;
  total_automated_tests: number;
}

export interface EvaluationOverview {
  status: string;
  selected_model: string;
  selected_model_name: string;
  checkpoint: string;
  temperature: number;
  uncertainty_threshold: number;
  metrics: EvaluationOverviewMetrics;
}

export interface AllEvaluationData {
  overview: EvaluationOverview;
  in_domain: any;
  generalization: any;
  semantic_mismatch: any;
  calibration: any;
  ablation: any;
}

export interface FakedditBenchmarkResults {
  status: string;
  dataset: string;
  evaluation_samples: number;
  training_source: string;
  training_samples: number;
  fakeddit_used_for_training: boolean;
  fakeddit_used_for_external_evaluation: boolean;
  class_distribution: { real: number; fake: number };
  accuracy: number;
  resolved_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  counts: {
    correct: number;
    incorrect: number;
    uncertain: number;
  };
  rates: {
    uncertain_rate: number;
    accuracy_percentage: number;
  };
  confusion_matrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
  };
}

export interface FakedditSample {
  sample_id: string;
  text: string;
  ground_truth: 'REAL' | 'FAKE';
  prediction: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;
  real_probability: number;
  fake_probability: number;
  correct: boolean;
  status: 'CORRECT' | 'INCORRECT' | 'UNCERTAIN';
  manipulation_type: string;
  image_url: string;
  explanation?: string;
  evidence_descriptions?: string[];
  token_saliency?: { token: string; score: number }[];
  has_heatmap?: boolean;
  heatmap_base64?: string;
}

export interface FakedditSamplesResponse {
  status: string;
  total_available: number;
  filter_applied: string;
  count: number;
  samples: FakedditSample[];
}
