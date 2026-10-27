// Exact TypeScript type definitions conforming to backend api/schemas.py

export interface HealthResponse {
  status: string;
  app: string;
  version: string;
  timestamp: string;
}

export interface ModelDiagnostic {
  path: string;
  exists: boolean;
  size_mb: number;
}

export interface DatabaseDiagnostic {
  db_path: string;
  exists: boolean;
  connected: boolean;
  table_exists: boolean;
  total_events: number;
  total_violations: number;
}

export interface DirectoryDiagnostic {
  exists: boolean;
  writable: boolean;
}

export interface OpenH264Diagnostic {
  path: string;
  exists: boolean;
  size_kb: number;
}

export interface EnvironmentDiagnostic {
  platform: string;
  python_version: string;
  torch_version: string | null;
  cuda_available: boolean;
}

export interface DiagnosticsResponse {
  status: string;
  models: Record<string, ModelDiagnostic>;
  database: DatabaseDiagnostic;
  directories: Record<string, DirectoryDiagnostic>;
  openh264: OpenH264Diagnostic;
  environment: EnvironmentDiagnostic;
}

export interface RecentEventItem {
  id: number;
  timestamp: string;
  track_id?: number | null;
  vehicle_class?: string | null;
  event_type: string;
  confidence?: number | null;
  plate_text?: string | null;
  evidence_url?: string | null;
  review_status: string;
  count_details?: string | null;
  total_count?: number | null;
}

export interface OverviewStatsResponse {
  total_vehicles: number;
  total_violations: number;
  violations_by_type: Record<string, number>;
  category_breakdown: Record<string, number>;
  recent_events: RecentEventItem[];
}

export interface ViolationItem {
  id: number;
  timestamp: string;
  track_id?: number | null;
  vehicle_class?: string | null;
  event_type: string;
  confidence?: number | null;
  plate_text?: string | null;
  evidence_url?: string | null;
  review_status: string;
  count_details?: string | null;
  total_count?: number | null;
}

export interface ViolationsListResponse {
  total: number;
  page: number;
  limit: number;
  pages: number;
  items: ViolationItem[];
}

export interface ReviewStatusResponse {
  id: number;
  review_status: string;
  updated: boolean;
}

export interface TelemetryItem {
  frame_idx: number;
  timestamp_s: number;
  cumulative_total: number;
  cumulative_classes: Record<string, number>;
  visible_total: number;
  visible_classes: Record<string, number>;
  newly_counted: number[];
}

export interface AnalyzeVideoResponse {
  status: string;
  video_url: string;
  total_counted: number;
  class_counts: Record<string, number>;
  violations_logged: number;
  telemetry: TelemetryItem[];
  frames_processed: number;
}

export interface DetectionBoxItem {
  class_name: string;
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface AnalyzeImageResponse {
  status: string;
  annotated_image_url: string;
  total_detected: number;
  class_counts: Record<string, number>;
  detections: DetectionBoxItem[];
}

export interface PlateItem {
  idx: number;
  plate_text: string;
  raw_text?: string | null;
  confidence: number;
  is_low_confidence?: boolean;
  confidence_status?: 'high' | 'medium' | 'low' | 'unreadable' | string;
  crop_url?: string | null;
  box: [number, number, number, number];
}

export interface PlateScanResponse {
  status: string;
  annotated_image_url?: string | null;
  total_plates: number;
  plates: PlateItem[];
}

export interface VideoAnalysisOptions {
  file?: File | null;
  use_sample?: boolean;
  max_frames?: number | null;
  frame_skip?: number;
  enable_counter?: boolean;
  enable_helmet?: boolean;
  enable_wrong_way?: boolean;
  enable_red_light?: boolean;
  expected_direction?: string;
  signal_state?: string;
  conf_threshold?: number;
}

export interface ImageAnalysisOptions {
  file: File;
  conf_threshold?: number;
}

export interface PlateScanOptions {
  file?: File | null;
  use_sample?: boolean;
  save_to_db?: boolean;
}
