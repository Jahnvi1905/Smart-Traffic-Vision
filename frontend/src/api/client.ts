import type {
  HealthResponse,
  DiagnosticsResponse,
  OverviewStatsResponse,
  ViolationsListResponse,
  ReviewStatusResponse,
  AnalyzeImageResponse,
  AnalyzeVideoResponse,
  PlateScanResponse,
  VideoAnalysisOptions,
  ImageAnalysisOptions,
  PlateScanOptions,
} from './types';

// Use environment variable if provided, or default to relative path (leveraging Vite dev server proxy)
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const MEDIA_BASE_URL = import.meta.env.VITE_MEDIA_BASE_URL || (import.meta.env.DEV ? '' : 'http://127.0.0.1:8000');

export function getFullMediaUrl(path?: string | null): string {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${MEDIA_BASE_URL}${cleanPath}`;
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = typeof errorData.detail === 'string' 
            ? errorData.detail 
            : JSON.stringify(errorData.detail);
        }
      } catch {
        // Response wasn't JSON
      }
      throw new Error(errorMessage);
    }

    return (await response.json()) as T;
  } catch (error: unknown) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('An unexpected network error occurred.');
  }
}

export const api = {
  // ── Health & Diagnostics ──
  getHealth: (): Promise<HealthResponse> => {
    return request<HealthResponse>('/api/health');
  },

  getDiagnostics: (): Promise<DiagnosticsResponse> => {
    return request<DiagnosticsResponse>('/api/diagnostics');
  },

  // ── Overview Stats ──
  getOverviewStats: (): Promise<OverviewStatsResponse> => {
    return request<OverviewStatsResponse>('/api/stats/overview');
  },

  // ── Violations Management ──
  getViolations: (params: {
    event_type?: string;
    review_status?: string;
    search?: string;
    include_counts?: boolean;
    page?: number;
    limit?: number;
  } = {}): Promise<ViolationsListResponse> => {
    const query = new URLSearchParams();
    if (params.event_type && params.event_type !== 'ALL') {
      query.set('event_type', params.event_type);
    }
    if (params.review_status && params.review_status !== 'ALL') {
      query.set('review_status', params.review_status);
    }
    if (params.search) {
      query.set('search', params.search);
    }
    if (params.include_counts !== undefined) {
      query.set('include_counts', params.include_counts.toString());
    }
    if (params.page) {
      query.set('page', params.page.toString());
    }
    if (params.limit) {
      query.set('limit', params.limit.toString());
    }
    const qs = query.toString();
    return request<ViolationsListResponse>(`/api/violations${qs ? `?${qs}` : ''}`);
  },

  updateViolationReview: (
    eventId: number,
    status: 'pending' | 'approved' | 'rejected'
  ): Promise<ReviewStatusResponse> => {
    return request<ReviewStatusResponse>(`/api/violations/${eventId}/review`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ status }),
    });
  },

  // ── Reports & Export ──
  getReportsExportUrl: (params: {
    event_type?: string;
    review_status?: string;
    include_counts?: boolean;
  } = {}): string => {
    const query = new URLSearchParams();
    if (params.event_type && params.event_type !== 'ALL') {
      query.set('event_type', params.event_type);
    }
    if (params.review_status && params.review_status !== 'ALL') {
      query.set('review_status', params.review_status);
    }
    if (params.include_counts !== undefined) {
      query.set('include_counts', params.include_counts.toString());
    }
    const qs = query.toString();
    return `${BASE_URL}/api/reports/export${qs ? `?${qs}` : ''}`;
  },

  // ── Traffic Analysis ──
  analyzeImage: async (options: ImageAnalysisOptions): Promise<AnalyzeImageResponse> => {
    const formData = new FormData();
    formData.append('file', options.file);
    if (options.conf_threshold !== undefined) {
      formData.append('conf_threshold', options.conf_threshold.toString());
    }

    return request<AnalyzeImageResponse>('/api/traffic/analyze-image', {
      method: 'POST',
      body: formData,
    });
  },

  analyzeVideo: async (options: VideoAnalysisOptions): Promise<AnalyzeVideoResponse> => {
    const formData = new FormData();

    if (options.use_sample) {
      formData.append('use_sample', 'true');
    } else if (options.file) {
      formData.append('file', options.file);
    }

    if (options.max_frames !== undefined && options.max_frames !== null) {
      formData.append('max_frames', options.max_frames.toString());
    }
    if (options.frame_skip !== undefined) {
      formData.append('frame_skip', options.frame_skip.toString());
    }
    if (options.enable_counter !== undefined) {
      formData.append('enable_counter', options.enable_counter.toString());
    }
    if (options.enable_helmet !== undefined) {
      formData.append('enable_helmet', options.enable_helmet.toString());
    }
    if (options.enable_wrong_way !== undefined) {
      formData.append('enable_wrong_way', options.enable_wrong_way.toString());
    }
    if (options.enable_red_light !== undefined) {
      formData.append('enable_red_light', options.enable_red_light.toString());
    }
    if (options.expected_direction) {
      formData.append('expected_direction', options.expected_direction);
    }
    if (options.signal_state) {
      formData.append('signal_state', options.signal_state);
    }
    if (options.conf_threshold !== undefined) {
      formData.append('conf_threshold', options.conf_threshold.toString());
    }

    return request<AnalyzeVideoResponse>('/api/traffic/analyze-video', {
      method: 'POST',
      body: formData,
    });
  },

  // ── Number Plate Scanner ──
  scanPlate: async (options: PlateScanOptions): Promise<PlateScanResponse> => {
    const formData = new FormData();
    if (options.use_sample) {
      formData.append('use_sample', 'true');
    } else if (options.file) {
      formData.append('file', options.file);
    }
    if (options.save_to_db !== undefined) {
      formData.append('save_to_db', options.save_to_db.toString());
    }

    return request<PlateScanResponse>('/api/plates/scan', {
      method: 'POST',
      body: formData,
    });
  },
};
