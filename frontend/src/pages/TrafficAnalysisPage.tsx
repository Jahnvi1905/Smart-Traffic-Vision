import React, { useState, useRef, useMemo } from 'react';
import {
  Upload,
  Sliders,
  CheckCircle2,
  FileVideo,
  FileImage,
  ShieldCheck,
} from 'lucide-react';
import { api, getFullMediaUrl } from '../api/client';
import type {
  AnalyzeVideoResponse,
  AnalyzeImageResponse,
  TelemetryItem,
} from '../api/types';
import { LoadingState } from '../components/LoadingState';
import { ErrorState } from '../components/ErrorState';

export const TrafficAnalysisPage: React.FC = () => {
  const [analysisType, setAnalysisType] = useState<'video' | 'image'>('video');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [useSample, setUseSample] = useState<boolean>(false);

  // Video Options
  const [maxFrames, setMaxFrames] = useState<number>(300);
  const [frameSkip, setFrameSkip] = useState<number>(1);
  const [enableCounter, setEnableCounter] = useState<boolean>(true);
  const [enableHelmet, setEnableHelmet] = useState<boolean>(true);
  const [enableWrongWay, setEnableWrongWay] = useState<boolean>(true);
  const [enableRedLight, setEnableRedLight] = useState<boolean>(false);
  const [expectedDirection, setExpectedDirection] = useState<string>('down');
  const [signalState, setSignalState] = useState<string>('GREEN');
  const [confThreshold, setConfThreshold] = useState<number>(0.35);

  // Execution & Results State
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [videoResult, setVideoResult] = useState<AnalyzeVideoResponse | null>(null);
  const [imageResult, setImageResult] = useState<AnalyzeImageResponse | null>(null);

  // Playback Telemetry Synchronization
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [currentPlaybackTime, setCurrentPlaybackTime] = useState<number>(0);

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0]);
      setUseSample(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
      setUseSample(false);
    }
  };

  const handleRunAnalysis = async () => {
    setIsProcessing(true);
    setError(null);

    try {
      if (analysisType === 'video') {
        const res = await api.analyzeVideo({
          file: useSample ? null : selectedFile,
          use_sample: useSample,
          max_frames: maxFrames > 0 ? maxFrames : null,
          frame_skip: frameSkip,
          enable_counter: enableCounter,
          enable_helmet: enableHelmet,
          enable_wrong_way: enableWrongWay,
          enable_red_light: enableRedLight,
          expected_direction: expectedDirection,
          signal_state: signalState,
          conf_threshold: confThreshold,
        });
        setVideoResult(res);
        setCurrentPlaybackTime(0);
      } else {
        if (!selectedFile) {
          throw new Error('Please select an image file to analyze.');
        }
        const res = await api.analyzeImage({
          file: selectedFile,
          conf_threshold: confThreshold,
        });
        setImageResult(res);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Find telemetry frame matching current playback time
  const activeTelemetry: TelemetryItem | null = useMemo(() => {
    if (!videoResult || !videoResult.telemetry || videoResult.telemetry.length === 0) {
      return null;
    }
    const tel = videoResult.telemetry;
    // Find closest frame at or before currentPlaybackTime
    let closest = tel[0];
    for (const item of tel) {
      if (item.timestamp_s <= currentPlaybackTime) {
        closest = item;
      } else {
        break;
      }
    }
    return closest;
  }, [videoResult, currentPlaybackTime]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* ── Mode Selection & Upload Area ── */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Sliders size={18} color="var(--accent-cyan)" />
            <span>Traffic Stream Ingestion &amp; Configuration</span>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => {
                setAnalysisType('video');
                setImageResult(null);
              }}
              className={`btn btn-sm ${analysisType === 'video' ? 'btn-primary' : 'btn-outline'}`}
            >
              <FileVideo size={14} /> Video Stream
            </button>
            <button
              onClick={() => {
                setAnalysisType('image');
                setVideoResult(null);
                setUseSample(false);
              }}
              className={`btn btn-sm ${analysisType === 'image' ? 'btn-primary' : 'btn-outline'}`}
            >
              <FileImage size={14} /> Photo Snapshot
            </button>
          </div>
        </div>

        <div className="grid-2">
          {/* Drag & Drop Upload Zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleFileDrop}
            style={{
              border: '2px dashed var(--border-muted)',
              borderRadius: 'var(--radius-lg)',
              backgroundColor: 'var(--bg-secondary)',
              padding: '30px 20px',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              cursor: 'pointer',
            }}
            onClick={() => document.getElementById('media-upload-input')?.click()}
          >
            <input
              id="media-upload-input"
              type="file"
              accept={analysisType === 'video' ? 'video/mp4,video/avi,video/quicktime' : 'image/jpeg,image/png'}
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />

            <Upload size={32} color="var(--accent-cyan)" />
            <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
              {selectedFile ? selectedFile.name : `Drop ${analysisType} file here, or click to browse`}
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              {analysisType === 'video' ? 'Supports MP4, AVI, MOV up to 100MB' : 'Supports JPG, PNG photos'}
            </p>

            {analysisType === 'video' && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setUseSample(true);
                  setSelectedFile(null);
                }}
                className={`btn btn-sm ${useSample ? 'btn-primary' : 'btn-secondary'}`}
                style={{ marginTop: '8px' }}
              >
                {useSample ? '✓ Benchmark sample.mp4 Selected' : 'Use Benchmark sample.mp4'}
              </button>
            )}
          </div>

          {/* Configuration Parameters */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {analysisType === 'video' ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div className="form-group" style={{ margin: 0 }}>
                    <label className="form-label">Frames Limit (CPU Safe)</label>
                    <select
                      className="select"
                      value={maxFrames}
                      onChange={(e) => setMaxFrames(Number(e.target.value))}
                    >
                      <option value={150}>150 Frames (~5s quick check)</option>
                      <option value={300}>300 Frames (~10s standard)</option>
                      <option value={600}>600 Frames (~20s extended)</option>
                      <option value={0}>Full Video Duration</option>
                    </select>
                  </div>

                  <div className="form-group" style={{ margin: 0 }}>
                    <label className="form-label">Frame Skip Interval</label>
                    <select
                      className="select"
                      value={frameSkip}
                      onChange={(e) => setFrameSkip(Number(e.target.value))}
                    >
                      <option value={1}>Every Frame (Highest Accuracy)</option>
                      <option value={2}>Every 2nd Frame (2x Speed)</option>
                    </select>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div className="form-group" style={{ margin: 0 }}>
                    <label className="form-label">Expected Direction (Wrong-Way)</label>
                    <select
                      className="select"
                      value={expectedDirection}
                      onChange={(e) => setExpectedDirection(e.target.value)}
                    >
                      <option value="down">Downwards Flow (↓)</option>
                      <option value="up">Upwards Flow (↑)</option>
                      <option value="left">Leftwards Flow (←)</option>
                      <option value="right">Rightwards Flow (→)</option>
                    </select>
                  </div>

                  <div className="form-group" style={{ margin: 0 }}>
                    <label className="form-label">Virtual Stop-Line Signal</label>
                    <select
                      className="select"
                      value={signalState}
                      onChange={(e) => setSignalState(e.target.value)}
                    >
                      <option value="GREEN">GREEN (Pass Allowed)</option>
                      <option value="RED">RED (Enforce Stop-Line)</option>
                    </select>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginTop: '4px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={enableCounter}
                      onChange={(e) => setEnableCounter(e.target.checked)}
                    />
                    Virtual Line Counter
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={enableHelmet}
                      onChange={(e) => setEnableHelmet(e.target.checked)}
                    />
                    Helmet Compliance
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={enableWrongWay}
                      onChange={(e) => setEnableWrongWay(e.target.checked)}
                    />
                    Wrong-Way Detector
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={enableRedLight}
                      onChange={(e) => setEnableRedLight(e.target.checked)}
                    />
                    Stop-Line Policy
                  </label>
                </div>
              </>
            ) : (
              <div className="form-group">
                <label className="form-label">Detection Confidence Threshold: {confThreshold}</label>
                <input
                  type="range"
                  min="0.1"
                  max="0.8"
                  step="0.05"
                  value={confThreshold}
                  onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>
            )}

            <button
              onClick={handleRunAnalysis}
              disabled={isProcessing || (!useSample && !selectedFile)}
              className="btn btn-primary"
              style={{ marginTop: 'auto', width: '100%', padding: '10px' }}
            >
              {isProcessing ? 'Processing Vision Pipeline...' : `Execute ${analysisType === 'video' ? 'Video' : 'Photo'} Analysis`}
            </button>
          </div>
        </div>
      </div>

      {/* ── Progress / Error Notification ── */}
      {isProcessing && (
        <LoadingState
          message="Running YOLO detection, multi-object tracking, and safety rule validation..."
          minHeight="140px"
        />
      )}

      {error && (
        <ErrorState
          title="Analysis Execution Failed"
          message={error}
          onRetry={handleRunAnalysis}
        />
      )}

      {/* ── Video Analysis Results with Synchronized Telemetry ── */}
      {videoResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <CheckCircle2 size={18} color="var(--color-success)" />
                <span>Video Analysis Output &amp; Real-Time Playback Synchronization</span>
              </div>
              <span className="badge badge-success">Pipeline Complete</span>
            </div>

            <div className="grid-2">
              {/* Processed Video Player */}
              <div>
                <video
                  ref={videoRef}
                  src={getFullMediaUrl(videoResult.video_url)}
                  controls
                  style={{
                    width: '100%',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: '#000',
                    border: '1px solid var(--border-subtle)',
                  }}
                  onTimeUpdate={() => {
                    if (videoRef.current) {
                      setCurrentPlaybackTime(videoRef.current.currentTime);
                    }
                  }}
                  onSeeked={() => {
                    if (videoRef.current) {
                      setCurrentPlaybackTime(videoRef.current.currentTime);
                    }
                  }}
                />
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginTop: '8px',
                    fontSize: '12px',
                    color: 'var(--text-muted)',
                  }}
                >
                  <div>
                    Playback Time:{' '}
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                      {currentPlaybackTime.toFixed(2)}s
                    </span>
                  </div>
                  <div>
                    Processed Frames:{' '}
                    <span style={{ fontFamily: 'var(--font-mono)' }}>{videoResult.frames_processed}</span>
                  </div>
                </div>
              </div>

              {/* Synchronized Telemetry Readout */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  SYNCHRONIZED COUNTS AT T = {currentPlaybackTime.toFixed(2)}s
                </div>

                <div className="grid-2">
                  <div
                    style={{
                      padding: '14px',
                      backgroundColor: 'var(--bg-secondary)',
                      borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      Currently Visible Vehicles
                    </div>
                    <div
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '24px',
                        fontWeight: 700,
                        color: 'var(--accent-cyan)',
                        marginTop: '4px',
                      }}
                    >
                      {activeTelemetry?.visible_total ?? 0}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      In-frame detection count
                    </div>
                  </div>

                  <div
                    style={{
                      padding: '14px',
                      backgroundColor: 'var(--bg-secondary)',
                      borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      Cumulative Line Crossings
                    </div>
                    <div
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '24px',
                        fontWeight: 700,
                        color: 'var(--color-success)',
                        marginTop: '4px',
                      }}
                    >
                      {activeTelemetry?.cumulative_total ?? videoResult.total_counted}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Virtual line crossings
                    </div>
                  </div>
                </div>

                {/* Per-class breakdown at this playback moment */}
                <div
                  style={{
                    padding: '14px',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '10px' }}>
                    Active Classification Breakdown
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
                    {Object.entries(activeTelemetry?.cumulative_classes || videoResult.class_counts).map(
                      ([cat, cnt]) => (
                        <div
                          key={cat}
                          style={{
                            padding: '8px 10px',
                            backgroundColor: 'var(--bg-surface)',
                            borderRadius: 'var(--radius-sm)',
                            border: '1px solid var(--border-subtle)',
                          }}
                        >
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                            {cat.replace('-', ' ')}
                          </div>
                          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '15px', fontWeight: 600 }}>
                            {cnt}
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>

                {/* Safety Violations Summary */}
                <div
                  style={{
                    padding: '12px 14px',
                    backgroundColor:
                      videoResult.violations_logged > 0 ? 'var(--color-danger-bg)' : 'var(--color-success-bg)',
                    borderRadius: 'var(--radius-md)',
                    border: `1px solid ${
                      videoResult.violations_logged > 0 ? 'var(--color-danger-border)' : 'var(--color-success-border)'
                    }`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <ShieldCheck
                      size={18}
                      color={videoResult.violations_logged > 0 ? 'var(--color-danger)' : 'var(--color-success)'}
                    />
                    <span style={{ fontSize: '13px', fontWeight: 500 }}>
                      {videoResult.violations_logged > 0
                        ? `${videoResult.violations_logged} safety violation(s) flagged & recorded to SQLite.`
                        : 'No safety violations logged during this video sequence.'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Image Analysis Results ── */}
      {imageResult && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <CheckCircle2 size={18} color="var(--color-success)" />
              <span>Snapshot Localization Results ({imageResult.total_detected} Vehicles Detected)</span>
            </div>
          </div>

          <div className="grid-2">
            <div>
              <img
                src={getFullMediaUrl(imageResult.annotated_image_url)}
                alt="Annotated Traffic Detections"
                style={{
                  width: '100%',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                }}
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Detection Breakdown
              </div>

              <div className="table-container" style={{ maxHeight: '340px' }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Class</th>
                      <th>Confidence</th>
                      <th>Bounding Box</th>
                    </tr>
                  </thead>
                  <tbody>
                    {imageResult.detections.map((det, idx) => (
                      <tr key={idx}>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{idx + 1}</td>
                        <td style={{ fontWeight: 600 }}>{det.label}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{(det.confidence * 100).toFixed(1)}%</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px', color: 'var(--text-muted)' }}>
                          [{det.box.join(', ')}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
