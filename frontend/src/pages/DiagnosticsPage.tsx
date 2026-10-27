import React, { useEffect, useState } from 'react';
import {
  Server,
  Database,
  CheckCircle2,
  XCircle,
  Cpu,
  FolderCheck,
  Film,
  RotateCcw,
} from 'lucide-react';
import { api } from '../api/client';
import type { DiagnosticsResponse } from '../api/types';
import { LoadingState } from '../components/LoadingState';
import { ErrorState } from '../components/ErrorState';

export const DiagnosticsPage: React.FC = () => {
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDiagnostics = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getDiagnostics();
      setDiagnostics(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to query system diagnostics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  if (loading) {
    return <LoadingState message="Running real-time diagnostics on models, database, and hardware..." />;
  }

  if (error || !diagnostics) {
    return (
      <ErrorState
        title="Diagnostics Query Failed"
        message={error || 'Unable to communicate with the FastAPI diagnostics endpoint.'}
        onRetry={fetchDiagnostics}
      />
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* ── System Status Banner ── */}
      <div
        className="card"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderLeft: `4px solid ${
            diagnostics.status === 'ok' ? 'var(--color-success)' : 'var(--color-warning)'
          }`,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: diagnostics.status === 'ok' ? 'var(--color-success-bg)' : 'var(--color-warning-bg)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: diagnostics.status === 'ok' ? 'var(--color-success)' : 'var(--color-warning)',
            }}
          >
            <Server size={22} />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 600, color: '#fff' }}>
              System Health Status: {diagnostics.status.toUpperCase()}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              All core services, database connections, and model endpoints verified.
            </div>
          </div>
        </div>

        <button onClick={fetchDiagnostics} className="btn btn-secondary btn-sm">
          <RotateCcw size={13} /> Refresh Diagnostics
        </button>
      </div>

      <div className="grid-2">
        {/* ── Model Weights Status ── */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Cpu size={17} color="var(--accent-cyan)" />
              <span>YOLO AI Model Weights</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {Object.entries(diagnostics.models).map(([key, info]) => (
              <div
                key={key}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '12px 14px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <div>
                  <div style={{ fontSize: '13px', fontWeight: 600, textTransform: 'capitalize' }}>
                    {key.replace('_', ' ')}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {info.path}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  {info.exists ? (
                    <span className="badge badge-success" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <CheckCircle2 size={12} /> {info.size_mb} MB
                    </span>
                  ) : (
                    <span className="badge badge-danger" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <XCircle size={12} /> Missing
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── SQLite Database Connectivity ── */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Database size={17} color="var(--accent-cyan)" />
              <span>SQLite Database Storage</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12.5px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Database Path</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{diagnostics.database.db_path}</span>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12.5px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Connection Status</span>
              <span>
                {diagnostics.database.connected ? (
                  <span className="badge badge-success">Connected</span>
                ) : (
                  <span className="badge badge-danger">Disconnected</span>
                )}
              </span>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12.5px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Events Table</span>
              <span>
                {diagnostics.database.table_exists ? (
                  <span className="badge badge-success">Initialized</span>
                ) : (
                  <span className="badge badge-danger">Not Initialized</span>
                )}
              </span>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12.5px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Total Logged Records</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                {diagnostics.database.total_events}
              </span>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12.5px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Actual Safety Violations</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--color-danger)' }}>
                {diagnostics.database.total_violations}
              </span>
            </div>
          </div>
        </div>

        {/* ── Directory Storage Permissions ── */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <FolderCheck size={17} color="var(--accent-cyan)" />
              <span>Storage &amp; Directory Permissions</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {Object.entries(diagnostics.directories).map(([dir, info]) => (
              <div
                key={dir}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '12px',
                }}
              >
                <span style={{ fontFamily: 'var(--font-mono)' }}>{dir}</span>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {info.exists && <span className="badge badge-success">Exists</span>}
                  {info.writable ? (
                    <span className="badge badge-info">Writable</span>
                  ) : (
                    <span className="badge badge-danger">Read-Only</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Codec & Execution Environment ── */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Film size={17} color="var(--accent-cyan)" />
              <span>Codec &amp; Execution Environment</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>OpenH264 Codec DLL</span>
              <span>
                {diagnostics.openh264.exists ? (
                  <span className="badge badge-success">{diagnostics.openh264.size_kb} KB Active</span>
                ) : (
                  <span className="badge badge-warning">Missing (Fallback FourCC)</span>
                )}
              </span>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Operating System Platform</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{diagnostics.environment.platform}</span>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Python Runtime Version</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{diagnostics.environment.python_version}</span>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>PyTorch Torch Version</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{diagnostics.environment.torch_version || 'N/A'}</span>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '10px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '12px',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Hardware Acceleration (CUDA)</span>
              <span>
                {diagnostics.environment.cuda_available ? (
                  <span className="badge badge-success">CUDA GPU Active</span>
                ) : (
                  <span className="badge badge-info">CPU Inference Mode</span>
                )}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
