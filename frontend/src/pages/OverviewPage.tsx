import React, { useEffect, useState } from 'react';
import {
  Car,
  ShieldAlert,
  AlertTriangle,
  RotateCcw,
  Activity,
  Layers,
  ArrowRight,
  ExternalLink,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { api, getFullMediaUrl } from '../api/client';
import type { OverviewStatsResponse, RecentEventItem } from '../api/types';
import { StatCard } from '../components/StatCard';
import { LoadingState } from '../components/LoadingState';
import { ErrorState } from '../components/ErrorState';
import { EmptyState } from '../components/EmptyState';
import { Modal } from '../components/Modal';

interface OverviewPageProps {
  onNavigate: (tab: 'traffic' | 'violations' | 'plates' | 'records' | 'diagnostics') => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  car: '#10B981',
  'two-wheeler': '#F59E0B',
  'three-wheeler': '#38BDF8',
  truck: '#EC4899',
  bus: '#F97316',
  van: '#8B5CF6',
};

export const OverviewPage: React.FC<OverviewPageProps> = ({ onNavigate }) => {
  const [data, setData] = useState<OverviewStatsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<{ url: string; title: string } | null>(null);

  const fetchOverview = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getOverviewStats();
      setData(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch overview metrics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverview();
  }, []);

  if (loading) {
    return <LoadingState message="Connecting to traffic vision database & calculating metrics..." />;
  }

  if (error || !data) {
    return (
      <ErrorState
        title="Failed to Load Traffic Overview"
        message={error || 'Unable to communicate with the FastAPI backend.'}
        onRetry={fetchOverview}
      />
    );
  }

  // Format category breakdown for Recharts
  const chartData = Object.entries(data.category_breakdown || {}).map(([key, val]) => ({
    name: key.replace('-', ' ').toUpperCase(),
    rawKey: key.toLowerCase(),
    count: val,
  }));

  const getBadgeForType = (type: string) => {
    switch (type) {
      case 'NO_HELMET':
        return <span className="badge badge-danger">No Helmet</span>;
      case 'WRONG_WAY':
        return <span className="badge badge-danger">Wrong-Way</span>;
      case 'RED_LIGHT':
        return <span className="badge badge-warning">Stop-Line</span>;
      case 'PLATE_READ':
        return <span className="badge badge-info">Plate Read</span>;
      case 'COUNT':
        return <span className="badge badge-muted">Flow Count</span>;
      default:
        return <span className="badge badge-muted">{type}</span>;
    }
  };

  const getBadgeForStatus = (status: string) => {
    switch (status.toLowerCase()) {
      case 'approved':
        return <span className="badge badge-success">Approved</span>;
      case 'rejected':
        return <span className="badge badge-danger">Rejected</span>;
      default:
        return <span className="badge badge-warning">Pending</span>;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* ── Top Level KPIs ── */}
      <div className="grid-4">
        <StatCard
          label="Total Vehicles Counted"
          value={data.total_vehicles.toLocaleString()}
          subtitle="Cumulative road line crossings"
          icon={<Car size={18} />}
          variant="info"
        />
        <StatCard
          label="Safety Violations"
          value={data.total_violations.toLocaleString()}
          subtitle="Excludes road COUNT summaries"
          icon={<ShieldAlert size={18} />}
          variant="danger"
        />
        <StatCard
          label="Helmet Non-Compliance"
          value={(data.violations_by_type?.NO_HELMET || 0).toLocaleString()}
          subtitle="Motorcycle rider safety triggers"
          icon={<AlertTriangle size={18} />}
          variant="warning"
        />
        <StatCard
          label="Wrong-Way Incidents"
          value={(data.violations_by_type?.WRONG_WAY || 0).toLocaleString()}
          subtitle="Opposing traffic vector alerts"
          icon={<RotateCcw size={18} />}
          variant="danger"
        />
      </div>

      {/* ── Visual Analytics & Classification ── */}
      <div className="grid-2">
        {/* Category Breakdown */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Layers size={17} color="var(--accent-cyan)" />
              <span>Vehicle Category Breakdown</span>
            </div>
            <button onClick={() => onNavigate('traffic')} className="btn btn-outline btn-sm">
              Run Video Analysis <ArrowRight size={13} />
            </button>
          </div>

          <div style={{ width: '100%', height: 260 }}>
            {chartData.length > 0 && chartData.some((d) => d.count > 0) ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <XAxis
                    dataKey="name"
                    stroke="var(--text-muted)"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: 'var(--border-subtle)' }}
                  />
                  <YAxis
                    stroke="var(--text-muted)"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: 'var(--border-subtle)' }}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--bg-secondary)',
                      borderColor: 'var(--border-muted)',
                      borderRadius: 'var(--radius-md)',
                      color: '#fff',
                      fontSize: '12px',
                    }}
                    cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry) => (
                      <Cell
                        key={`cell-${entry.rawKey}`}
                        fill={CATEGORY_COLORS[entry.rawKey] || 'var(--accent-cyan)'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState
                title="No Classification Data Yet"
                description="Process a traffic video in the Traffic Analysis tab to populate vehicle categories."
              />
            )}
          </div>
        </div>

        {/* System Intelligence Feed */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Activity size={17} color="var(--accent-cyan)" />
              <span>Safety Rule Violations</span>
            </div>
            <button onClick={() => onNavigate('violations')} className="btn btn-outline btn-sm">
              Review Queue <ArrowRight size={13} />
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '14px 16px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Motorcycle Helmet Enforcement
                </div>
                <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                  Two-wheeler rider head compliance audit
                </div>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 700, color: 'var(--color-warning)' }}>
                {data.violations_by_type?.NO_HELMET || 0}
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '14px 16px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Wrong-Way Lane Movement
                </div>
                <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                  Multi-frame trajectory displacement vector validation
                </div>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 700, color: 'var(--color-danger)' }}>
                {data.violations_by_type?.WRONG_WAY || 0}
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '14px 16px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Virtual Stop-Line Policy Violations
                </div>
                <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                  Configured signal policy red-light crossing
                </div>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                {data.violations_by_type?.RED_LIGHT || 0}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Recent Activity Feed ── */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <span>Recent Traffic Events Feed</span>
            <span style={{ fontSize: '11.5px', color: 'var(--text-muted)', fontWeight: 400 }}>
              (Last 10 SQLite database transactions)
            </span>
          </div>
          <button onClick={() => onNavigate('records')} className="btn btn-outline btn-sm">
            View All Records <ArrowRight size={13} />
          </button>
        </div>

        {data.recent_events && data.recent_events.length > 0 ? (
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Event ID</th>
                  <th>Timestamp</th>
                  <th>Type</th>
                  <th>Vehicle / Class</th>
                  <th>Track ID</th>
                  <th>Plate / Summary</th>
                  <th>Status</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_events.map((evt: RecentEventItem) => (
                  <tr key={evt.id}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>#{evt.id}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-secondary)' }}>
                      {evt.timestamp}
                    </td>
                    <td>{getBadgeForType(evt.event_type)}</td>
                    <td style={{ textTransform: 'capitalize' }}>
                      {evt.vehicle_class || '—'}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>
                      {evt.track_id !== null && evt.track_id !== undefined ? `#${evt.track_id}` : '—'}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
                      {evt.plate_text && !evt.plate_text.startsWith('{') ? evt.plate_text : '—'}
                    </td>
                    <td>{getBadgeForStatus(evt.review_status)}</td>
                    <td>
                      {evt.evidence_url ? (
                        <button
                          onClick={() =>
                            setSelectedEvidence({
                              url: getFullMediaUrl(evt.evidence_url),
                              title: `Evidence Frame: Event #${evt.id} (${evt.event_type})`,
                            })
                          }
                          className="btn btn-outline btn-sm"
                          style={{ padding: '3px 8px', fontSize: '11px' }}
                        >
                          <ExternalLink size={12} /> View
                        </button>
                      ) : (
                        <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>None</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No Activity Logged"
            description="Process videos or image frames to record traffic activity into SQLite."
          />
        )}
      </div>

      {/* ── Evidence Lightbox Modal ── */}
      <Modal
        isOpen={selectedEvidence !== null}
        onClose={() => setSelectedEvidence(null)}
        title={selectedEvidence?.title || 'Evidence Preview'}
      >
        {selectedEvidence?.url.endsWith('.mp4') ? (
          <video
            src={selectedEvidence.url}
            controls
            autoPlay
            style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
          />
        ) : (
          <img
            src={selectedEvidence?.url}
            alt="Violation Evidence"
            style={{ width: '100%', borderRadius: 'var(--radius-md)', display: 'block' }}
          />
        )}
      </Modal>
    </div>
  );
};
