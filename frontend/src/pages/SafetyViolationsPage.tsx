import React, { useEffect, useState } from 'react';
import {
  ShieldAlert,
  Search,
  CheckCircle,
  XCircle,
  ChevronLeft,
  ChevronRight,
  AlertOctagon,
} from 'lucide-react';
import { api, getFullMediaUrl } from '../api/client';
import type { ViolationItem, ViolationsListResponse } from '../api/types';
import { LoadingState } from '../components/LoadingState';
import { ErrorState } from '../components/ErrorState';
import { EmptyState } from '../components/EmptyState';
import { Modal } from '../components/Modal';

export const SafetyViolationsPage: React.FC = () => {
  const [data, setData] = useState<ViolationsListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [eventType, setEventType] = useState<string>('ALL');
  const [reviewStatus, setReviewStatus] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [page, setPage] = useState<number>(1);
  const limit = 9;

  // Selected Evidence for Lightbox
  const [activeEvidence, setActiveEvidence] = useState<{ url: string; item: ViolationItem } | null>(null);
  const [updatingId, setUpdatingId] = useState<number | null>(null);

  const fetchViolations = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getViolations({
        event_type: eventType,
        review_status: reviewStatus,
        search: searchQuery.trim() || undefined,
        page,
        limit,
      });
      setData(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch violations records.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchViolations();
  }, [eventType, reviewStatus, page]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchViolations();
  };

  const handleUpdateStatus = async (id: number, status: 'approved' | 'rejected' | 'pending') => {
    setUpdatingId(id);
    try {
      await api.updateViolationReview(id, status);
      // Update local state directly for crisp, instant UI response
      setData((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          items: prev.items.map((item) => (item.id === id ? { ...item, review_status: status } : item)),
        };
      });
    } catch (err: unknown) {
      alert(`Review status update failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUpdatingId(null);
    }
  };

  const getBadgeForType = (type: string) => {
    switch (type) {
      case 'NO_HELMET':
        return <span className="badge badge-danger">No Helmet</span>;
      case 'WRONG_WAY':
        return <span className="badge badge-danger">Wrong-Way</span>;
      case 'RED_LIGHT':
        return <span className="badge badge-warning">Stop-Line Policy</span>;
      default:
        return <span className="badge badge-info">{type}</span>;
    }
  };

  const getBadgeForStatus = (status: string) => {
    switch (status.toLowerCase()) {
      case 'approved':
        return <span className="badge badge-success">Approved</span>;
      case 'rejected':
        return <span className="badge badge-danger">Rejected</span>;
      default:
        return <span className="badge badge-warning">Pending Review</span>;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* ── Filters & Search Control Bar ── */}
      <div className="card">
        <div className="card-header" style={{ marginBottom: 0 }}>
          <div className="card-title">
            <ShieldAlert size={18} color="var(--color-danger)" />
            <span>Safety Violation Auditing &amp; Review Workflow</span>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', marginTop: '16px' }}>
          {/* Search Box */}
          <form onSubmit={handleSearchSubmit} style={{ flex: 1, minWidth: '240px', display: 'flex', gap: '8px' }}>
            <div style={{ position: 'relative', width: '100%' }}>
              <input
                type="text"
                className="input"
                placeholder="Search by plate number, vehicle class..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ paddingLeft: '32px' }}
              />
              <Search
                size={14}
                style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}
              />
            </div>
            <button type="submit" className="btn btn-secondary">
              Search
            </button>
          </form>

          {/* Type Filter */}
          <div style={{ minWidth: '170px' }}>
            <select
              className="select"
              value={eventType}
              onChange={(e) => {
                setEventType(e.target.value);
                setPage(1);
              }}
            >
              <option value="ALL">All Violation Types</option>
              <option value="NO_HELMET">Helmet Violations</option>
              <option value="WRONG_WAY">Wrong-Way Incidents</option>
              <option value="RED_LIGHT">Stop-Line Policy</option>
            </select>
          </div>

          {/* Review Status Filter */}
          <div style={{ minWidth: '150px' }}>
            <select
              className="select"
              value={reviewStatus}
              onChange={(e) => {
                setReviewStatus(e.target.value);
                setPage(1);
              }}
            >
              <option value="ALL">All Review Statuses</option>
              <option value="pending">Pending</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
            </select>
          </div>
        </div>
      </div>

      {/* ── Content View ── */}
      {loading ? (
        <LoadingState message="Fetching safety violation records from database..." />
      ) : error ? (
        <ErrorState
          title="Error Loading Violations"
          message={error}
          onRetry={fetchViolations}
        />
      ) : !data || data.items.length === 0 ? (
        <EmptyState
          icon={<AlertOctagon size={36} color="var(--accent-cyan)" />}
          title="No Safety Violations Found"
          description="There are no safety violations matching your selected criteria. All vehicles are in compliance or no incidents were flagged."
        />
      ) : (
        <>
          {/* Violation Grid */}
          <div className="grid-3">
            {data.items.map((item: ViolationItem) => (
              <div
                key={item.id}
                className="card"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  padding: '16px',
                  borderTop: '3px solid var(--color-danger)',
                }}
              >
                {/* Evidence Thumbnail */}
                <div
                  style={{
                    position: 'relative',
                    height: '180px',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: 'var(--radius-md)',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px solid var(--border-subtle)',
                    marginBottom: '14px',
                    cursor: item.evidence_url ? 'pointer' : 'default',
                  }}
                  onClick={() => {
                    if (item.evidence_url) {
                      setActiveEvidence({ url: getFullMediaUrl(item.evidence_url), item });
                    }
                  }}
                >
                  {item.evidence_url ? (
                    <img
                      src={getFullMediaUrl(item.evidence_url)}
                      alt="Violation Evidence"
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  ) : (
                    <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>No Evidence Snapshot</div>
                  )}

                  <div style={{ position: 'absolute', top: '8px', left: '8px' }}>
                    {getBadgeForType(item.event_type)}
                  </div>

                  <div style={{ position: 'absolute', top: '8px', right: '8px' }}>
                    {getBadgeForStatus(item.review_status)}
                  </div>
                </div>

                {/* Metadata Details */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Incident ID</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>#{item.id}</span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Timestamp</span>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>{item.timestamp}</span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Vehicle Type</span>
                    <span style={{ fontWeight: 500, textTransform: 'capitalize' }}>
                      {item.vehicle_class || 'Vehicle'}
                    </span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Track ID</span>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>
                      {item.track_id !== null && item.track_id !== undefined ? `#${item.track_id}` : '—'}
                    </span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>License Plate</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                      {item.plate_text && item.plate_text !== 'UNKNOWN' ? item.plate_text : 'Not Identified'}
                    </span>
                  </div>

                  {item.confidence && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Model Confidence</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{(item.confidence * 100).toFixed(0)}%</span>
                    </div>
                  )}
                </div>

                {/* Review Action Buttons */}
                <div
                  style={{
                    display: 'flex',
                    gap: '8px',
                    marginTop: '16px',
                    paddingTop: '12px',
                    borderTop: '1px solid var(--border-subtle)',
                  }}
                >
                  <button
                    onClick={() => handleUpdateStatus(item.id, 'approved')}
                    disabled={updatingId === item.id || item.review_status === 'approved'}
                    className="btn btn-secondary btn-sm"
                    style={{ flex: 1, color: 'var(--color-success)', borderColor: 'var(--color-success-border)' }}
                  >
                    <CheckCircle size={14} /> Approve
                  </button>

                  <button
                    onClick={() => handleUpdateStatus(item.id, 'rejected')}
                    disabled={updatingId === item.id || item.review_status === 'rejected'}
                    className="btn btn-secondary btn-sm"
                    style={{ flex: 1, color: 'var(--color-danger)', borderColor: 'var(--color-danger-border)' }}
                  >
                    <XCircle size={14} /> Reject
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination Footer */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '14px 20px',
              backgroundColor: 'var(--bg-surface)',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
              Showing Page <span style={{ fontWeight: 600, color: '#fff' }}>{data.page}</span> of{' '}
              <span style={{ fontWeight: 600, color: '#fff' }}>{data.pages}</span> ({data.total} total violations)
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={data.page <= 1}
                className="btn btn-secondary btn-sm"
              >
                <ChevronLeft size={14} /> Previous
              </button>
              <button
                onClick={() => setPage((p) => Math.min(data.pages, p + 1))}
                disabled={data.page >= data.pages}
                className="btn btn-secondary btn-sm"
              >
                Next <ChevronRight size={14} />
              </button>
            </div>
          </div>
        </>
      )}

      {/* ── Evidence Lightbox Modal ── */}
      <Modal
        isOpen={activeEvidence !== null}
        onClose={() => setActiveEvidence(null)}
        title={activeEvidence ? `Violation Evidence: Event #${activeEvidence.item.id}` : 'Evidence Preview'}
      >
        {activeEvidence && (
          <div>
            <img
              src={activeEvidence.url}
              alt="Annotated Evidence"
              style={{ width: '100%', borderRadius: 'var(--radius-md)', display: 'block' }}
            />
            <div
              style={{
                marginTop: '14px',
                padding: '12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-md)',
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '12px',
              }}
            >
              <div>
                Type: <b>{activeEvidence.item.event_type}</b> | Vehicle: <b>{activeEvidence.item.vehicle_class || 'Vehicle'}</b>
              </div>
              <div>
                Timestamp: <span style={{ fontFamily: 'var(--font-mono)' }}>{activeEvidence.item.timestamp}</span>
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};
