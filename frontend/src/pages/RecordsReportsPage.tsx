import React, { useEffect, useState } from 'react';
import {
  FileText,
  Download,
  Search,
  Eye,
  ChevronLeft,
  ChevronRight,
  Database,
  X,
} from 'lucide-react';
import { api, getFullMediaUrl } from '../api/client';
import type { ViolationItem, ViolationsListResponse } from '../api/types';
import { LoadingState } from '../components/LoadingState';
import { ErrorState } from '../components/ErrorState';
import { EmptyState } from '../components/EmptyState';
import { Modal } from '../components/Modal';

export const RecordsReportsPage: React.FC = () => {
  const [data, setData] = useState<ViolationsListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [eventType, setEventType] = useState<string>('ALL');
  const [reviewStatus, setReviewStatus] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [includeCounts, setIncludeCounts] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const limit = 12;

  // Selected Record Details Drawer & Modal
  const [selectedRecord, setSelectedRecord] = useState<ViolationItem | null>(null);
  const [previewMediaUrl, setPreviewMediaUrl] = useState<string | null>(null);

  const fetchRecords = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getViolations({
        event_type: eventType === 'ALL' ? undefined : eventType,
        review_status: reviewStatus === 'ALL' ? undefined : reviewStatus,
        search: searchQuery.trim() || undefined,
        include_counts: includeCounts,
        page,
        limit,
      });
      setData(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch database records.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, [eventType, reviewStatus, includeCounts, page]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchRecords();
  };

  const handleExportCsv = () => {
    const exportUrl = api.getReportsExportUrl({
      event_type: eventType,
      review_status: reviewStatus,
      include_counts: includeCounts,
    });
    window.open(exportUrl, '_blank');
  };

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
      {/* ── Control Header & CSV Export ── */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <FileText size={18} color="var(--accent-cyan)" />
            <span>SQLite Database Audit Ledger &amp; Export</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={includeCounts}
                onChange={(e) => setIncludeCounts(e.target.checked)}
              />
              Include Flow COUNT in Export
            </label>

            <button onClick={handleExportCsv} className="btn btn-secondary btn-sm">
              <Download size={14} /> Export CSV
            </button>
          </div>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', marginTop: '14px' }}>
          <form onSubmit={handleSearchSubmit} style={{ flex: 1, minWidth: '220px', display: 'flex', gap: '8px' }}>
            <div style={{ position: 'relative', width: '100%' }}>
              <input
                type="text"
                className="input"
                placeholder="Search plate, vehicle class..."
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

          <div style={{ minWidth: '170px' }}>
            <select
              className="select"
              value={eventType}
              onChange={(e) => {
                setEventType(e.target.value);
                setPage(1);
              }}
            >
              <option value="ALL">All Event Types</option>
              <option value="NO_HELMET">Helmet Violations</option>
              <option value="WRONG_WAY">Wrong-Way Incidents</option>
              <option value="RED_LIGHT">Stop-Line Policy</option>
              <option value="PLATE_READ">Plate Scans</option>
              <option value="COUNT">Traffic Flow Counts</option>
            </select>
          </div>

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

      {/* ── Table & Drawer Grid ── */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedRecord ? '1.8fr 1fr' : '1fr', gap: '20px' }}>
        {/* Main Records Table */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          {loading ? (
            <LoadingState message="Querying SQLite event records..." minHeight="300px" />
          ) : error ? (
            <ErrorState title="Database Query Error" message={error} onRetry={fetchRecords} />
          ) : !data || data.items.length === 0 ? (
            <EmptyState
              title="No Database Records Found"
              description="No database events match your filter query."
              minHeight="260px"
            />
          ) : (
            <div className="table-container" style={{ border: 'none', borderRadius: 0 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Timestamp</th>
                    <th>Event Type</th>
                    <th>Vehicle Class</th>
                    <th>Track ID</th>
                    <th>Plate Text</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((item: ViolationItem) => (
                    <tr
                      key={item.id}
                      style={{
                        backgroundColor: selectedRecord?.id === item.id ? 'var(--bg-surface-elevated)' : undefined,
                        cursor: 'pointer',
                      }}
                      onClick={() => setSelectedRecord(item)}
                    >
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>#{item.id}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                        {item.timestamp}
                      </td>
                      <td>{getBadgeForType(item.event_type)}</td>
                      <td style={{ textTransform: 'capitalize' }}>{item.vehicle_class || '—'}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>
                        {item.track_id !== null && item.track_id !== undefined ? `#${item.track_id}` : '—'}
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--accent-cyan)' }}>
                        {item.plate_text && !item.plate_text.startsWith('{') ? item.plate_text : '—'}
                      </td>
                      <td>{getBadgeForStatus(item.review_status)}</td>
                      <td>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedRecord(item);
                          }}
                          className="btn btn-outline btn-sm"
                          style={{ padding: '3px 8px', fontSize: '11px' }}
                        >
                          <Eye size={12} /> Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination Footer */}
          {data && data.pages > 1 && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 18px',
                borderTop: '1px solid var(--border-subtle)',
                backgroundColor: 'var(--bg-secondary)',
              }}
            >
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Page {data.page} of {data.pages} ({data.total} total rows)
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={data.page <= 1}
                  className="btn btn-secondary btn-sm"
                >
                  <ChevronLeft size={13} /> Prev
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(data.pages, p + 1))}
                  disabled={data.page >= data.pages}
                  className="btn btn-secondary btn-sm"
                >
                  Next <ChevronRight size={13} />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Selected Record Detail Panel */}
        {selectedRecord && (
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div className="card-header" style={{ marginBottom: '6px' }}>
              <div className="card-title">
                <Database size={16} color="var(--accent-cyan)" />
                <span>Transaction #{selectedRecord.id}</span>
              </div>
              <button
                onClick={() => setSelectedRecord(null)}
                className="btn btn-outline btn-sm"
                style={{ padding: '4px', border: 'none', color: 'var(--text-muted)' }}
              >
                <X size={16} />
              </button>
            </div>

            {/* Evidence Thumbnail */}
            {selectedRecord.evidence_url ? (
              <div
                style={{
                  height: '180px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: 'var(--radius-md)',
                  overflow: 'hidden',
                  cursor: 'pointer',
                  border: '1px solid var(--border-subtle)',
                }}
                onClick={() => setPreviewMediaUrl(getFullMediaUrl(selectedRecord.evidence_url))}
              >
                {selectedRecord.evidence_url.endsWith('.mp4') ? (
                  <video
                    src={getFullMediaUrl(selectedRecord.evidence_url)}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                ) : (
                  <img
                    src={getFullMediaUrl(selectedRecord.evidence_url)}
                    alt="Evidence Preview"
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                )}
              </div>
            ) : (
              <div
                style={{
                  height: '100px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: 'var(--radius-md)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '12px',
                  color: 'var(--text-muted)',
                }}
              >
                No Media Attached
              </div>
            )}

            {/* Fields List */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12.5px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Event Type</span>
                <span>{getBadgeForType(selectedRecord.event_type)}</span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Review Status</span>
                <span>{getBadgeForStatus(selectedRecord.review_status)}</span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Timestamp</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{selectedRecord.timestamp}</span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Track ID</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>
                  {selectedRecord.track_id !== null && selectedRecord.track_id !== undefined
                    ? `#${selectedRecord.track_id}`
                    : '—'}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Vehicle Class</span>
                <span style={{ fontWeight: 500, textTransform: 'capitalize' }}>
                  {selectedRecord.vehicle_class || '—'}
                </span>
              </div>

              {selectedRecord.event_type === 'COUNT' ? (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Total Counted</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                      {selectedRecord.total_count !== null && selectedRecord.total_count !== undefined
                        ? `${selectedRecord.total_count} vehicles`
                        : '—'}
                    </span>
                  </div>
                  {selectedRecord.count_details && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Class Breakdown</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-secondary)', textAlign: 'right' }}>
                        {selectedRecord.count_details}
                      </span>
                    </div>
                  )}
                </>
              ) : (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Plate Text</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                    {selectedRecord.plate_text || '—'}
                  </span>
                </div>
              )}

              {selectedRecord.confidence !== null && selectedRecord.confidence !== undefined && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Confidence</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}>
                    {(selectedRecord.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Media Lightbox Modal ── */}
      <Modal
        isOpen={previewMediaUrl !== null}
        onClose={() => setPreviewMediaUrl(null)}
        title="Evidence Media View"
      >
        {previewMediaUrl && (
          <div>
            {previewMediaUrl.endsWith('.mp4') ? (
              <video
                src={previewMediaUrl}
                controls
                autoPlay
                style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
              />
            ) : (
              <img
                src={previewMediaUrl}
                alt="Evidence"
                style={{ width: '100%', borderRadius: 'var(--radius-md)', display: 'block' }}
              />
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};
