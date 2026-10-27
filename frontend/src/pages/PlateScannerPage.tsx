import React, { useState } from 'react';
import {
  ScanLine,
  Upload,
  CheckCircle2,
  AlertCircle,
  Database,
  ZoomIn,
} from 'lucide-react';
import { api, getFullMediaUrl } from '../api/client';
import type { PlateScanResponse } from '../api/types';
import { LoadingState } from '../components/LoadingState';
import { ErrorState } from '../components/ErrorState';
import { EmptyState } from '../components/EmptyState';
import { Modal } from '../components/Modal';

export const PlateScannerPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [useSample, setUseSample] = useState<boolean>(true);
  const [saveToDb, setSaveToDb] = useState<boolean>(false);

  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PlateScanResponse | null>(null);

  const [previewModalImg, setPreviewModalImg] = useState<string | null>(null);

  const handleScan = async () => {
    setIsScanning(true);
    setError(null);

    try {
      const res = await api.scanPlate({
        file: useSample ? null : selectedFile,
        use_sample: useSample,
        save_to_db: saveToDb,
      });
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Plate detection or OCR extraction failed.');
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* ── Input & Control Card ── */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <ScanLine size={18} color="var(--accent-cyan)" />
            <span>AI License Plate Localization &amp; EasyOCR Text Extraction</span>
          </div>
        </div>

        <div className="grid-2">
          {/* File Upload Zone */}
          <div
            style={{
              border: '2px dashed var(--border-muted)',
              borderRadius: 'var(--radius-lg)',
              backgroundColor: 'var(--bg-secondary)',
              padding: '24px 20px',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              cursor: 'pointer',
            }}
            onClick={() => document.getElementById('plate-file-input')?.click()}
          >
            <input
              id="plate-file-input"
              type="file"
              accept="image/jpeg,image/png"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setSelectedFile(e.target.files[0]);
                  setUseSample(false);
                }
              }}
              style={{ display: 'none' }}
            />

            <Upload size={28} color="var(--accent-cyan)" />
            <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
              {selectedFile ? selectedFile.name : 'Upload vehicle photo (JPG, PNG)'}
            </div>

            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setUseSample(true);
                setSelectedFile(null);
              }}
              className={`btn btn-sm ${useSample ? 'btn-primary' : 'btn-secondary'}`}
              style={{ marginTop: '6px' }}
            >
              {useSample ? '✓ Benchmark plate_test.jpg Active' : 'Try Benchmark plate_test.jpg'}
            </button>
          </div>

          {/* Execution Options */}
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Locates license plates using fine-tuned YOLO plate weights and extracts alphanumeric registration strings using EasyOCR.
              </div>

              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '12.5px',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                  padding: '10px 12px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <input
                  type="checkbox"
                  checked={saveToDb}
                  onChange={(e) => setSaveToDb(e.target.checked)}
                />
                <Database size={15} color="var(--accent-cyan)" />
                <span>Log identified plates to SQLite database (PLATE_READ)</span>
              </label>
            </div>

            <button
              onClick={handleScan}
              disabled={isScanning || (!useSample && !selectedFile)}
              className="btn btn-primary"
              style={{ marginTop: '16px', padding: '10px' }}
            >
              {isScanning ? 'Extracting License Plate OCR...' : 'Run License Plate OCR'}
            </button>
          </div>
        </div>
      </div>

      {/* ── Status Messages ── */}
      {isScanning && (
        <LoadingState
          message="Localizing bounding boxes and extracting alphanumeric characters with EasyOCR..."
          minHeight="140px"
        />
      )}

      {error && (
        <ErrorState
          title="Plate Recognition Error"
          message={error}
          onRetry={handleScan}
        />
      )}

      {/* ── Results View ── */}
      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {result.total_plates === 0 ? (
            <EmptyState
              icon={<AlertCircle size={36} color="var(--color-warning)" />}
              title="No License Plate Detected"
              description="No vehicle license plate could be localized in this image at current confidence thresholds. Ensure the plate is clearly illuminated and unoccluded."
            />
          ) : (
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <CheckCircle2 size={18} color="var(--color-success)" />
                  <span>Scan Results: {result.total_plates} License Plate(s) Extracted</span>
                </div>
              </div>

              <div className="grid-2">
                {/* Annotated Localization Image */}
                {result.annotated_image_url && (
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                      VEHICLE LOCALIZATION VIEW
                    </div>
                    <div
                      style={{
                        position: 'relative',
                        borderRadius: 'var(--radius-md)',
                        overflow: 'hidden',
                        border: '1px solid var(--border-subtle)',
                        cursor: 'pointer',
                      }}
                      onClick={() => setPreviewModalImg(getFullMediaUrl(result.annotated_image_url))}
                    >
                      <img
                        src={getFullMediaUrl(result.annotated_image_url)}
                        alt="Plate Localization"
                        style={{ width: '100%', display: 'block' }}
                      />
                      <div
                        style={{
                          position: 'absolute',
                          bottom: '8px',
                          right: '8px',
                          background: 'rgba(0,0,0,0.6)',
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '11px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                        }}
                      >
                        <ZoomIn size={12} /> Click to Enlarge
                      </div>
                    </div>
                  </div>
                )}

                {/* Extracted Plates Cards */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    EXTRACTED CROPS &amp; OCR RESULTS
                  </div>

                  {result.plates.map((plate) => (
                    <div
                      key={plate.idx}
                      style={{
                        padding: '16px',
                        backgroundColor: 'var(--bg-secondary)',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        gap: '16px',
                        alignItems: 'center',
                      }}
                    >
                      {/* Cropped Plate Image */}
                      {plate.crop_url ? (
                        <img
                          src={getFullMediaUrl(plate.crop_url)}
                          alt="Plate Crop"
                          style={{
                            width: '120px',
                            height: '56px',
                            objectFit: 'contain',
                            backgroundColor: '#000',
                            borderRadius: 'var(--radius-sm)',
                            border: '1px solid var(--border-muted)',
                          }}
                        />
                      ) : (
                        <div
                          style={{
                            width: '120px',
                            height: '56px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backgroundColor: '#000',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: '11px',
                            color: 'var(--text-muted)',
                          }}
                        >
                          No Crop
                        </div>
                      )}

                      {/* Plate Tag & Confidence */}
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                          <div
                            style={{
                              display: 'inline-block',
                              backgroundColor: '#0F172A',
                              border: plate.is_low_confidence ? '2px solid #EAB308' : '2px solid #F59E0B',
                              borderRadius: 'var(--radius-sm)',
                              padding: '4px 14px',
                              fontFamily: 'var(--font-mono)',
                              fontSize: '18px',
                              fontWeight: 700,
                              letterSpacing: '0.12em',
                              color: '#F8FAFC',
                            }}
                          >
                            {plate.plate_text}
                          </div>

                          {plate.is_low_confidence && (
                            <span
                              className="badge badge-warning"
                              style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '11px' }}
                            >
                              <AlertCircle size={11} /> Low Confidence (&lt; 50%)
                            </span>
                          )}
                        </div>

                        {plate.raw_text && plate.raw_text !== plate.plate_text && (
                          <div
                            style={{
                              marginTop: '6px',
                              fontSize: '11.5px',
                              color: 'var(--text-secondary)',
                              fontFamily: 'var(--font-mono)',
                            }}
                          >
                            Raw OCR Output: <span style={{ color: 'var(--text-muted)' }}>"{plate.raw_text}"</span>
                          </div>
                        )}

                        <div style={{ display: 'flex', gap: '14px', marginTop: '6px', fontSize: '12px', color: 'var(--text-muted)', flexWrap: 'wrap' }}>
                          <div>
                            OCR Confidence:{' '}
                            <span
                              style={{
                                fontFamily: 'var(--font-mono)',
                                color: plate.is_low_confidence ? 'var(--color-warning)' : 'var(--text-primary)',
                                fontWeight: plate.is_low_confidence ? 600 : 400,
                              }}
                            >
                              {(plate.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            Bounding Box:{' '}
                            <span style={{ fontFamily: 'var(--font-mono)' }}>[{plate.box.join(', ')}]</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Photo Preview Modal ── */}
      <Modal
        isOpen={previewModalImg !== null}
        onClose={() => setPreviewModalImg(null)}
        title="Full Vehicle Localization View"
        maxWidth="840px"
      >
        {previewModalImg && (
          <img
            src={previewModalImg}
            alt="Full Localization"
            style={{ width: '100%', borderRadius: 'var(--radius-md)', display: 'block' }}
          />
        )}
      </Modal>
    </div>
  );
};
