import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  minHeight?: number | string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Service Connection Error',
  message,
  onRetry,
  minHeight = '200px',
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight,
        padding: '24px',
        textAlign: 'center',
        backgroundColor: 'var(--color-danger-bg)',
        border: '1px solid var(--color-danger-border)',
        borderRadius: 'var(--radius-lg)',
        gap: '12px',
      }}
    >
      <AlertTriangle size={32} color="var(--color-danger)" />
      <div style={{ fontWeight: 600, fontSize: '15px', color: 'var(--text-primary)' }}>
        {title}
      </div>
      <p style={{ fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '480px', lineHeight: 1.5 }}>
        {message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="btn btn-secondary btn-sm"
          style={{ marginTop: '6px' }}
        >
          <RotateCcw size={14} /> Retry Request
        </button>
      )}
    </div>
  );
};
