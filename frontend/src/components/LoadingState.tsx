import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  minHeight?: number | string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading live data from traffic pipeline...',
  minHeight = '240px',
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight,
        gap: '12px',
        color: 'var(--text-secondary)',
      }}
    >
      <Loader2
        size={28}
        className="animate-spin"
        style={{
          color: 'var(--accent-cyan)',
          animation: 'spin 1s linear infinite',
        }}
      />
      <span style={{ fontSize: '13px' }}>{message}</span>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};
