import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  minHeight?: number | string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  minHeight = '220px',
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight,
        padding: '32px 20px',
        textAlign: 'center',
        backgroundColor: 'var(--bg-surface)',
        border: '1px dashed var(--border-muted)',
        borderRadius: 'var(--radius-lg)',
        gap: '10px',
      }}
    >
      <div style={{ color: 'var(--text-muted)' }}>
        {icon || <Inbox size={32} />}
      </div>
      <div style={{ fontWeight: 600, fontSize: '14.5px', color: 'var(--text-primary)' }}>
        {title}
      </div>
      <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', maxWidth: '420px', lineHeight: 1.5 }}>
        {description}
      </p>
      {action && <div style={{ marginTop: '8px' }}>{action}</div>}
    </div>
  );
};
