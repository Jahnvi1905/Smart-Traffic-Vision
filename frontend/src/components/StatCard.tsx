import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  variant?: 'default' | 'danger' | 'warning' | 'success' | 'info';
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  subtitle,
  icon,
  variant = 'default',
}) => {
  const getVariantColor = () => {
    switch (variant) {
      case 'danger':
        return 'var(--color-danger)';
      case 'warning':
        return 'var(--color-warning)';
      case 'success':
        return 'var(--color-success)';
      case 'info':
        return 'var(--accent-cyan)';
      default:
        return 'var(--border-subtle)';
    }
  };

  return (
    <div
      className="stat-card"
      style={{
        borderLeft: variant !== 'default' ? `3px solid ${getVariantColor()}` : undefined,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div className="stat-card-label">{label}</div>
        {icon && <div style={{ color: 'var(--text-muted)' }}>{icon}</div>}
      </div>
      <div className="stat-card-value">{value}</div>
      {subtitle && <div className="stat-card-sub">{subtitle}</div>}
    </div>
  );
};
