import React, { useState, useEffect } from 'react';
import { Activity, Clock } from 'lucide-react';
import { api } from '../api/client';

interface HeaderProps {
  title: string;
  subtitle?: string;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
}) => {
  const [time, setTime] = useState<string>('');
  const [backendStatus, setBackendStatus] = useState<'online' | 'degraded' | 'offline'>('online');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime(
        now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
      );
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await api.getHealth();
        if (isMounted) {
          setBackendStatus(res.status === 'healthy' ? 'online' : 'degraded');
        }
      } catch {
        if (isMounted) setBackendStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="topbar">
      <div className="topbar-left">
        <div>
          <div className="page-title">{title}</div>
          {subtitle && (
            <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginTop: '1px' }}>
              {subtitle}
            </div>
          )}
        </div>
      </div>

      <div className="topbar-right">
        <div className="system-pill" style={{ background: 'var(--bg-surface)', padding: '5px 12px', borderRadius: 'var(--radius-sm)' }}>
          <div className={`status-dot ${backendStatus}`} />
          <span style={{ fontSize: '12px', fontWeight: 500, fontFamily: 'var(--font-mono)' }}>
            API: {backendStatus.toUpperCase()}
          </span>
        </div>

        <div className="system-pill" style={{ background: 'var(--bg-surface)', padding: '5px 12px', borderRadius: 'var(--radius-sm)' }}>
          <Clock size={13} color="var(--text-muted)" />
          <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
            {time || '--:--:--'}
          </span>
        </div>

        <div className="badge badge-info" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Activity size={12} />
          <span>PORT 8000</span>
        </div>
      </div>
    </header>
  );
};
