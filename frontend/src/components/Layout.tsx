import React from 'react';
import {
  LayoutDashboard,
  Video,
  ShieldAlert,
  ScanLine,
  FileText,
  Server,
  Radio,
} from 'lucide-react';
import { Header } from './Header';

export type NavTab = 'overview' | 'traffic' | 'violations' | 'plates' | 'records' | 'diagnostics';

interface LayoutProps {
  currentTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  children: React.ReactNode;
}

const NAV_ITEMS: Array<{ id: NavTab; label: string; icon: React.ReactNode }> = [
  { id: 'overview', label: 'Command Overview', icon: <LayoutDashboard size={18} /> },
  { id: 'traffic', label: 'Traffic Analysis', icon: <Video size={18} /> },
  { id: 'violations', label: 'Safety Violations', icon: <ShieldAlert size={18} /> },
  { id: 'plates', label: 'Plate Scanner', icon: <ScanLine size={18} /> },
  { id: 'records', label: 'Records & Reports', icon: <FileText size={18} /> },
  { id: 'diagnostics', label: 'System Diagnostics', icon: <Server size={18} /> },
];

export const Layout: React.FC<LayoutProps> = ({ currentTab, onTabChange, children }) => {
  const currentNav = NAV_ITEMS.find((item) => item.id === currentTab);

  return (
    <div className="app-layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">
            <Radio size={20} />
          </div>
          <div>
            <div className="brand-title">TRAFFIC VISION</div>
            <div className="brand-sub">Command Center</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`nav-item ${currentTab === item.id ? 'active' : ''}`}
            >
              <span style={{ color: currentTab === item.id ? 'var(--accent-cyan)' : 'inherit' }}>
                {item.icon}
              </span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            <div>Smart Traffic Vision v1.0</div>
            <div style={{ fontFamily: 'var(--font-mono)', marginTop: '2px' }}>FastAPI + YOLO11</div>
          </div>
        </div>
      </aside>

      {/* ── Main View ── */}
      <div className="main-content">
        <Header
          title={currentNav?.label || 'Smart Traffic Vision'}
          subtitle="Real-time multi-object tracking, counting & violation surveillance"
        />
        <main className="page-container">{children}</main>
      </div>
    </div>
  );
};
