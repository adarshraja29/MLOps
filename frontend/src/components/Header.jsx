import React, { useEffect, useState } from 'react';

export default function Header() {
  const [health, setHealth] = useState({ status: 'checking', rss_memory_mb: null, device: 'cpu' });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/health');
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        } else {
          setHealth({ status: 'offline', rss_memory_mb: null, device: 'cpu' });
        }
      } catch (err) {
        setHealth({ status: 'offline', rss_memory_mb: null, device: 'cpu' });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000); // Check every 15s
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="header-wrapper">
      <div className="header-brand">
        <div className="header-icon-container">
          <svg className="header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M9 8h6M9 12h4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div>
          <h1 className="header-title">Academic Literature Summarizer</h1>
          <p className="header-subtitle">
            Continuous Delivery Pipeline for Computational Linguistics & NLP Research
          </p>
        </div>
      </div>

      <div className="header-badges">
        {/* System Health */}
        <div className={`status-pill ${health.status === 'healthy' ? 'status-online' : 'status-offline'}`}>
          <span className="status-dot"></span>
          <span>{health.status === 'healthy' ? 'System Online' : 'Connecting...'}</span>
          {health.rss_memory_mb && (
            <span className="font-mono text-xs opacity-75">({health.rss_memory_mb} MB RSS)</span>
          )}
        </div>

        {/* Model tags */}
        <div className="tag-pill">
          <span className="tag-label">Tokenizer</span>
          <span className="tag-val font-mono">SciBERT</span>
        </div>

        <div className="tag-pill">
          <span className="tag-label">Summarizer</span>
          <span className="tag-val font-mono">LongT5-Base</span>
        </div>

        <div className="tag-pill tag-cpu">
          <span className="tag-val font-mono">CPU Float32</span>
        </div>
      </div>
    </header>
  );
}
