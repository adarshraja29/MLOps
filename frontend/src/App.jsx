import React, { useState } from 'react';
import Header from './components/Header';
import SummarizerForm from './components/SummarizerForm';
import SummaryOutput from './components/SummaryOutput';
import MetricsBar from './components/MetricsBar';
import './App.css';

export default function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSummarize = async (payload) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        let errorMsg = 'Failed to generate summary. Please check backend service.';
        if (errorData?.detail) {
          if (Array.isArray(errorData.detail)) {
            errorMsg = errorData.detail.map((d) => d.msg || d).join(', ');
          } else {
            errorMsg = errorData.detail;
          }
        }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during inference.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />

      {error && (
        <div className="alert-error" role="alert">
          <span>⚠️ {error}</span>
          <button
            type="button"
            className="alert-close"
            onClick={() => setError(null)}
          >
            ✕
          </button>
        </div>
      )}

      <main style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <SummarizerForm
          onSummarize={handleSummarize}
          isLoading={isLoading}
        />

        {result && (
          <>
            <MetricsBar result={result} />
            <SummaryOutput result={result} />
          </>
        )}
      </main>
    </div>
  );
}
