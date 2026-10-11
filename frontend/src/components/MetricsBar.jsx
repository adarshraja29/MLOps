import React from 'react';

export default function MetricsBar({ result }) {
  if (!result) return null;

  const {
    compression_ratio,
    latency_seconds,
    rss_memory_mb,
    input_stats,
    summary_stats,
    device,
    cache_hit,
    rouge_scores
  } = result;

  const compressionPercent = Math.round((compression_ratio || 0) * 100);
  const timeSaved = Math.max(
    0,
    Math.round(
      (input_stats.estimated_read_time_seconds - summary_stats.estimated_read_time_seconds)
    )
  );

  return (
    <div className="metrics-grid">
      {/* Metric 1: Compression Ratio */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Compression Ratio</span>
          <span className="metric-badge badge-cyan">{compressionPercent}% condensed</span>
        </div>
        <div className="metric-value font-mono">
          {compressionPercent}
          <span className="metric-unit">%</span>
        </div>
        <div className="metric-footnote">
          {input_stats.word_count} words → {summary_stats.word_count} words
        </div>
      </div>

      {/* Metric 2: Inference Latency & Cache Status */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Inference Latency</span>
          {cache_hit ? (
            <span className="metric-badge badge-emerald">⚡ Cache Hit (Instant)</span>
          ) : (
            <span className="metric-badge badge-indigo">torch.no_grad</span>
          )}
        </div>
        <div className="metric-value font-mono">
          {cache_hit ? '0.00' : latency_seconds}
          <span className="metric-unit">s</span>
        </div>
        <div className="metric-footnote">
          {cache_hit
            ? 'Zero CPU compute • Served via SQLite hash cache'
            : 'CPU execution time on host'}
        </div>
      </div>

      {/* Metric 3: ROUGE Evaluation Scores */}
      {rouge_scores && (
        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-title">Evaluation Metrics</span>
            <span className="metric-badge badge-indigo">ROUGE F1</span>
          </div>
          <div className="metric-value font-mono">
            {(rouge_scores.rouge1.fmeasure * 100).toFixed(1)}
            <span className="metric-unit">%</span>
          </div>
          <div className="metric-footnote">
            R-1: {(rouge_scores.rouge1.fmeasure * 100).toFixed(1)}% • R-2: {(rouge_scores.rouge2.fmeasure * 100).toFixed(1)}% • R-L: {(rouge_scores.rougeL.fmeasure * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {/* Metric 4: Memory Footprint */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Process RSS Memory</span>
          <span className="metric-badge badge-emerald">RAM Safe</span>
        </div>
        <div className="metric-value font-mono">
          {rss_memory_mb}
          <span className="metric-unit">MB</span>
        </div>
        <div className="metric-footnote">
          Post-inference gc.collect() active
        </div>
      </div>

      {/* Metric 5: Human Reading Time Saved */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Reading Time Saved</span>
          <span className="metric-badge badge-purple">Efficiency</span>
        </div>
        <div className="metric-value font-mono">
          {timeSaved}
          <span className="metric-unit">sec</span>
        </div>
        <div className="metric-footnote">
          ~{(timeSaved / 60).toFixed(1)} minutes saved per paper
        </div>
      </div>
    </div>
  );
}
