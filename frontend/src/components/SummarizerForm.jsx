import React, { useState, useEffect, useRef } from 'react';

const PRESET_PAPERS = [
  {
    title: 'Attention Is All You Need (Vaswani et al., 2017)',
    text: `The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.`
  },
  {
    title: 'SciBERT: Pretrained Language Model for Scientific Text (Beltagy et al., 2019)',
    text: `Obtaining pre-trained language models has drastically improved performance on many natural language processing (NLP) tasks. Although pre-training on general domain text such as Wikipedia and BooksCorpus is standard practice, transferring such representations to specialized domains like scientific literature frequently yields suboptimal performance due to domain shift and specialized vocabulary. To address this challenge, we present SciBERT, a pretrained language model based on BERT trained on a large corpus of scientific text. SciBERT leverages a custom scientific vocabulary (SCIVOCAB) constructed from scientific papers to capture specialized terms in computational linguistics, biomedicine, and computer science. We evaluate SciBERT across multiple scientific NLP benchmarks, including named entity recognition, relation extraction, and text classification, demonstrating substantial state-of-the-art improvements over standard BERT across diverse datasets.`
  },
  {
    title: 'LongT5: Efficient Text-to-Text Transformer for Long Sequences (Guo et al., 2021)',
    text: `Recent advances in Transformer models have achieved remarkable success across NLP benchmarks. However, the standard quadratic memory and computational complexity of self-attention mechanisms with respect to input sequence length strictly prohibits their application to long-document understanding and academic literature summarization. We present LongT5, an efficient sequence-to-sequence model designed to process thousands of tokens in a single forward pass. LongT5 integrates Transient Global (TGlobal) attention, combining local sliding-window attention with transient global representations to scale efficiently to extended sequence lengths while maintaining competitive summarization performance. Our evaluations on long-document summarization datasets show that LongT5 sets new state-of-the-art results while reducing memory overhead by up to 70% compared to dense attention baselines.`
  }
];

export default function SummarizerForm({ onSummarize, isLoading }) {
  const [inputText, setInputText] = useState('');
  const [maxTokens, setMaxTokens] = useState(256);
  const [minTokens, setMinTokens] = useState(40);
  const [elapsedTime, setElapsedTime] = useState(0);

  // PDF Upload & Drag-and-Drop state
  const [isDragging, setIsDragging] = useState(false);
  const [isUploadingPdf, setIsUploadingPdf] = useState(false);
  const [pdfUploadError, setPdfUploadError] = useState(null);
  const [pdfMeta, setPdfMeta] = useState(null);
  const fileInputRef = useRef(null);

  // Loading timer simulation & tracking
  useEffect(() => {
    let timer;
    if (isLoading) {
      setElapsedTime(0);
      timer = setInterval(() => {
        setElapsedTime((prev) => +(prev + 0.1).toFixed(1));
      }, 100);
    } else {
      setElapsedTime(0);
    }
    return () => clearInterval(timer);
  }, [isLoading]);

  const charCount = inputText.length;
  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
  const isTooShort = charCount > 0 && charCount < 50;
  const isTooLong = charCount > 50000;
  const isValid = charCount >= 50 && charCount <= 50000;
  const isChunkingActive = charCount > 15000 && charCount <= 50000;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isValid || isLoading) return;
    onSummarize({
      text: inputText,
      max_summary_tokens: maxTokens,
      min_summary_tokens: minTokens
    });
  };

  const loadPreset = (preset) => {
    setInputText(preset.text);
    setPdfMeta(null);
    setPdfUploadError(null);
  };

  const handleClear = () => {
    setInputText('');
    setPdfMeta(null);
    setPdfUploadError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // PDF processing pipeline
  const processPdfFile = async (file) => {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setPdfUploadError('Invalid file type. Please upload a PDF document (.pdf).');
      return;
    }

    const maxBytes = 5 * 1024 * 1024; // 5 MB
    if (file.size > maxBytes) {
      setPdfUploadError(`File exceeds 5MB limit (${(file.size / (1024 * 1024)).toFixed(2)} MB). Please select a smaller academic document.`);
      return;
    }

    setPdfUploadError(null);
    setIsUploadingPdf(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/v1/extract-pdf', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || 'Failed to extract text from PDF.');
      }

      const data = await res.json();
      setInputText(data.extracted_text);
      setPdfMeta({
        filename: data.filename,
        page_count: data.page_count,
        char_count: data.char_count,
      });
    } catch (err) {
      setPdfUploadError(err.message || 'Error occurred while processing PDF.');
    } finally {
      setIsUploadingPdf(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processPdfFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processPdfFile(e.target.files[0]);
    }
  };

  return (
    <div className="academic-card form-card">
      <div className="card-header">
        <div>
          <h2 className="card-title">Academic Text Corpus</h2>
          <p className="card-description">
            Input scientific paper abstract, full-text excerpts, or upload a research PDF directly.
          </p>
        </div>

        {/* Preset Selector */}
        <div className="preset-container">
          <span className="preset-label">Benchmark Corpus:</span>
          <div className="preset-buttons">
            {PRESET_PAPERS.map((paper, idx) => (
              <button
                key={idx}
                type="button"
                className="preset-btn"
                onClick={() => loadPreset(paper)}
                disabled={isLoading || isUploadingPdf}
                title={paper.title}
              >
                Sample {idx + 1}
              </button>
            ))}
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* PDF Drag-and-Drop Zone */}
        <div
          className={`pdf-dropzone ${isDragging ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !isUploadingPdf && !isLoading && fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,application/pdf"
            style={{ display: 'none' }}
            disabled={isLoading || isUploadingPdf}
          />

          <svg className="pdf-dropzone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinecap="round" strokeLinejoin="round" />
            <polyline points="14 2 14 8 20 8" strokeLinecap="round" strokeLinejoin="round" />
            <line x1="12" y1="18" x2="12" y2="12" strokeLinecap="round" strokeLinejoin="round" />
            <polyline points="9 15 12 12 15 15" strokeLinecap="round" strokeLinejoin="round" />
          </svg>

          {isUploadingPdf ? (
            <div className="pdf-dropzone-status">
              <span className="spinner-icon" style={{ borderColor: 'var(--brand-navy)', borderTopColor: 'transparent' }}></span>
              <span>Extracting text from PDF (pypdf stream parser)...</span>
            </div>
          ) : (
            <div className="pdf-dropzone-content">
              <span className="pdf-dropzone-title">
                {isDragging ? 'Drop PDF file here' : 'Drop academic PDF here, or click to browse'}
              </span>
              <span className="pdf-dropzone-subtitle">
                Lightweight extraction via pypdf • Max 5MB • Populates input corpus below
              </span>
            </div>
          )}
        </div>

        {/* PDF Error Alert */}
        {pdfUploadError && (
          <div className="alert-error" style={{ marginBottom: '12px' }}>
            <span>⚠️ {pdfUploadError}</span>
            <button
              type="button"
              className="alert-close"
              onClick={() => setPdfUploadError(null)}
            >
              ✕
            </button>
          </div>
        )}

        {/* PDF Metadata Badge */}
        {pdfMeta && (
          <div className="pdf-meta-badge">
            <span>📄</span>
            <span>
              Extracted <strong>{pdfMeta.page_count}</strong> {pdfMeta.page_count === 1 ? 'page' : 'pages'} from <strong>{pdfMeta.filename}</strong> ({pdfMeta.char_count.toLocaleString()} characters)
            </span>
          </div>
        )}

        <div className="textarea-container">
          <textarea
            id="academic-input-text"
            rows="10"
            placeholder="Paste academic literature or manuscript text here (min 50 characters, max 50,000 characters)..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isLoading || isUploadingPdf}
          />
        </div>

        {/* Counter and Validation Bar */}
        <div className="counter-bar">
          <div className="counter-info">
            <span className={`count-badge font-mono ${isTooShort || isTooLong ? 'count-error' : 'count-ok'}`}>
              {charCount.toLocaleString()} / 50,000 characters
            </span>
            <span className="count-words font-mono">
              ({wordCount.toLocaleString()} words)
            </span>
          </div>

          {isTooShort && (
            <span className="validation-msg text-error">
              Minimum 50 characters required for model inference.
            </span>
          )}
          {isTooLong && (
            <span className="validation-msg text-error">
              Maximum character boundary exceeded (limit: 50,000 characters).
            </span>
          )}
        </div>

        {/* Model Token Sliders */}
        <div className="controls-row">
          <div className="control-group">
            <label htmlFor="max-tokens-slider" className="control-label">
              <span>Max Output Tokens:</span>
              <span className="font-mono text-cyan">{maxTokens}</span>
            </label>
            <input
              id="max-tokens-slider"
              type="range"
              min="50"
              max="512"
              step="16"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              disabled={isLoading || isUploadingPdf}
              className="slider"
            />
          </div>

          <div className="control-group">
            <label htmlFor="min-tokens-slider" className="control-label">
              <span>Min Output Tokens:</span>
              <span className="font-mono text-cyan">{minTokens}</span>
            </label>
            <input
              id="min-tokens-slider"
              type="range"
              min="20"
              max="150"
              step="10"
              value={minTokens}
              onChange={(e) => setMinTokens(Number(e.target.value))}
              disabled={isLoading || isUploadingPdf}
              className="slider"
            />
          </div>
        </div>

        {/* Actions & Loading State */}
        <div className="actions-row">
          {isChunkingActive && (
            <span
              className="badge-chunking-active"
              title="Map-Reduce chunking active: Input will be divided into ~12,000 character chunks and summarized sequentially"
            >
              <span>⚡</span>
              <span>Multi-pass chunking active: Inference may take 2-3 minutes</span>
            </span>
          )}

          <button
            type="button"
            className="btn-secondary"
            onClick={handleClear}
            disabled={isLoading || isUploadingPdf || !inputText}
          >
            Clear Input
          </button>

          <button
            type="submit"
            className="btn-primary"
            disabled={!isValid || isLoading || isUploadingPdf}
          >
            {isLoading ? (
              <span className="btn-loading-content">
                <span className="spinner-icon"></span>
                <span>Generating Summary ({elapsedTime}s)...</span>
              </span>
            ) : (
              <span>Generate Academic Summary</span>
            )}
          </button>
        </div>

        {/* Enterprise Progress Animation */}
        {isLoading && (
          <div className="inference-wait-box">
            <div className="progress-bar-wrapper">
              <div className="progress-bar-indeterminate"></div>
            </div>
            <div className="wait-status-text">
              <span>
                {isChunkingActive
                  ? 'Executing multi-pass Map-Reduce chunking & deterministic memory flushes on CPU...'
                  : elapsedTime < 1.2
                  ? 'Preprocessing LaTeX markup & generating SciBERT tokenization...'
                  : elapsedTime < 3.2
                  ? 'Executing LongT5 sequence-to-sequence beam search on CPU...'
                  : 'Reclaiming intermediate tensors and executing garbage collection...'}
              </span>
              <span className="font-mono text-xs opacity-75">Elapsed: {elapsedTime}s</span>
            </div>
          </div>
        )}
      </form>
    </div>
  );
}
