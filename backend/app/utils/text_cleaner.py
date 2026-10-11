"""Scientific and academic text cleaning utility.

Preprocesses academic papers, abstracts, and computational linguistics manuscripts:
- Strips LaTeX commands, inline math, and equation environments
- Removes numerical citation markers (e.g., [1], [2, 3]) and author-year citations
- Normalizes typographic ligatures (fi, fl) and Unicode whitespace
- Eliminates URLs, DOIs, and extraneous section titles
"""

import re
import unicodedata
from typing import Dict, Any


# Common typographical ligatures to replace
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2014": " - ",
    "\u2013": " - ",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters and replace typography ligatures."""
    for ligature, replacement in LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)
    return unicodedata.normalize("NFKC", text)


def remove_latex_markup(text: str) -> str:
    """Remove LaTeX formatting, math environments, and citations."""
    # Remove equation environments: \begin{equation}...\end{equation}
    text = re.sub(r"\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\}", " ", text, flags=re.DOTALL)

    # Remove display math $$ ... $$ and inline math $ ... $
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)

    # Remove \cite{...}, \citet{...}, \citep{...}, \ref{...}, \label{...}
    text = re.sub(r"\\cite[tp]?\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:ref|eqref|label|pageref)\{[^}]*\}", " ", text)

    # Strip inline styling commands like \textbf{content} -> content
    text = re.sub(r"\\(?:textbf|textit|emph|textsc|underline|texttt)\{([^}]*)\}", r"\1", text)

    # Remove residual commands like \item, \section, \noindent
    text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{([^}]*)\})?", r"\3 ", text)

    return text


def remove_citations(text: str) -> str:
    """Remove numbered brackets [1], [1, 2], [1-4] and author-year (Smith et al., 2020)."""
    # Bracketed numbers: [1], [1, 2], [12-15]
    text = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", " ", text)

    # Parenthetical author-year: (Vaswani et al., 2017) or (Chomsky, 1957; Manning, 2014)
    text = re.sub(
        r"\([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?(?:,?\s+\d{4}[a-z]?(?:;\s+[A-Z][a-zA-Z]+(?:\s+et\s+al\.)?,?\s+\d{4}[a-z]?)*)\)",
        " ",
        text,
    )

    return text


def remove_urls_and_dois(text: str) -> str:
    """Remove web URLs and DOI identifiers."""
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # DOIs (e.g. 10.1000/182 or doi:10.xxx)
    text = re.sub(r"(?:doi:\s*)?10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", " ", text, flags=re.IGNORECASE)
    # ArXiv identifiers (e.g. arXiv:1706.03762)
    text = re.sub(r"arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?", " ", text, flags=re.IGNORECASE)
    return text


def remove_section_headers(text: str) -> str:
    """Remove boilerplate section headers like '1. Introduction' or 'Abstract.'."""
    text = re.sub(
        r"^(?:(?:\d+\.|\b[IVXLCDM]+\.)\s+)?(?:Abstract|Introduction|Related\s+Work|Methodology|Experiments|Results|Discussion|Conclusion)\b[:.]?\s*",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return text


def clean_academic_text(raw_text: str) -> str:
    """Full cleaning pipeline for academic literature and computational linguistics papers."""
    if not raw_text or not raw_text.strip():
        return ""

    text = normalize_unicode(raw_text)
    text = remove_latex_markup(text)
    text = remove_citations(text)
    text = remove_urls_and_dois(text)
    text = remove_section_headers(text)

    # Collapse hyphenated line-breaks (e.g., 'transfor-\nmer' -> 'transformer')
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Collapse multiple whitespaces and tabs into single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize paragraph breaks (max 2 consecutive newlines)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def compute_text_stats(text: str) -> Dict[str, Any]:
    """Compute structural metrics from academic text."""
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "estimated_read_time_seconds": 0,
        }

    words = [w for w in text.split() if w]
    word_count = len(words)
    char_count = len(text)
    # Approximate sentence boundaries by period, exclamation, or question marks
    sentences = re.findall(r"[^.!?]+(?:[.!?]+|$)", text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Average human reading speed: ~200 words per minute (~3.3 words/sec)
    estimated_read_time = round(word_count / 3.3, 1)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "estimated_read_time_seconds": estimated_read_time,
    }


def normalize_pdf_text(text: str) -> str:
    """Clean and normalize raw text extracted from PDF document pages."""
    if not text:
        return ""

    text = normalize_unicode(text)
    # Collapse hyphenated line-breaks (e.g., 'transfor-\nmer' -> 'transformer')
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Convert isolated newlines within sentences to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple newlines between paragraphs
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# Curated academic stopword list (pure Python, <10KB memory)
ACADEMIC_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
    # Academic meta-lexicon
    "paper", "papers", "study", "studies", "approach", "approaches", "method", "methods", "methodology",
    "propose", "proposes", "proposed", "results", "result", "demonstrate", "demonstrates", "demonstrated",
    "using", "uses", "used", "shows", "shown", "show", "based", "also", "model", "models", "data",
    "work", "works", "et", "al", "evaluation", "evaluates", "evaluated", "performance", "task", "tasks",
    "present", "presents", "presented", "experiments", "experimental", "findings", "compare", "compared",
    "analysis", "section", "table", "figure", "significant", "significantly", "novel", "new", "well",
    "two", "three", "one", "first", "second", "different", "existing", "large", "small", "across"
}


def extract_statistical_keywords(text: str, top_k: int = 5) -> list[str]:
    """Extract top scientific keywords/phrases using a pure-Python statistical frequency approach.

    Runs in <2ms with zero external NLP/tokenizer dependencies (RAM footprint < 1MB).
    """
    if not text:
        return []

    from collections import Counter

    raw_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b", text.lower())
    content_tokens = [w for w in raw_tokens if w not in ACADEMIC_STOPWORDS and len(w) > 2]

    # Contiguous bigram phrases
    bigrams = [
        f"{raw_tokens[i]} {raw_tokens[i+1]}"
        for i in range(len(raw_tokens) - 1)
        if raw_tokens[i] not in ACADEMIC_STOPWORDS and raw_tokens[i+1] not in ACADEMIC_STOPWORDS
    ]

    unigram_counts = Counter(content_tokens)
    bigram_counts = Counter(bigrams)

    candidate_scores: Dict[str, float] = {}

    # Boost repeating multi-word scientific collocations
    for bg, count in bigram_counts.items():
        if count > 1:
            candidate_scores[bg] = count * 2.2
    for ug, count in unigram_counts.items():
        candidate_scores[ug] = count * 1.0

    # Fallback to single bigrams if candidate pool is small
    if len(candidate_scores) < top_k:
        for bg, count in bigram_counts.items():
            if bg not in candidate_scores:
                candidate_scores[bg] = count * 0.95

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

    selected: list[str] = []
    for cand, _ in sorted_candidates:
        cand_clean = cand.title()
        # Prevent overlapping subphrases (e.g., "Transformer" if "Transformer Model" is present)
        if not any(cand in s.lower() for s in selected):
            selected.append(cand_clean)
        if len(selected) >= top_k:
            break

    return selected


def compute_rouge_scores(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores using rouge-score without stemmer.

    use_stemmer=False prevents NLTK download prompts ('punkt') inside containers.
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)

        return {
            "rouge1": {
                "precision": round(scores["rouge1"].precision, 4),
                "recall": round(scores["rouge1"].recall, 4),
                "fmeasure": round(scores["rouge1"].fmeasure, 4),
            },
            "rouge2": {
                "precision": round(scores["rouge2"].precision, 4),
                "recall": round(scores["rouge2"].recall, 4),
                "fmeasure": round(scores["rouge2"].fmeasure, 4),
            },
            "rougeL": {
                "precision": round(scores["rougeL"].precision, 4),
                "recall": round(scores["rougeL"].recall, 4),
                "fmeasure": round(scores["rougeL"].fmeasure, 4),
            },
        }
    except Exception:
        # Fallback empty metrics on any unexpected computation failure
        return {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        }


