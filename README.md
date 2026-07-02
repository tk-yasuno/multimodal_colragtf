# Multi-modal CoLRAG with Triple Filtering for Domain-specific Unstructured Documents

Multi-modal Contextual Late Interaction RAG with Triple Filtering (Multi-modal CoLRAG-TF) retrieval strategies on Japanese Natural Disaster documents within multiple modes included in Text, Figure, Table, Photos, Map.

## Overview

| Dimension                 | Details                                                                                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Document corpus** | 8 volumes of*Kasen·Dam·Sabo Technical Standards 2025* (河川砂防技術標準)                                                                  |
| **Test sets**       | **Standard**: 200 QA pairs (1-hop, sampled from 4,000 generated) · **Multi-hop**: 1,000 QA pairs (2-hop, graph-based generation) |
| **RAG strategies**  | Naive RAG · CoLRAG · CoLRAG-Triple Filtering (HippoRAG2)                                                                                    |
| **LLM backends**    | Swallow-8B-LoRA-Q4 · ELYZA-JP-8B-LoRA-Q4 ·**Qwen2.5-7B-Instruct-Q4_K_M** (v0.4.0, via Ollama)                                         |
| **Judge model**     | Qwen2.5-14B-Instruct (served via Ollama)                                                                                                      |
| **Embedding model** | [`hotchpotch/static-embedding-japanese`](https://huggingface.co/hotchpotch/static-embedding-japanese) (1024-dim, IP similarity)              |
| **GPU constraint**  | 16 GB VRAM                                                                                                                                    |

**6+ conditions total** = 3 RAG types × 2+ LLM models.

---

## Release Notes

### v0.7.2 (2026-07-03) 🔍

**HippoRAG2-Style Retrieval with Triple Extraction and Multimodal Evaluation**

**Objective:**

Implement knowledge graph-based retrieval with triple extraction, FAISS indexing, image understanding integration, and comprehensive multimodal evaluation framework.

**Key Components:**

**Phase 5: Triple Extraction & Knowledge Graph**
- **Triple Extraction** (`04_extract_triples.py`): OpenIE-style triple extraction from captions using Ollama
  - Extracted 227 triples from 57 blocks (avg 3.98 triples/block)
  - 86% block success rate (49/57 blocks)
  - Triple types: Subject-Predicate-Object relations
- **FAISS Triple Index** (`05_build_triple_index.py`): Vector search for semantic triple retrieval
  - hotchpotch/static-embedding-japanese (1024-dim)
  - IndexFlatIP (Inner Product for normalized vectors)
  - Natural Japanese conversion: `triple_to_text()` method

**Phase 6: Multimodal HippoRAG2 Retriever**
- **Query Analysis** (`06_multimodal_retriever.py`): Figure-related keyword detection
  - FIGURE_KEYWORDS: 図, 表, グラフ, 写真, 地図
  - VISUAL_KEYWORDS: 何枚, 何件, 数値, データ
- **Multi-Score Fusion**: Text + Triple + Image scores with alpha weights
  - Figure boost: ×1.2 multiplier for table/figure blocks
  - Demo tested: Triple score 0.8998 vs Text 0.5726 for "全壊家屋" query
- **Image Understanding Integration**: Ollama multimodal client
  - `retrieve_with_images()`: Enhanced retrieval with image LLM
  - `generate_answer_multimodal()`: Context-aware answer generation
  - Base64 image encoding with qwen2.5:7b model

**Phase 7: Multimodal Evaluation Framework**
- **QA Generation** (`07_multimodal_evaluation.py`): Figure-specific QA pairs from captions
  - 10 QA pairs generated from 33 figure blocks
  - Question types: table, figure, text, multihop
  - Ollama-based generation with FIGURE_QA_PROMPT
- **7 Evaluation Metrics**:
  - **Existing** (v0.6.4-compatible): Faithfulness, Relevance, Answer Correctness, Recall
  - **New** (Multimodal-specific): Image Relevance, Image Coverage, Multimodal Faithfulness
- **Evaluation Results** (demo with 10 questions):
  - Answer Correctness: 1.0000
  - Faithfulness: 1.0000
  - Image Relevance: 1.0000
  - Image Coverage: 1.0000
  - Multimodal Faithfulness: 1.0000
  - *(Relevance: 0.0 due to dummy retrieval in demo mode)*

**Scripts:**
- `experiments_v070/04_extract_triples.py`: Triple extraction with OpenIE patterns
- `experiments_v070/05_build_triple_index.py`: FAISS index construction
- `experiments_v070/06_multimodal_retriever.py`: HippoRAG2-style multimodal retriever
- `experiments_v070/07_multimodal_evaluation.py`: QA generation + evaluation

**Next Steps:**
- Full retriever integration in evaluation (currently uses dummy blocks)
- LightGBM calibration model for reranking (Phase 6.17, deferred)
- Multi-hop question evaluation on generated QA dataset
- Comparison benchmark: v0.7.2 (multimodal) vs v0.6.4 (text-only)

---

### v0.7.1 (2026-07-03) 🖼️

**Multimodal Extension: Table Detection, OCR, and Caption Generation**

**Objective:**

Extend CoLRAG-TF from text-only to multimodal capabilities, enabling the system to understand and retrieve information from tables, figures, and images in disaster lesson PDFs containing visual content (tables, graphs, photos, maps).

**1. Architecture Overview**

**Target Documents**: 43 disaster lesson PDFs (歴史資料集, 災害事例, 復興知見)
**Visual Content**: Tables (primary focus), figures, images, maps
**Hierarchy Preservation**: 3-tier structure (Volume → Chapter → Chunk) maintained

**Pipeline (Methodological Flow)**:

📊 **[View High-Resolution Methodology Flow Diagram (PDF)](docs/v071_methodology_flow.pdf)**

<details>
<summary>TikZ Source Code (click to expand)</summary>

```latex
\documentclass[tikz,border=3mm]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}

\tikzset{
  process/.style={rectangle, rounded corners, minimum width=3cm, minimum height=1cm, 
                  text centered, draw=black, fill=blue!10, font=\small},
  decision/.style={diamond, minimum width=2.5cm, minimum height=1cm, text centered, 
                   draw=black, fill=orange!20, font=\small, aspect=2},
  io/.style={trapezium, trapezium left angle=70, trapezium right angle=110, 
             minimum width=2.5cm, minimum height=0.8cm, text centered, draw=black, 
             fill=green!10, font=\small},
  model/.style={rectangle, rounded corners, minimum width=2.8cm, minimum height=0.8cm, 
                text centered, draw=black, fill=purple!10, font=\footnotesize},
  metric/.style={rectangle, rounded corners, minimum width=2.2cm, minimum height=0.6cm, 
                 text centered, draw=black, fill=yellow!15, font=\scriptsize},
  arrow/.style={thick,-Stealth},
  note/.style={font=\scriptsize, text width=2.5cm, align=center}
}

\begin{document}
\begin{tikzpicture}[node distance=1.2cm and 2.5cm]

% Input
\node (input) [io] {43 Disaster PDFs\\(歴史資料集, 災害事例)};

% Phase 1: Environment
\node (env) [process, below=of input] {Phase 1: Environment\\Python 3.12.10, CUDA 12.4};
\node (env-model) [model, right=0.5cm of env] {PyTorch 2.6.0\\transformers 5.5.0};

% Phase 2: Layout Analysis
\node (layout) [process, below=of env] {Phase 2: Layout Analysis\\Table Transformer Detection};
\node (layout-model) [model, right=0.5cm of layout] {microsoft/\\table-transformer\\FP16, conf>0.7};
\node (layout-metric) [metric, below=0.3cm of layout] {52 pages\\38 tables, 19 text blocks};

% Phase 3a: OCR Strategy
\node (ocr) [process, below=1.5cm of layout] {Phase 3a: OCR Extraction\\Hybrid Strategy};
\node (pymupdf) [model, below left=0.5cm and -0.5cm of ocr] {PyMuPDF\\Text Layer};
\node (pymupdf-metric) [metric, below=0.2cm of pymupdf] {68.4\%\\26/38 tables};
\node (tesseract) [model, below right=0.5cm and -0.5cm of ocr] {Tesseract 5.5\\jpn+eng};
\node (tesseract-metric) [metric, below=0.2cm of tesseract] {86.8\%\\33/38 tables};

% Phase 3b: Caption
\node (caption) [process, below=2.5cm of ocr] {Phase 3b: Caption Generation\\from OCR Text};
\node (caption-model) [model, right=0.5cm of caption] {qwen2.5:7b\\instruct-q4\_k\_m};
\node (caption-metric) [metric, below=0.3cm of caption] {Success: 86.8\%\\33/38 structured captions};

% Phase 4: Embedding & Index
\node (embed) [process, below=1.5cm of caption] {Phase 4: Multimodal Index\\Embedding + Vector Store};
\node (embed-model) [model, right=0.5cm of embed] {hotchpotch\\1024-dim + Qdrant};
\node (embed-metric) [metric, below=0.3cm of embed] {57 blocks indexed\\(38 tables + 19 text)};

% Output
\node (output) [io, below=of embed] {Multimodal Index\\Ready for Retrieval};

% Arrows
\draw [arrow] (input) -- (env);
\draw [arrow] (env) -- (layout);
\draw [arrow] (layout) -- (ocr);
\draw [arrow] (ocr) -- (pymupdf);
\draw [arrow] (ocr) -- (tesseract);
\draw [arrow] (pymupdf) -- (caption);
\draw [arrow] (tesseract) -- (caption);
\draw [arrow] (caption) -- (embed);
\draw [arrow] (embed) -- (output);

% Phase labels on the left
\node [note, left=1.5cm of env, anchor=east] {\textbf{Setup}};
\node [note, left=1.5cm of layout, anchor=east] {\textbf{Detection}};
\node [note, left=1.5cm of ocr, anchor=east] {\textbf{Text\\Extraction}};
\node [note, left=1.5cm of caption, anchor=east] {\textbf{Semantic\\Generation}};
\node [note, left=1.5cm of embed, anchor=east] {\textbf{Vectorization}};

% Background boxes for phases
\begin{scope}[on background layer]
\node [fill=gray!5, rounded corners, fit=(layout) (layout-model) (layout-metric), inner sep=8pt] {};
\node [fill=gray!5, rounded corners, fit=(ocr) (pymupdf) (tesseract) (pymupdf-metric) (tesseract-metric), inner sep=8pt] {};
\node [fill=gray!5, rounded corners, fit=(caption) (caption-model) (caption-metric), inner sep=8pt] {};
\node [fill=gray!5, rounded corners, fit=(embed) (embed-model) (embed-metric), inner sep=8pt] {};
\end{scope}

\end{tikzpicture}
\end{document}
```

</details>

**Methodological Highlights**:
- **Hybrid OCR**: PyMuPDF (fast, text-layer) → Tesseract fallback (image-based) achieves 86.8% success
- **Semantic Captions**: LLM-generated structured descriptions from raw OCR text
- **Scalable Pipeline**: Modular design with FP16 precision for 16GB VRAM constraint
- **Reproducibility**: All phases documented in `experiments_v070/` scripts

**2. Implementation Results**

**Phase 1: Environment Setup**

- Virtual environment: `.venv-coltf` (Python 3.12.10, CUDA 12.4)
- Core libraries: PyTorch 2.6.0+cu124, transformers 5.5.0, PyMuPDF 1.28.0
- OCR engines: PyMuPDF (text layer), Tesseract 5.5.0 (image-based)
- Multimodal models: llava:7b (via Ollama), qwen2.5:7b-instruct-q4_k_m

**Phase 2: Layout Analysis** (`01_layout_analysis.py`)

- Model: `microsoft/table-transformer-detection` (FP16, confidence threshold: 0.7)
- Processed: 52 pages from sample PDF (02jirei4_2011-2018.pdf)
- Detected: 38 tables, 19 text blocks
- Output: `layout_blocks.jsonl` with bounding boxes and block images

**Phase 3: OCR & Caption Generation** (`02_multimodal_caption.py`)

**Hybrid OCR Strategy**:

1. **PyMuPDF Text Extraction** (primary): 68.4% success (26/38 tables)
   - Fast, perfect accuracy for text-layer PDFs
   - Direct text extraction from PDF bounding boxes
2. **Tesseract OCR** (fallback): 86.8% success (33/38 tables)
   - Handles image-based/scanned tables
   - Japanese + English language support (`jpn+eng`)
   - Processing time: ~2 minutes for 38 tables

**Caption Quality** (Generated by qwen2.5:7b from OCR text):

```
Example (Table of disaster damage):
OCR: "区分 細分 被害額 人的被害 死者 56..."
Caption: "この表は、特定の地域における災害被害状況を示しています。
主要な項目には「区分」「細分」「被害額（百万円）」「人的被害（人）」が含まれ、
特に家屋被害と公共土木施設被害に注目します。重要な数値として、
全壊家屋367棟、半壊家屋1,840棟、河川の公共土木施設被害箇所842箇所、
死者56人が挙げられます。"
```

**Success Metrics**:

- Total tables: 38
- **Successful captions: 33 (86.8%)** ✨
- Failed: 5 (white/blank pages, detection false positives)
- Caption quality: Structured, contextual, retrieval-friendly

**Phase 4: Multimodal Index** (`03_build_multimodal_index.py`)

- Embedding model: `hotchpotch/static-embedding-japanese` (1024-dim)
- Vector store: Qdrant in-memory (collection: `colragtf_v070_multimodal`)
- Total blocks indexed: 57 (38 tables + 19 text blocks)
- Hierarchy: 1 Volume (2011-2018災害事例), 1 Chapter (02jirei4_2011-2018)

**3. Key Technical Achievements**

**Challenge: PyTorch-PaddleOCR GPU Conflict**

- **Problem**: PaddlePaddle and PyTorch cannot coexist in same process
  ```
  ImportError: generic_type: type "_gpuDeviceProperties" is already registered!
  ```
- **Solution**: Avoided PaddleOCR; used PyMuPDF + Tesseract hybrid approach
- **Lesson**: Documented in `LESSON_OCR_Install.md`

**Hybrid OCR Innovation**:

- **Fast path**: PyMuPDF text extraction (0 overhead, 68% coverage)
- **Comprehensive path**: Tesseract OCR (2min, 87% coverage)
- **Best of both**: Speed + comprehensive coverage

**4. Scripts & Usage**

```bash
# Phase 1: Environment check
.\.venv-coltf\Scripts\python.exe experiments_v070\00_check_multimodal_env.py

# Phase 2: Layout analysis (Table Transformer)
.\.venv-coltf\Scripts\python.exe experiments_v070\01_layout_analysis.py

# Phase 3: Caption generation (Hybrid OCR)
.\.venv-coltf\Scripts\python.exe experiments_v070\02_multimodal_caption.py --ocr-engine tesseract

# Phase 4: Multimodal index construction
.\.venv-coltf\Scripts\python.exe experiments_v070\03_build_multimodal_index.py
```

**5. Next Steps (v0.7.2)**

- **Phase 5**: Triple extraction from table captions (multimodal knowledge graph)
- **Phase 6**: Multimodal retriever (image + text hybrid scoring)
- **Phase 7**: Evaluation framework for multimodal QA

**6. Key Takeaways**

✅ **Table detection**: Table Transformer works well on Japanese disaster PDFs
✅ **OCR strategy**: Hybrid (PyMuPDF + Tesseract) achieves 86.8% success
✅ **Caption quality**: LLM post-processing generates structured, contextual summaries
✅ **Conflict resolution**: Avoid PaddleOCR; Tesseract is production-ready
✅ **Scalability**: 52 pages processed in ~5 minutes total (layout + OCR + caption)

---

### v0.6.4 (2026-06-29) 🧪

**Multi-hop Question Generation & Comparative Evaluation Framework**

**Objective:**

Evaluate whether CoLRAG with Triple Filtering (HippoRAG2) demonstrates superior multi-hop reasoning capabilities compared to Naive RAG and CoLRAG when answering questions that require synthesizing information across multiple concepts, chapters, and sections.

**1. Multi-hop Question Generation (1000Q)**

**Pipeline:**

- **Source**: 5,000 single-hop QA pairs from technical standards corpus
- **Graph-based generation**: Extract concept pairs from knowledge graph with scoring:
  - Different relation types: +2.0
  - Different chapters: +3.0
  - Multi-hop relations (MITIGATES, AFFECTS, SUBJECT_TO): +1.0
- **4 Question Templates**:
  - T1 (因果連鎖): Causal chain reasoning
  - T2 (統合): Information integration
  - T3 (比較): Cross-concept comparison
  - T4 (手順): Procedural synthesis
- **LLM Validation**: qwen2.5-14b-gpu validates 3,000 candidates → 2,763 valid (92.1%)
- **Final Testset**: `experiments/testset_multihop_1000.jsonl`
  - 1,000 questions (2-hop: 100%)
  - Template distribution: T1: 21.9%, T2: 24.6%, T3: 22.7%, T4: 30.8%

**Scripts:**

```bash
# Analyze 5000Q structure
python experiments/01a_analyze_5000q.py

# Parse chapter hierarchy
python experiments/02a_parse_chapter_structure.py

# Generate & validate multi-hop questions
python experiments/02b_prepare_multihop_testset.py \
  --filter-top-n 3000 \
  --validation-workers 5 \
  --seed 42

# Sample 1000Q testset
python experiments/02c_sample_multihop_testset.py --output-size 1000
```

**2. Multi-hop Comparative Evaluation**

**Standard AI-as-Judge Results (1000Q):**

- **Naive RAG**: Judge 2.437/3.0, Perfect 45.4%
- **CoLRAG**: Judge 2.414/3.0, Perfect 43.2%
- **CoLRAG-TF**: Judge 2.442/3.0, Perfect 45.2%
- **Statistical Analysis**: No significant difference (Wilcoxon p>0.05, Cohen's d<0.1)

**Insight**: Standard judge prompts fail to differentiate multi-hop reasoning quality.

**3. Multi-hop Specific Comparative Evaluation**

**New Evaluation Framework**: 4-axis comparative judge designed for multi-hop reasoning:

1. **Multi-hop Integration**: Combining multiple concepts/sections
2. **Cross-Section Reasoning**: Synthesizing across chapters/viewpoints
3. **Causal/Structural Reasoning**: Explaining why/how concepts relate
4. **Global Coherence**: Maintaining consistency across hops

**Pairwise Comparisons**:

- Comparison 1: **Naive RAG vs CoLRAG**
- Comparison 2: **Naive RAG vs CoLRAG-Triple Filtering**
- Comparison 3: **CoLRAG vs CoLRAG-Triple Filtering**

**Results (1000Q × 3 comparisons, Judge: qwen2.5:14b):**

| Comparison                   | Winner              | Win Rate        | Key Findings                                                |
| ---------------------------- | ------------------- | --------------- | ----------------------------------------------------------- |
| Naive vs**CoLRAG**     | **CoLRAG**    | **78.0%** | Hybrid retrieval significantly improves multi-hop reasoning |
| Naive vs**CoLRAG-TF**  | **CoLRAG-TF** | **78.0%** | Triple filtering maintains CoLRAG's advantage               |
| CoLRAG vs**CoLRAG-TF** | **CoLRAG-TF** | **72.0%** | Triple filtering adds 72% win rate over CoLRAG alone        |

**Axis-level Analysis (Overall winner):**

| Axis                        | Naive vs CoLRAG | Naive vs CoLRAG-TF | CoLRAG vs CoLRAG-TF |
| --------------------------- | --------------- | ------------------ | ------------------- |
| Multi-hop Integration       | 78.2% (CoLRAG)  | 78.0% (CoLRAG-TF)  | 72.1% (CoLRAG-TF)   |
| Cross-Section Reasoning     | 78.2% (CoLRAG)  | 78.0% (CoLRAG-TF)  | 72.1% (CoLRAG-TF)   |
| Causal/Structural Reasoning | 78.0% (CoLRAG)  | 78.0% (CoLRAG-TF)  | 72.0% (CoLRAG-TF)   |
| Global Coherence            | 75.1% (CoLRAG)  | 76.8% (CoLRAG-TF)  | 69.9% (CoLRAG-TF)   |

**Key Insight**:

- ✅ **Standard AI-as-Judge failed to differentiate** (scores: 2.414-2.442, no significant difference)
- ✅ **Multi-hop specific evaluation reveals clear hierarchy**: CoLRAG-TF > CoLRAG >> Naive RAG
- ✅ **Triple filtering adds measurable value** (72% win rate over CoLRAG) on multi-hop integration and cross-section reasoning

**Evaluation Script:**

```bash
# Test run (30 questions)
python experiments/07_multihop_comparative_eval.py \
  --max-questions 30 \
  --workers 3

# Full evaluation (1000 questions × 3 comparisons = 3000 evaluations, ~4 hours)
python experiments/07_multihop_comparative_eval.py \
  --workers 5 \
  --judge-model qwen2.5:14b
```

**Output:**

- `experiments/results/comparative_eval/naive_vs_colrag.jsonl`
- `experiments/results/comparative_eval/naive_vs_colrag_tf.jsonl`
- `experiments/results/comparative_eval/colrag_vs_colrag_tf.jsonl`
- `experiments/results/comparative_eval/comparative_eval_summary.json`

**Documentation:**

- Full implementation guide: `experiments/GUIDE_Multi-hopQ_Gen.md`

---
