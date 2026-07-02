# CoLRAG-TF v0.7.0 - マルチモーダル拡張

**災害教訓PDFに対応したマルチモーダルRAGシステム**

## 概要

CoLRAG with Triple Filtering (CoLRAG-TF) v0.6.4 を拡張し、図・表・写真・地図を含む災害教訓PDFに対応したマルチモーダル版です。

### 主要な拡張機能

- **レイアウト解析**: Table Transformerによるテキスト/図表ブロック分離
- **マルチモーダルLLM**: Qwen2.5-Omni-7Bによる画像キャプション生成
- **マルチモーダル埋め込み**: テキスト＋画像の統合埋め込み（1024次元）
- **LlamaIndex + Qdrant**: マルチモーダルベクトルストア
- **拡張Triple Filtering**: 図表からの知識三つ組抽出
- **3-tier階層**: Volume (事例集カテゴリ) → Chapter (PDF) → Chunk (ページブロック)

## アーキテクチャ

```
災害教訓PDF (46件、~3,000ページ)
    ↓ [PyMuPDF: PDF→PNG変換]
ページ画像
    ↓ [Table Transformer: レイアウト解析]
テキストブロック + 図表ブロック
    ↓ [Qwen2.5-Omni: キャプション生成]
テキスト化データ
    ↓ [マルチモーダル埋め込み: hotchpotch + CLIP]
統合埋め込み (1024次元)
    ↓ [LlamaIndex + Qdrant]
マルチモーダルインデックス
    ↓ [Triple Filtering拡張]
Volume → Chapter → Block 検索
    ↓ [マルチモーダルLLM: 回答生成]
最終回答 (テキスト＋図表参照)
```

## セットアップ

### 1. 仮想環境

```powershell
# CoLRAG-TF v0.7.0 専用仮想環境
.\.venv-coltf\Scripts\Activate.ps1
```

### 2. 依存パッケージインストール

```powershell
pip install -r experiments_v070/requirements_v070.txt
```

**主要パッケージ:**
- LlamaIndex (`llama-index-core`, `llama-index-vector-stores-qdrant`)
- Qdrant (`qdrant-client`)
- PyMuPDF (`fitz`) - PDF処理
- Transformers (`transformers`) - Table Transformer
- PyTorch (`torch`, `torchvision`)
- OpenCLIP (`open-clip-torch`) - 画像埋め込み

### 3. Ollama マルチモーダルモデル

```powershell
# Qwen2.5-Omni（推奨: 7B量子化版）
ollama pull qwen2.5-omni:7b-instruct-q4_k_m

# または Qwen2.5-VL（ビジョン特化）
ollama pull qwen2.5-vl:7b

# オプション: llama-4-scout（詳細理解用）
ollama pull llama-4-scout:17b-16e-instruct-q4_k_m
```

### 4. 環境チェック

```powershell
python experiments_v070/00_check_multimodal_env.py
```

**確認項目:**
- ✅ Python 3.10+
- ✅ GPU / CUDA (16GB VRAM推奨)
- ✅ マルチモーダルパッケージ
- ✅ Ollama サービス稼働
- ✅ データディレクトリ存在

## データ構造

### Volume/Chapter階層

| Volume | 説明 | PDFs | 例 |
|--------|------|------|-----|
| **歴史資料集** | 昭和33年～平成21年事例 | 5 | `01jirei1-5_S33-S46-Kasen.pdf` |
| **2011-2018災害事例** | 平成23年～30年事例 | 9 | `02jirei1-9_H23-H30_1.pdf` |
| **2018-2019災害事例** | 平成30年～令和元年 | 5 | `03jirei1-5_H30-R1_1.pdf` |
| **災害教訓報告** | 主要災害詳細報告 | 13 | `disaster-H28_熊本地震.pdf` |
| **復興知見** | 復興ハンドブック等 | 3 | `202103_fukku-fukko-handbook_vol1.pdf` |

**合計**: 46 PDFs、~3,000ページ

### ディレクトリ構成

```
experiments_v070/
├── 00_check_multimodal_env.py        # 環境チェック
├── 01_layout_analysis.py             # レイアウト解析
├── 02_multimodal_caption.py          # キャプション生成
├── 03_build_multimodal_index.py      # インデックス構築
├── 04_extract_mm_triples.py          # Triple抽出
├── 05_multimodal_retrievers.py       # Retriever実装
├── 06_eval_multimodal_rag.py         # 評価
├── configs/
│   ├── layout_config.yaml            # レイアウト設定
│   └── mm_llm_config.yaml            # LLM設定
├── indices/                           # インデックス保存
│   ├── layout_blocks.jsonl
│   ├── block_images/                  # 切り出し画像
│   ├── qdrant_collection/
│   ├── mm_triples.json
│   └── mm_hierarchy.json
├── disaster_volume_mapping.json      # Volume定義
└── requirements_v070.txt             # 依存パッケージ
```

## 使用方法

### Phase 1: 環境確認

```powershell
python experiments_v070/00_check_multimodal_env.py
```

### Phase 2: レイアウト解析

```powershell
# 全PDFを解析
python experiments_v070/01_layout_analysis.py

# サンプル10ページのみ（デバッグ）
python experiments_v070/01_layout_analysis.py --sample 10

# 特定PDFのみ
python experiments_v070/01_layout_analysis.py --pdf "disaster-H28_熊本地震"
```

**出力:**
- `indices/layout_blocks.jsonl` - レイアウト情報
- `indices/block_images/` - 図表画像

### Phase 3: キャプション生成

```powershell
python experiments_v070/02_multimodal_caption.py
```

**処理:**
- 図表ブロックに対してQwen2.5-Omniでキャプション生成
- テキストブロックはOCR結果を使用
- バッチサイズ8、FP16混合精度

### Phase 4: マルチモーダルインデックス構築

```powershell
python experiments_v070/03_build_multimodal_index.py
```

**処理:**
- テキスト埋め込み: `hotchpotch/static-embedding-japanese` (1024次元)
- 画像埋め込み: CLIP → projection (1024次元)
- Qdrantコレクション構築
- Volume/Chapter代表ベクトル生成

### Phase 5: Triple Filtering

```powershell
python experiments_v070/04_extract_mm_triples.py
```

**処理:**
- 図表キャプションからOpenIE triple抽出
- Triple embedding生成
- FAISS インデックス構築

### Phase 6: RAG評価

```powershell
python experiments_v070/06_eval_multimodal_rag.py --retriever hipporag2 --testset testset_multimodal_200.jsonl
```

**評価指標:**
- Faithfulness (忠実性)
- Relevance (関連性)
- AnswerCorrectness (正答性)
- ImageRelevance (図表関連性)
- ImageCoverage (図表活用度)

## 設定ファイル

### layout_config.yaml

```yaml
model:
  name: "microsoft/table-transformer-detection"
  device: "cuda"
  precision: "fp16"

detection:
  confidence_threshold: 0.7
  target_types: ["table", "figure", "image"]
  
pdf:
  dpi: 150  # PDF→PNG解像度
```

### mm_llm_config.yaml

```yaml
models:
  primary:
    name: "qwen2.5-omni-7b"
    ollama_model: "qwen2.5-omni:7b-instruct-q4_k_m"
    batch_size: 8
    temperature: 0.2

embedding:
  text:
    model: "hotchpotch/static-embedding-japanese"
    dimension: 1024
  image:
    model: "openai/clip-vit-large-patch14"
    dimension: 768
    projection_dim: 1024
```

## パフォーマンス

### GPU VRAM使用量

| フェーズ | VRAM使用量 | 処理時間（目安） |
|---------|-----------|----------------|
| レイアウト解析 | ~5GB | 2-3分/PDF |
| キャプション生成 | ~6GB | 30-60秒/100図表 |
| 埋め込み生成 | ~4GB | 10秒/100ブロック |
| 検索＋回答生成 | ~7GB | 5-10秒/クエリ |

**合計**: ピーク ~8GB VRAM（16GB環境で安全に動作）

### スループット

- **レイアウト解析**: ~200ページ/分（バッチサイズ4）
- **キャプション生成**: ~50図表/分（バッチサイズ8）
- **検索**: <1秒/クエリ（Qdrant検索）
- **回答生成**: 5-10秒/クエリ（Qwen2.5-7B推論）

## トラブルシューティング

### CUDA Out of Memory

```python
# configs/mm_llm_config.yaml
models:
  primary:
    batch_size: 4  # 8 → 4 に削減

# configs/layout_config.yaml
pdf:
  max_image_dimension: 1536  # 2048 → 1536 に削減
```

### Ollama接続エラー

```powershell
# Ollamaサービス起動確認
ollama list

# モデルダウンロード
ollama pull qwen2.5-omni:7b-instruct-q4_k_m

# ベースURL確認
$env:OLLAMA_HOST = "http://localhost:11434"
```

### PyMuPDF ImportError

```powershell
pip install --upgrade PyMuPDF
# または
pip install pymupdf-fonts
```

## 既存v0.6.4との互換性

**再利用コンポーネント:**
- ✅ BM25検索（`rank_bm25`）
- ✅ テキスト埋め込み（`hotchpotch/static-embedding-japanese`）
- ✅ Triple抽出プロンプト（Ollama Qwen2.5）
- ✅ Calibration Model（LightGBM）
- ✅ 評価パイプライン（AI-as-Judge）

**拡張ポイント:**
- 🆕 マルチモーダルブロック（テキスト＋図表）
- 🆕 Qdrantベクトルストア（FAISS → Qdrant）
- 🆕 画像埋め込み（CLIP統合）
- 🆕 図表参照回答生成

## 参考文献

- **CoLRAG-TF v0.6.4**: `0_LogBAK/2b_kasensabo_colrag-tf/`
- **RAPTOR参考実装**: `0_LogBAK/2a_multimodal_raptor_colvbert_blip/`
- **Table Transformer**: [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection)
- **Qwen2.5-Omni**: [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- **LlamaIndex**: [https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)

## ライセンス

MIT License (既存のCoLRAG-TF v0.6.4と同じ)

## 更新履歴

- **2026-07-03**: v0.7.0 初版リリース（マルチモーダル対応）
- **2026-06-29**: v0.6.4 ベース版（テキストのみ）
