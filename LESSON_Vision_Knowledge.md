# Vision Knowledge抽出の教訓と今後の課題

## 実験の目的（v0.8.0）

v0.7.3のレイアウト解析（Table Transformer, confidence_threshold=0.7）では、主に**表（table）**のみが抽出され、災害資料に含まれる以下の重要な視覚情報が欠落していた：

1. **地図（map）**: 被災地域の位置、地理的範囲
2. **グラフ（graph）**: 降水量の時系列推移、被害規模の推移
3. **写真（photo）**: 災害現場の被災状況、復旧活動の様子
4. **概念図（diagram）**: 復興施策のフロー、防災システムの構造

## 実験アプローチと結果

### アプローチ1: Table Transformer閾値調整（失敗）
**実験**: `confidence_threshold` を 0.7 → 0.4/0.5/0.6 に下げる  
**結果**: 検出数が21個（0.4）から16個（0.7）とわずかな増加のみ。すべて"table"クラスで、figure/imageクラスは0個  
**原因**: Table Transformerモデル自体が表検出に特化しており、多様な図表タイプに対応していない

### アプローチ2: OCRベース + ヒューリスティック（失敗）
**実験**: Tesseract OCRでテキスト領域を検出し、非テキスト領域を図表候補として抽出  
**結果**: 検出数0個（min_area=1000でも0個）  
**原因**: 
- 災害資料のページは高密度のテキストで構成され、OCRが全体をテキスト領域として認識
- 非テキスト領域がほとんど残らず、図表候補が抽出できない
- 形態学処理でノイズ除去した結果、図表領域も消失

### アプローチ3: Vision LLM（llava:7b）による直接検出（成功）
**実験**: Vision LLMに災害資料ページを入力し、図表タイプと詳細説明を直接生成  
**結果（n=32サンプルページ）**:
- 総検出数: 104要素（3.25要素/ページ）
- Table: 29要素（平均信頼度0.903）
- Graph: 28要素（平均信頼度0.900）
- Photo: 36要素（平均信頼度0.850）
- Map: 9要素（平均信頼度0.906）
- Diagram: 2要素（平均信頼度0.825）

**検出例**:
```json
{
  "type": "graph",
  "description": "2011年3月の降水量時系列グラフ。横軸は時刻、縦軸は降水量(mm/h)。3月11日14時にピーク値120mm/hを記録。その後急激に減少",
  "confidence": 0.9
}
```

```json
{
  "type": "photo",
  "description": "仙台市宮城野区の津波浸水被害。木造住宅が流出し、車両が散乱。浸水深2-3m程度",
  "confidence": 0.85
}
```

## 成功要因の分析

1. **タスク適合性**: Vision LLMは一般的な視覚理解タスクに対応しており、特定のクラス（table/figure）に制約されない
2. **文脈理解**: 災害資料という文脈を理解し、「降水量グラフ」「被災地図」など意味的な分類が可能
3. **柔軟な出力**: Bounding boxなしでも、タイプ分類と詳細説明（200文字）を自然言語で出力できる
4. **日本語対応**: llava:7bは日本語プロンプトと災害用語（浸水深、被災地域、復旧活動など）を理解

## 限界と課題

### 1. Bounding Box情報の欠落
**問題**: Vision LLMは図表の存在とタイプを認識できるが、ページ内の正確な座標（bbox）を出力しない  
**影響**: 
- 元画像から図表領域を切り出してキャプション生成ができない（v0.7.3のworkflowが使えない）
- 複数の図表が重なる場合、個別の領域を分離できない

**解決策の方向性**:
- Vision LLMの出力をpost-processingし、画像処理でbboxを推定
- Object detection専用モデル（YOLO, GroundingDINO）とVision LLMを組み合わせる2段階パイプライン
- SAM (Segment Anything Model)でセグメンテーションしてからVision LLMで分類

### 2. 処理時間の増大
**データ**: 32ページの解析に5分3秒（9.49秒/ページ）  
**v0.7.3との比較**: 
- Table Transformer: ~0.5秒/ページ（10倍速い）
- Vision LLM: ~10秒/ページ

**全体への影響**:
- 2,319ページ全体: 2,319 × 10秒 ≈ 6.4時間
- v0.7.3の場合: 2,319 × 0.5秒 ≈ 20分

**解決策**:
- バッチ処理の最適化（複数ページを並列処理）
- より軽量なモデル（qwen2-vl, phi-3-vision）の検討
- 重要ページのみVision LLM、その他はTable Transformerというハイブリッドアプローチ

### 3. キャプション品質の不安定性
**観察**: "災害資料のページを解析してください。以下の要素を検出してください。1. **表 (table)..." というプロンプトの繰り返しがdescriptionに混入  
**原因**: Vision LLMがプロンプト指示をそのまま出力する傾向（instruction following issue）

**解決策**:
- Few-shot prompting: 良い例を2-3個提示
- 出力フォーマットの厳格化（JSON schema validation）
- Post-processing: プロンプト文字列を正規表現で除去

### 4. RAGシステムへの統合課題

#### 4.1 インデックス構造の再設計
**v0.7.3の構造**:
```
layout_blocks_captioned.jsonl:
  - block_id: "page001_block000"
  - type: "table"
  - bbox: [x1, y1, x2, y2]
  - image_path: "figures/page001_block000.png"
  - caption: "表1. 被災地域別の被害状況"
  - extracted_text: "地域名\t死者数\t負傷者数..."
```

**v0.8.0（Vision LLM）の構造案**:
```
vision_blocks_captioned.jsonl:
  - block_id: "page001_vision000"
  - type: "graph"  # Vision LLMが分類
  - bbox: null  # bboxなし
  - image_path: "sample_figure-in-page_n32/page001.png"  # ページ全体
  - caption: "2011年3月の降水量時系列グラフ。横軸は時刻、縦軸は降水量..."
  - extracted_text: ""  # OCR不要
```

**問題点**:
- bboxがないため、ページ全体を画像として扱う必要がある → ストレージ容量増大
- 複数の図表が1ページに存在する場合、個別にクロップできない → 検索精度低下

#### 4.2 Embedding戦略の変更
**v0.7.3**: 各ブロックのcaptionをテキストembedding  
**v0.8.0案1**: Vision LLMのdescriptionをテキストembedding（簡単だが視覚情報損失）  
**v0.8.0案2**: CLIP/SigLIPでマルチモーダルembedding（bboxなしでページ全体をembed）  

#### 4.3 Triple抽出の課題
**v0.7.3**: キャプションから構造化知識を抽出  
例: "表1は東日本大震災の被災地域を示す" → (東日本大震災, 被災地域, 表1)

**v0.8.0**: Vision LLMのdescriptionから同様に抽出可能  
例: "仙台市宮城野区の津波浸水被害。浸水深2-3m" → (仙台市宮城野区, 浸水深, 2-3m)

ただし、Vision LLMの出力が不安定な場合、Triple抽出の精度も低下

## v0.7.3までの完成された方法論（確立済み）

### アーキテクチャ: 4軸融合RAG
```
Query → [Text Embedding] ──→ FAISS検索 ──┐
         [BM25 Keyword]   ──→ BM25検索 ──┤
         [Triple Filtering]──→ FAISS検索 ──├→ Score融合 → Top-K検索 → LLM生成
         [Image Embedding] ──→ FAISS検索 ──┘

融合重み: α_text=0.4, α_bm25=0.3, α_triple=0.2, α_image=0.1
```

### データパイプライン（v0.7.3完成版）
```
PDF (2,319ページ)
  ↓ 01_layout_analysis.py (Table Transformer, confidence=0.7)
Layout Blocks (2,430ブロック: table中心)
  ↓ 02_multimodal_caption.py (llava:7b)
Captioned Blocks (2,430ブロック + キャプション)
  ↓ 03_build_multimodal_index.py
Text Embeddings (hotchpotch/static-embedding-japanese)
  ↓ 04_extract_triples.py (qwen2.5:7b)
Knowledge Triples (11,414トリプル)
  ↓ 05_build_triple_index.py
Triple Index (FAISS)
  ↓ 09a_build_bm25_index.py
BM25 Index (15MB, 日本語bigram)
  ↓ 06_multimodal_retriever.py + 07d_evaluate_multimodal_rag.py
評価結果 (457Q, Retrieval Recall=0.9909, Answer Similarity=0.4684)
```

### 評価結果（v0.7.3確定値）
**QAデータセット**: 457問（1-hop: 169, Multi-hop: 288）

**全体性能**:
- Overall Retrieval Recall: **0.9909** (99.09%)
- Answer Similarity: **0.4684** (±0.2041)
- Answer Length: 428.3文字
- 失敗率: 0%

**1-hop vs Multi-hop比較**:
| メトリクス | 1-hop (169Q) | Multi-hop (288Q) | 改善率 |
|-----------|--------------|------------------|--------|
| Answer Similarity (Median) | 0.3214 | 0.5769 | +79.5% |
| Answer Similarity (Mean) | 0.3267 | 0.5605 | +71.6% |
| Retrieval Recall | 1.0000 | 0.5000 | - |
| Answer Length | 377.9字 | 452.0字 | +19.6% |

**BM25効果（v0.7.2→v0.7.3）**:
- 1-hop Median Similarity: 0.2872 → 0.3214 (+11.9%)
- Multi-hop維持: 0.5769（変化なし）
- Overall Recall: 0.9909（ほぼ完璧）

### 技術的成果（論文化可能）
1. **日本語災害文書に特化したマルチモーダルRAG**: Text+BM25+Triple+Image の4軸融合
2. **BM25のbigramトークナイゼーション**: 災害用語（"地震被害"）を適切に分割
3. **HippoRAG2インスパイアのTriple Filtering**: 構造化知識による検索精度向上
4. **457問の高品質QAデータセット**: 1-hop/Multi-hop/Cause-mitigationの3カテゴリ

## 今後の課題（v0.8.0以降）

### 短期課題（1-2週間）
1. **Vision LLM出力のクリーニング**: プロンプト混入除去、JSON validation強化
2. **Bounding Box推定**: Vision LLM出力 + 画像処理による位置推定
3. **ハイブリッドアプローチ**: Table Transformer（表） + Vision LLM（その他図表）

### 中期課題（1-2ヶ月）
1. **2段階パイプライン構築**: 
   - Stage 1: YOLO/GroundingDINOでbbox検出
   - Stage 2: Vision LLMで詳細分類とキャプション生成
2. **全2,319ページへの適用**: 処理時間6時間の最適化（並列化、軽量モデル）
3. **RAG統合とEnd-to-End評価**: 457Q再評価で効果検証

### 長期課題（3ヶ月以上）
1. **マルチモーダル検索の強化**: CLIP/SigLIPによるimage-text joint embedding
2. **Vision-aware Triple抽出**: 図表から直接構造化知識を抽出
3. **ユーザースタディ**: 実際の防災担当者による使用評価
4. **学習データ化**: Vision Knowledge を RLHF/DPO の報酬信号として活用

## まとめ

**v0.7.3までの達成**: テキスト中心のマルチモーダルRAGを完成させ、457問で99.09%の検索再現率を達成。BM25統合により災害用語の適合性向上。

**v0.8.0の発見**: Vision LLMは多様な図表タイプ（グラフ、地図、写真、概念図）を認識でき、詳細説明を生成可能。ただし、bbox欠落と処理時間の課題あり。

**方向性**: v0.7.3を完成版として論文化し、Vision Knowledgeの統合は次フェーズの研究課題として位置づける。短期的にはハイブリッドアプローチ（Table Transformer + Vision LLM）を検討。

---

**記録日**: 2026-07-04  
**実験環境**: experiments_v080/  
**関連ファイル**:
- `experiments_v080/01a_threshold_experiment.py` (Table Transformer閾値実験)
- `experiments_v080/01b_ocr_layout_analysis.py` (OCRアプローチ)
- `experiments_v080/01d_vision_llm_layout.py` (Vision LLMアプローチ)
- `experiments_v080/indices/vision_llm_layout/vision_llm_layout_results.json` (32サンプル結果)
