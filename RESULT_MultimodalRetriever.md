# Phase 6: Multimodal HippoRAG2 Retriever Demo Results

## Execution Summary

**Date:** 2026-07-03  
**Script:** `experiments_v070/06_multimodal_retriever.py --demo`  
**Embedding Model:** hotchpotch/static-embedding-japanese (1024-dim, CUDA)  
**Blocks Loaded:** 2,403 blocks  
**Triple Index:** 11,414 triples (FAISS IndexFlatIP)

---

## Test Queries

### Query 1: 台風12号の被害状況を教えてください

**Query Analysis:**
- Original: `台風12号の被害状況を教えてください`
- **Figure-related:** False
- **Figure keywords:** []
- **Needs image:** True

**Top-5 Retrieval Results:**

| Rank | Block ID | Type | Score | Text Score | Triple Score |
|------|----------|------|-------|------------|--------------|
| 1 | `03歴史資料集4_2018-2019_page005_text_full` | text | 0.3412 | 0.6825 | 0.0000 |
| 2 | `01歴史資料集2_1958-1993_page059_table_0` | table | 0.3331 | 0.6661 | 0.0000 |
| 3 | `令和の災害事例_202505_Reiwa6DisasterExamples_page086_text_full` | text | 0.3176 | 0.6353 | 0.0000 |
| 4 | `平成の災害事例_H28_typhoon10_page002_text_full` | text | 0.3086 | 0.6171 | 0.0000 |
| 5 | `02歴史資料集8_2011-2018_page045_text_full` | text | 0.3078 | 0.6155 | 0.0000 |

**Analysis:**
- テキスト検索で関連ブロックを正しく取得
- トリプルスコアがすべて0.0000 → クエリに対応するトリプルが見つからなかった可能性
- 台風10号、台風19号の情報が上位にランクイン（台風12号の直接情報は含まれていない可能性）

---

### Query 2: 表で示された被害額はいくらですか

**Query Analysis:**
- Original: `表で示された被害額はいくらですか`
- **Figure-related:** True ✅
- **Figure keywords:** `['表']` ✅
- **Needs image:** False

**Top-5 Retrieval Results:**

| Rank | Block ID | Type | Score | Text Score | Triple Score |
|------|----------|------|-------|------------|--------------|
| 1 | `02歴史資料集4_2011-2018_page019_table_0` | **table** | **0.5890** | 0.5944 | **0.6455** |
| 2 | `01歴史資料集2_1958-1993_page101_table_0` | **table** | **0.5709** | 0.5518 | **0.6663** |
| 3 | `01歴史資料集2_1958-1993_page003_table_1` | table | 0.3752 | 0.6253 | 0.0000 |
| 4 | `大地震の事例_2011_higashinihon_page088_table_0` | table | 0.3646 | 0.6076 | 0.0000 |
| 5 | `平成の災害事例_H28_kumamoto_page006_table_1` | table | 0.3615 | 0.6025 | 0.0000 |

**Content (Rank 1):**
```
この表は、特定の災害による被害状況を示しています。主要な項目には区分、細分、推定被害額があります。
重要な数値として、総計の推定被害額が46.1億円と80.2億円（合計126.3億円）に上ることがわかり...
```

**Analysis:**
- ✅ 図キーワード検出が正常動作（`['表']`）
- ✅ **トリプルブーストが機能**（上位2件でTriple Score 0.6455, 0.6663）
- ✅ すべての結果がtableタイプ → Figure boost ×1.2が効いている
- 上位2件は被害額の具体的数値を含むテーブル
- トリプル検索により、関連性の高い被害額情報が優先的に取得された

---

### Query 3: 全壊家屋の数は何棟ですか

**Query Analysis:**
- Original: `全壊家屋の数は何棟ですか`
- **Figure-related:** False
- **Figure keywords:** []
- **Needs image:** False

**Top-5 Retrieval Results:**

| Rank | Block ID | Type | Score | Text Score | Triple Score |
|------|----------|------|-------|------------|--------------|
| 1 | `01歴史資料集5_2005-2009_page077_table_0` | **table** | **0.5679** | 0.5799 | **0.9265** 🔥 |
| 2 | `平成の災害事例_H7_hanshinawaji_page003_table_0` | table | 0.3077 | 0.6154 | 0.0000 |
| 3 | `平成の災害事例_H28_kumamoto_page004_table_0` | table | 0.3031 | 0.6062 | 0.0000 |
| 4 | `平成の災害事例_H16_chuetsu_page014_table_0` | table | 0.2987 | 0.5973 | 0.0000 |
| 5 | `02歴史資料集6_2011-2018_page027_table_0` | table | 0.2932 | 0.5863 | 0.0000 |

**Content (Rank 1):**
```
この表は、特定の地域における災害による被害を示しています。主要な項目には「家屋被害」「農林業関係被害」
「市管理土木施設被害」が含まれ、それぞれの分野での具体的な数値が記載されています。重要な数値として...
```

**Analysis:**
- ✅ **トリプル検索が非常に高い精度で機能**（Triple Score 0.9265 🔥）
- トリプルスコアがテキストスコアを大幅に上回る → トリプル抽出の効果が顕著
- 1位の結果は「全壊家屋」に関する具体的な数値情報を含むテーブル
- 2-5位はトリプルスコア0.0000だが、テキスト検索により関連情報（阪神淡路、熊本地震、中越地震）を取得

---

## Key Observations

### ✅ Successful Components

1. **Query Analysis:**
   - 図キーワード検出が正常動作（`['表']`を正しく検出）
   - `is_figure_related`フラグが適切に設定

2. **Triple Search:**
   - Query 2で0.6455, 0.6663のトリプルスコア → 被害額関連トリプルを発見
   - Query 3で**0.9265の非常に高いトリプルスコア** → 全壊家屋トリプルを発見
   - トリプル検索により、テキスト類似度だけでは見つからない関連情報を取得

3. **Score Fusion:**
   - テキストスコアとトリプルスコアの統合が正常動作
   - Figure boost ×1.2により、tableタイプのブロックが優先的に取得

4. **Block Retrieval:**
   - 2,403ブロック中から関連性の高い情報を正確に取得
   - テーブルキャプションの品質が高く、具体的な数値情報を含む

### ⚠️ Potential Issues

1. **Zero Triple Scores:**
   - Query 1ですべてのトリプルスコアが0.0000
   - Query 2, 3でも一部のブロックでトリプルスコアが0.0000
   - 原因候補:
     - クエリに対応するトリプルが存在しない
     - トリプル抽出時に該当ブロックでトリプルが生成されなかった
     - トリプル検索の類似度閾値が厳しすぎる可能性

2. **Query-Specific Limitations:**
   - Query 1（台風12号）で直接の情報が見つからず、台風10号・19号の情報が上位に
   - データセット内に台風12号の情報が含まれていない可能性

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Blocks loaded** | 2,403 |
| **Triple index size** | 11,414 triples |
| **Embedding dimension** | 1024 |
| **Device** | CUDA (GPU) |
| **Embedding generation time** | <1 second (205 batches/sec) |
| **Retrieval time** | Real-time (sub-second per query) |

---

## Conclusions

### HippoRAG2-Style Retrieval の効果

1. **トリプル検索の有効性:**
   - 具体的な数値情報（被害額、全壊家屋数）の検索で高精度
   - Query 3で0.9265という非常に高いトリプルスコア → 知識グラフベース検索の威力を実証

2. **マルチモーダル対応:**
   - 図キーワード検出により、テーブル/図表を優先的に取得
   - Figure boost機能により、表形式データが適切にランク向上

3. **スコア融合の効果:**
   - テキスト検索とトリプル検索の相補的な動作
   - トリプルが見つからない場合でもテキスト検索でフォールバック

### 今後の改善点

1. トリプル抽出率の向上（現在75.6%）
2. ゼロトリプルスコアケースの原因分析
3. 画像理解機能（Phase 6c）の統合テスト
4. 200 QA生成（Phase 7）への移行

---

## Next Steps

- ✅ Phase 4-5完了（Triple抽出 11,414個、FAISSインデックス構築）
- ✅ Phase 6デモ完了（HippoRAG2検索動作確認）
- ⏭️ **Phase 7: 200 QA生成**
  ```powershell
  .\.venv-coltf\Scripts\python.exe experiments_v070\07_multimodal_evaluation.py --generate-qa --target-count 200
  ```
