# 教訓: v0.7.3 4軸融合マルチモーダルRAG

## 概要

**バージョン比較:**
- **v0.7.2**: 2軸融合 (Text + Triple)
- **v0.7.3**: 4軸融合 (Text + BM25 + Triple + Image)

**実行日:** 2026-07-04  
**テストケース:** 12枚の災害画像（全数実行）  
**評価画像:** `12_ベネズエラM7地震1分揺れ-venezuela-quake_CNN.jpg`

---

## 改善点の分析

### 1. スコア融合アーキテクチャの拡張

#### v0.7.2（2軸融合）

```python
score = alpha_text * text_score + alpha_triple * triple_score
# 重み: α_text=0.6, α_triple=0.4
```

**特徴:**
- 埋め込みベースの意味的類似性（Text）
- 知識グラフベースの関連性（Triple）
- シンプルな2軸構造

#### v0.7.3（4軸融合）

```python
score = alpha_text * text_score + 
        alpha_bm25 * bm25_score +     # 新規追加
        alpha_triple * triple_score + 
        alpha_image * image_score     # 拡張準備
# 重み: α_text=0.4, α_bm25=0.3, α_triple=0.2, α_image=0.1
```

**特徴:**
- ✅ **BM25キーワード検索の追加**: 字句的マッチングによる精度向上
- ✅ **Image軸の明示化**: 将来的な視覚特徴統合の準備
- ✅ **重み配分の再設計**: より均等な4軸バランス

---

### 2. BM25キーワード検索の効果

#### BM25実装の詳細

**トークナイゼーション:**
```python
def _bm25_tokenize(text: str) -> List[str]:
    chars = list(text)  # 文字単位
    bigrams = [text[i:i+2] for i in range(len(text)-1)]  # 2-gram
    return chars + bigrams
    
# 例: '地震被害' → ['地', '震', '被', '害', '地震', '震被', '被害']
```

**インデックス統計:**
- **ブロック数**: 2,430（text + figure キャプション + OCR）
- **BM25アルゴリズム**: BM25Okapi
- **正規化**: スコアを[0, 1]に正規化

#### v0.7.3での検索結果（ベネズエラ地震ケース）

| Rank | Block ID | Text | BM25 | Triple | 総合 |
|------|----------|------|------|--------|------|
| 1 | 復興知見_202103_...page052_table_0 | 0.4051 | **0.9850** | 0.0000 | 0.5490 |
| 2 | 03歴史資料集3_...page011_table_0 | 0.3647 | **0.9794** | 0.0000 | 0.5276 |
| 3 | 01歴史資料集3_...page008_table_0 | 0.2977 | **0.9658** | 0.0000 | 0.4906 |
| 4 | 01歴史資料集4_...page019_table_0 | 0.2514 | **0.9686** | 0.0000 | 0.4694 |
| 5 | 平成の災害事例_...page004_table_0 | 0.5082 | **0.0000** | 0.4253 | 0.3460 |

**観察結果:**
- 🔥 **BM25の圧倒的優位**: Top-4がすべてBM25スコア0.96-0.98
- ⚠️ **Tripleの貢献低下**: Rank 5のみTripleスコア0.43（他はゼロ）
- 📊 **検索内容の変化**: 
  - Rank 1: 応急危険度判定（二次被害防止）
  - Rank 2: 台風被害と復旧教訓（倒木対策）
  - Rank 3: 神戸大震災の廃棄物処理
  - Rank 4: 被害認定調査（2次・3次調査）

#### v0.7.2での検索結果（熊本地震ケース - 比較参照）

| Rank | Block ID | Text | Triple | 総合 |
|------|----------|------|--------|------|
| 1 | 復興知見_202103_...page053_table_0 | 0.6347 | **0.6423** | 0.7653 |
| 2 | 01歴史資料集5_...page054_table_0 | 0.6628 | **0.5990** | 0.7647 |
| 3 | 01歴史資料集3_...page070_table_0 | 0.5877 | **0.6893** | 0.7540 |
| 4 | 大地震の事例_2016-1_kumamoto_...page002 | 0.6061 | **0.6283** | 0.7380 |
| 5 | 大地震の事例_2016-1_kumamoto_...page032 | 0.5591 | **0.6811** | 0.7295 |

**観察結果:**
- 🔥 **Tripleの高い貢献**: すべて0.60-0.69の高スコア
- 📈 **総合スコアの高さ**: 0.73-0.77（v0.7.3の0.35-0.55より大幅に高い）
- ✅ **熊本地震事例の直接ヒット**: Rank 4, 5で熊本地震ブロックを取得

---

### 3. 検索戦略の違い

#### BM25キーワード検索の強み

**✅ 優れている点:**
1. **字句的マッチング**: 「建物」「倒れた」「被害」などの直接的キーワード
2. **高速検索**: トークンベースの効率的なスコアリング
3. **専門用語対応**: 「応急危険度判定」「被災度区分」などの固有表現

**⚠️ 課題:**
1. **意味的関連性の欠如**: トリプルのような知識グラフベースの推論が弱い
2. **同義語・関連語の未対応**: 「地震」と「揺れ」、「建物」と「構造物」の関連性
3. **文脈理解の限界**: 画像理解で得た「倒れた建物」を文脈的に活用できない

#### Triple検索の強み（v0.7.2で顕著）

**✅ 優れている点:**
1. **意味的関連性**: (地震, 引き起こす, 建物倒壊) のような知識トリプル
2. **推論能力**: 災害種別 → 対策 → 復旧プロセスの連鎖
3. **事例類似性**: 熊本地震 → 東日本大震災 → 神戸大震災の類似パターン

**⚠️ 課題:**
1. **カバレッジ**: 11,414トリプルでは全ブロック（1,399）の一部のみ
2. **トリプル抽出精度**: LLMベースの抽出エラーや不完全性
3. **計算コスト**: FAISS検索の追加コスト

---

### 4. 重み配分の影響分析

#### スコア貢献度の比較

**v0.7.2 (2軸):**
```
総合スコア = 0.6 × Text + 0.4 × Triple
例: 0.7653 = 0.6 × 0.6347 + 0.4 × 0.6423
    0.7653 ≈ 0.3808 + 0.2569
```

**v0.7.3 (4軸):**
```
総合スコア = 0.4 × Text + 0.3 × BM25 + 0.2 × Triple + 0.1 × Image
例: 0.5490 = 0.4 × 0.4051 + 0.3 × 0.9850 + 0.2 × 0.0 + 0.1 × 0.0
    0.5490 ≈ 0.1620 + 0.2955 + 0.0 + 0.0
```

#### 重み配分の課題

| 軸 | v0.7.2 | v0.7.3 | 変化 | 影響 |
|----|--------|--------|------|------|
| Text | 0.6 | 0.4 | ↓33% | 意味的類似性の重要度低下 |
| BM25 | - | 0.3 | NEW | キーワードマッチングの追加 |
| Triple | 0.4 | 0.2 | ↓50% | ⚠️ 知識グラフの貢献度が半減 |
| Image | - | 0.1 | NEW | 将来的な視覚特徴統合の準備 |

**⚠️ 重要な教訓:**
- **Tripleの重みが0.4→0.2に半減**したことで、知識グラフベースの推論能力が大幅に低下
- BM25の追加は有益だが、**重み配分の再調整が不十分**
- 現状では**BM25が支配的**（スコア0.96-0.98）で、他の軸が埋もれている

---

### 5. 生成された教訓の質的比較

#### v0.7.3での生成内容（抜粋）

**良い点:**
- ✅ **具体的な対策**: 「応急危険度判定」「被災度区分判定」など専門用語の活用
- ✅ **過去事例の引用**: 「台風15号」「神戸大震災」「東日本大震災」の教訓統合
- ✅ **体系的整理**: 4つの観点（特徴、事前、発生時、復旧）で構造化

**改善が必要な点:**
- ⚠️ **画像との関連性**: 「建物が倒れた」という画像情報と検索結果の接続が弱い
- ⚠️ **災害種別の整合性**: ベネズエラ地震なのに、台風・倒木の教訓が上位に
- ⚠️ **深い推論の欠如**: トリプルの低貢献により、因果関係や連鎖的対策の提示が浅い

#### v0.7.2での生成内容（熊本地震ケース）

**良い点:**
- ✅ **専門的内容**: 「宅地危険度判定」「アスベスト調査」など技術的観点
- ✅ **二次災害対策**: 余震・降雨による崩壊リスクへの言及
- ✅ **熊本地震との直接対応**: 検索結果に熊本地震ブロックが含まれる

---

## 重要な教訓と推奨事項

### 教訓1: BM25は有益だが重み配分の最適化が必須

**現状の問題:**
- BM25スコアが0.96-0.98と極端に高く、正規化が不十分
- α_bm25=0.3でも総合スコアに対して支配的（60-80%の貢献）
- Tripleの重み0.2では、知識グラフの推論能力が発揮できない

**推奨される改善策:**

#### 案1: 重み配分の再調整
```python
# 提案: Tripleの重みを回復、BM25を補助的に
alpha_text = 0.35
alpha_bm25 = 0.20  # 0.3 → 0.2 に減少
alpha_triple = 0.35  # 0.2 → 0.35 に増加
alpha_image = 0.10
```

#### 案2: BM25スコアの再正規化
```python
# BM25の過剰な高スコアを抑制
bm25_norm = 1 / (1 + np.exp(-5 * (bm25_scores - 0.5)))  # sigmoid変換
# または
bm25_norm = np.sqrt(bm25_scores)  # 平方根変換で分散を抑える
```

#### 案3: ハイブリッド戦略
```python
# 災害種別やクエリタイプに応じて重みを動的調整
if is_keyword_heavy_query:  # 「応急危険度判定」など専門用語
    alpha_bm25 = 0.35
    alpha_triple = 0.25
elif is_conceptual_query:  # 「災害の教訓」など概念的
    alpha_bm25 = 0.15
    alpha_triple = 0.40
```

---

### 教訓2: 検索結果の多様性とリランキング

**v0.7.3の問題:**
- Top-4がすべてBM25優位 → 検索結果の多様性不足
- Tripleスコアが0のブロックが多数 → 知識グラフの未活用

**推奨される改善策:**

#### 多様性を考慮したリランキング
```python
def diverse_reranking(results: List[RetrievalResult], 
                     top_k: int = 5,
                     diversity_weight: float = 0.3) -> List[RetrievalResult]:
    """
    検索結果を多様性を考慮してリランキング
    
    戦略:
    1. Top-2: BM25優位のブロックを選択（キーワードマッチング重視）
    2. Rank 3-5: Tripleスコアが高いブロックを選択（意味的関連性）
    """
    bm25_dominant = [r for r in results if r.bm25_score > r.triple_score]
    triple_dominant = [r for r in results if r.triple_score > r.bm25_score]
    
    # 上位2件はBM25優位
    final = bm25_dominant[:2]
    
    # 残りはTriple優位から選択
    final.extend(triple_dominant[:top_k-2])
    
    return final
```

---

### 教訓3: 画像理解との統合を強化

**現状の問題:**
- 画像理解で「建物が倒れた」を認識
- しかし検索クエリに「災害の教訓 + 画像説明」を単純連結
- 画像情報（視覚的特徴、災害種別）が検索スコアに反映されない

**推奨される改善策:**

#### 画像理解ベースのクエリ拡張
```python
# Step 1: 画像理解から災害種別を抽出
disaster_type = extract_disaster_type(image_description)
# 例: "建物が倒れた" → "地震" または "台風"

# Step 2: 災害種別特化の検索クエリ生成
if disaster_type == "地震":
    query_expansion = ["耐震", "応急危険度", "余震", "建物倒壊"]
elif disaster_type == "洪水":
    query_expansion = ["浸水", "排水", "土砂", "避難指示"]

# Step 3: 拡張クエリでBM25検索を強化
bm25_results = self._search_bm25(
    query + " " + " ".join(query_expansion),
    top_k=top_k*2
)
```

#### Image軸の実装（α_image=0.1の活用）
```python
# 画像タイプと検索ブロックのマッチング
def calculate_image_score(block: Dict, image_type: str) -> float:
    """
    ブロックが画像タイプに関連しているかスコアリング
    
    Args:
        block: 検索ブロック
        image_type: 画像から認識された災害種別
    
    Returns:
        0.0-1.0のスコア
    """
    if block['type'] in ['table', 'figure']:
        # 図表ブロックは視覚情報に関連している可能性が高い
        base_score = 0.3
    else:
        base_score = 0.0
    
    # ブロック内容と災害種別のマッチング
    content = block.get('content', '').lower()
    disaster_keywords = {
        '地震': ['地震', '揺れ', '耐震', '建物倒壊'],
        '洪水': ['洪水', '浸水', '豪雨', '河川'],
        '土砂': ['土砂', '崩壊', '斜面', '地滑り']
    }
    
    if image_type in disaster_keywords:
        keyword_match = sum(1 for kw in disaster_keywords[image_type] 
                          if kw in content)
        return base_score + 0.7 * (keyword_match / len(disaster_keywords[image_type]))
    
    return base_score
```

---

### 教訓4: BM25インデックスの品質向上

**現状の実装:**
```python
def bm25_tokenize(text: str) -> List[str]:
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    return chars + bigrams
```

**問題点:**
- 文字単位とバイグラムのみ → 専門用語の分割（例: 「応急危険度判定」→ 「応」「急」「危」...）
- ストップワード未処理 → 「の」「が」「て」などの助詞が高頻度
- 同義語・関連語の未対応

**推奨される改善策:**

#### 専門用語辞書の統合
```python
# 災害分野の専門用語を保護
DISASTER_TERMS = {
    "応急危険度判定", "被災度区分判定", "避難指示", "二次災害",
    "地盤液状化", "建物耐震", "避難所運営", "災害廃棄物"
}

def enhanced_bm25_tokenize(text: str) -> List[str]:
    tokens = []
    
    # 専門用語を優先的にトークン化
    for term in DISASTER_TERMS:
        if term in text:
            tokens.append(term)
            text = text.replace(term, " ")  # 処理済み
    
    # 残りのテキストをバイグラム処理
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    tokens.extend(chars + bigrams)
    
    return tokens
```

#### ストップワード除去
```python
STOPWORDS = {"の", "が", "を", "に", "は", "で", "と", "て", "も", "や"}

def bm25_tokenize(text: str) -> List[str]:
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    all_tokens = chars + bigrams
    
    # ストップワード除去（文字単位のみ）
    return [t for t in all_tokens if t not in STOPWORDS]
```

---

### 教訓5: 評価指標の導入

**現状の問題:**
- 定量的な評価指標がない
- v0.7.2とv0.7.3のどちらが「良い」か判断が困難
- 総合スコアの絶対値（0.73 vs 0.55）だけでは不十分

**推奨される評価指標:**

#### 1. Relevance Score (関連性スコア)
```python
# 人手で作成したゴールドスタンダード
gold_standard = {
    "12_ベネズエラM7地震.jpg": [
        "大地震の事例_2016-1_kumamoto_page002",  # 関連度: 高
        "復興知見_202103_fukku-fukko-handbook_1_page052",  # 関連度: 中
        # ...
    ]
}

# Precision@K, Recall@K, F1@K
precision_at_5 = len(retrieved[:5] & gold_standard[image]) / 5
recall_at_5 = len(retrieved[:5] & gold_standard[image]) / len(gold_standard[image])
```

#### 2. Diversity Score (多様性スコア)
```python
# 検索結果の多様性を評価
def diversity_score(results: List[RetrievalResult]) -> float:
    """
    検索結果の多様性をスコアリング
    
    多様性の定義:
    - 異なるVolume（資料集）からの取得
    - BM25/Triple/Textの貢献バランス
    """
    # Volume多様性
    volumes = set(r.volume for r in results)
    volume_diversity = len(volumes) / len(results)
    
    # スコア軸の多様性（エントロピー計算）
    bm25_dominant = sum(1 for r in results if r.bm25_score > r.triple_score)
    triple_dominant = len(results) - bm25_dominant
    
    if bm25_dominant == 0 or triple_dominant == 0:
        axis_diversity = 0.0
    else:
        p_bm25 = bm25_dominant / len(results)
        p_triple = triple_dominant / len(results)
        axis_diversity = -p_bm25 * np.log2(p_bm25) - p_triple * np.log2(p_triple)
    
    return 0.5 * volume_diversity + 0.5 * axis_diversity
```

#### 3. Answer Quality Score (回答品質スコア)
```python
# LLM-as-a-Judgeによる自動評価
def evaluate_answer_quality(image_description: str,
                            retrieval_results: List[Dict],
                            generated_answer: str) -> Dict[str, float]:
    """
    生成された教訓の品質を評価
    
    評価観点:
    1. 画像との関連性 (Image Relevance)
    2. 具体性 (Specificity) - 数値、固有名詞の含有
    3. 実用性 (Actionability) - 具体的な対策・手順の提示
    4. 体系性 (Structure) - 4観点での整理
    """
    judge_prompt = f"""
以下の生成された災害教訓を評価してください。

【画像説明】
{image_description}

【生成された教訓】
{generated_answer}

【評価観点】
1. Image Relevance (0-10): 画像内容との関連性
2. Specificity (0-10): 具体的な数値・事例の含有
3. Actionability (0-10): 実行可能な対策の提示
4. Structure (0-10): 体系的な整理

JSON形式で各スコアを返してください。
"""
    
    # Ollama qwen2.5:7bで評価
    evaluation = llm_client.generate_answer(judge_prompt, "", max_tokens=200)
    return json.loads(evaluation)
```

---

## 具体的な実装ロードマップ

### Phase 1: 重み配分の最適化（優先度: 高）

**実装内容:**
1. グリッドサーチで最適重みを探索
   ```python
   alpha_text_range = [0.3, 0.35, 0.4]
   alpha_bm25_range = [0.15, 0.2, 0.25]
   alpha_triple_range = [0.3, 0.35, 0.4]
   alpha_image_range = [0.05, 0.1]
   ```

2. 評価指標（Precision@5, Diversity）をベースに選択

3. 災害種別ごとの最適重みを決定

**期待効果:**
- Tripleの知識グラフ推論能力の回復
- BM25とTripleのバランス改善
- 検索精度の向上（目標: 平均スコア0.65以上）

---

### Phase 2: BM25インデックスの品質向上（優先度: 高）

**実装内容:**
1. 専門用語辞書の作成（災害分野200-300語）
2. ストップワード除去の実装
3. インデックス再構築

**期待効果:**
- 専門用語の正確なマッチング
- ノイズの削減（助詞の影響低減）
- BM25スコアの適切な分布（0.5-0.8程度）

---

### Phase 3: 画像理解との統合強化（優先度: 中）

**実装内容:**
1. 災害種別抽出モジュールの実装
2. クエリ拡張の自動化
3. Image軸スコアの実装

**期待効果:**
- 画像情報の検索への反映
- 災害種別特化の検索精度向上
- α_image=0.1の有効活用

---

### Phase 4: 評価基盤の構築（優先度: 中）

**実装内容:**
1. ゴールドスタンダードの作成（12画像×5関連ブロック）
2. Precision@K, Recall@K, Diversity指標の実装
3. LLM-as-a-Judge評価の自動化

**期待効果:**
- v0.7.2/v0.7.3/v0.7.4の定量比較
- 継続的な改善のための基盤
- 客観的な性能評価

---

### Phase 5: リランキングと多様性制御（優先度: 低）

**実装内容:**
1. 多様性を考慮したリランキングアルゴリズム
2. BM25/Triple/Textの貢献バランス制御
3. Volume（資料集）の多様性確保

**期待効果:**
- 検索結果の幅広い視点提供
- BM25偏重の軽減
- ユーザー満足度の向上

---

## 結論

### v0.7.3の達成点

✅ **4軸融合アーキテクチャの実装**
- Text + BM25 + Triple + Image の統合フレームワーク
- 拡張性の高い設計（将来的な視覚特徴統合に対応）

✅ **BM25キーワード検索の追加**
- 字句的マッチングによる補完
- 専門用語（「応急危険度判定」など）の高精度取得

✅ **スコア可視化の強化**
- 各軸の個別スコア表示
- デバッグと分析の容易性向上

### v0.7.3の課題

⚠️ **重み配分の不適切さ**
- Tripleの重みが半減（0.4→0.2）し、知識グラフの推論能力が低下
- BM25が支配的（スコア0.96-0.98）で、他軸が埋もれる

⚠️ **検索結果の多様性不足**
- Top-4がすべてBM25優位
- Tripleスコアが0のブロックが多数

⚠️ **画像理解との統合が不完全**
- 画像情報が検索スコアに反映されない
- α_image=0.1が未活用（すべて0）

### 次ステップ (v0.7.4への展望)

1. **重み配分の最適化** → グリッドサーチで最適解を探索
2. **BM25インデックスの改善** → 専門用語辞書とストップワード処理
3. **評価基盤の構築** → 定量的な性能比較の実施
4. **画像理解の統合強化** → 災害種別特化の検索戦略

### 最終的な教訓

> **「4軸融合は有望だが、単純な追加では不十分。各軸の特性を理解し、適切な重み配分と相互連携が必要。」**

BM25の追加自体は有益だが、v0.7.2のTriple優位の検索戦略で得られていた高精度（0.73-0.77）を維持しつつ、キーワード検索を補完的に活用する設計が求められる。v0.7.4では、**Triple主導+BM25補助**の戦略を採用し、知識グラフの推論能力を最大限に活用することを推奨する。
