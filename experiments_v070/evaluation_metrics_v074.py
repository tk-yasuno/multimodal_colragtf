#!/usr/bin/env python3
# Copyright 2026 Multimodal CoLRAG-TF contributors
# Licensed under the Apache License, Version 2.0

"""
CoLRAG-TF v0.7.4 - Phase 10a: 評価指標実装

Precision@K, Diversity, AnswerQualityの3つの評価指標を実装。
ベイズ最適化の目的関数として使用。

Usage:
    from experiments_v070.10a_evaluation_metrics import calc_precision_at_k, calc_diversity, calc_answer_quality
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np
from collections import Counter


def calc_precision_at_k(retrieved_block_ids: List[str], 
                        gold_block_ids: List[str], 
                        k: int = 5) -> float:
    """
    Precision@K を計算
    
    Args:
        retrieved_block_ids: 検索で取得したブロックIDのリスト（順位順）
        gold_block_ids: ゴールドスタンダードの正解ブロックIDリスト
        k: 評価する上位K件
    
    Returns:
        Precision@K スコア (0.0-1.0)
    
    Example:
        >>> retrieved = ["block_1", "block_2", "block_3", "block_4", "block_5"]
        >>> gold = ["block_1", "block_3", "block_6"]
        >>> calc_precision_at_k(retrieved, gold, k=5)
        0.4  # 5件中2件が正解
    """
    if not retrieved_block_ids or not gold_block_ids:
        return 0.0
    
    # 上位K件を取得
    top_k = retrieved_block_ids[:k]
    
    # ゴールドスタンダードとの一致数
    gold_set = set(gold_block_ids)
    hits = sum(1 for block_id in top_k if block_id in gold_set)
    
    return hits / k


def calc_recall_at_k(retrieved_block_ids: List[str], 
                     gold_block_ids: List[str], 
                     k: int = 5) -> float:
    """
    Recall@K を計算
    
    Args:
        retrieved_block_ids: 検索で取得したブロックIDのリスト（順位順）
        gold_block_ids: ゴールドスタンダードの正解ブロックIDリスト
        k: 評価する上位K件
    
    Returns:
        Recall@K スコア (0.0-1.0)
    """
    if not retrieved_block_ids or not gold_block_ids:
        return 0.0
    
    # 上位K件を取得
    top_k = retrieved_block_ids[:k]
    
    # ゴールドスタンダードとの一致数
    gold_set = set(gold_block_ids)
    hits = sum(1 for block_id in top_k if block_id in gold_set)
    
    return hits / len(gold_set)


def calc_diversity(retrieval_results: List[Dict[str, Any]]) -> float:
    """
    検索結果の多様性を評価
    
    多様性の定義:
    1. BM25/Triple/Text スコアの貢献バランス（エントロピー）
    2. Volume（資料集）の多様性
    
    Args:
        retrieval_results: 検索結果のリスト
            各要素は以下のキーを持つ辞書:
            - block_id: str
            - text_score: float
            - bm25_score: float
            - triple_score: float
            - (オプション) volume: str
    
    Returns:
        Diversity スコア (0.0-1.0)
        1.0に近いほど多様性が高い
    
    Example:
        >>> results = [
        ...     {"block_id": "block_1", "text_score": 0.8, "bm25_score": 0.2, "triple_score": 0.0},
        ...     {"block_id": "block_2", "text_score": 0.1, "bm25_score": 0.9, "triple_score": 0.0},
        ...     {"block_id": "block_3", "text_score": 0.3, "bm25_score": 0.3, "triple_score": 0.4}
        ... ]
        >>> calc_diversity(results)
        0.75  # 3軸のバランスが良い
    """
    if not retrieval_results:
        return 0.0
    
    # 1. スコア軸の多様性（どの軸が支配的か）
    bm25_dominant = 0
    triple_dominant = 0
    text_dominant = 0
    balanced = 0
    
    for result in retrieval_results:
        text_score = result.get("text_score", 0.0)
        bm25_score = result.get("bm25_score", 0.0)
        triple_score = result.get("triple_score", 0.0)
        
        max_score = max(text_score, bm25_score, triple_score)
        
        # 最大スコアを持つ軸をカウント
        if max_score == 0:
            continue
        
        # 3軸が近い場合（差が20%以内）はバランス型
        scores = [text_score, bm25_score, triple_score]
        score_range = max(scores) - min(scores)
        
        if score_range < 0.2 * max_score:
            balanced += 1
        elif bm25_score == max_score:
            bm25_dominant += 1
        elif triple_score == max_score:
            triple_dominant += 1
        elif text_score == max_score:
            text_dominant += 1
    
    total = len(retrieval_results)
    
    # エントロピー計算（最大log2(4)=2.0）
    counts = [bm25_dominant, triple_dominant, text_dominant, balanced]
    probs = [c / total for c in counts if c > 0]
    
    if len(probs) <= 1:
        # すべて同じ軸が支配的 → 多様性なし
        axis_diversity = 0.0
    else:
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(probs))  # 完全に均等な場合
        axis_diversity = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # 2. Volume（資料集）の多様性
    block_ids = [r.get("block_id", "") for r in retrieval_results]
    
    # block_idから資料集名を抽出（例: "復興知見_202103_..." → "復興知見"）
    volumes = []
    for block_id in block_ids:
        parts = block_id.split("_")
        if len(parts) > 0:
            volumes.append(parts[0])
    
    unique_volumes = len(set(volumes))
    volume_diversity = unique_volumes / len(volumes) if volumes else 0.0
    
    # 総合多様性（軸とVolumeを同等に重視）
    diversity = 0.6 * axis_diversity + 0.4 * volume_diversity
    
    return float(diversity)


def calc_answer_quality(image_description: str,
                        generated_answer: str,
                        ollama_model: str = "qwen2.5:7b-instruct-q4_k_m",
                        base_url: str = "http://localhost:11434") -> float:
    """
    LLM-as-a-Judge による回答品質の評価
    
    評価観点:
    1. Image Relevance (0-10): 画像内容との関連性
    2. Specificity (0-10): 具体的な数値・事例の含有
    3. Actionability (0-10): 実行可能な対策の提示
    4. Structure (0-10): 体系的な整理
    
    Args:
        image_description: 画像理解LLMによる画像の説明
        generated_answer: 生成された災害教訓
        ollama_model: 評価に使用するOllamaモデル
        base_url: Ollama APIのベースURL
    
    Returns:
        Answer Quality スコア (0.0-1.0)
        合計40点満点を0-1に正規化
    
    Example:
        >>> image_desc = "建物が倒れた地震災害の様子"
        >>> answer = "1. 耐震性の強化が必要\\n2. 避難訓練の実施\\n..."
        >>> calc_answer_quality(image_desc, answer)
        0.75  # 30/40点 = 0.75
    """
    try:
        judge_prompt = f"""以下の生成された災害教訓を評価してください。

【画像説明】
{image_description[:300]}

【生成された教訓】
{generated_answer[:800]}

【評価観点】
各項目を0-10点で評価してください：

1. Image Relevance: 画像内容との関連性
   - 10点: 画像の災害種別・状況と完全に一致
   - 5点: 部分的に関連
   - 0点: 無関係

2. Specificity: 具体的な数値・事例の含有
   - 10点: 具体的な数値、固有名詞が複数含まれる
   - 5点: 一般的な記述
   - 0点: 抽象的すぎる

3. Actionability: 実行可能な対策の提示
   - 10点: 具体的な手順・対策が明確
   - 5点: 対策の方向性は示されている
   - 0点: 対策が不明確

4. Structure: 体系的な整理
   - 10点: 観点ごとに整理され読みやすい
   - 5点: ある程度構造化されている
   - 0点: 羅列的で構造がない

【回答形式】
以下のJSON形式で返してください：
{{"image_relevance": 8, "specificity": 7, "actionability": 9, "structure": 8}}

数値のみを返し、説明は不要です。"""

        payload = {
            "model": ollama_model,
            "prompt": judge_prompt,
            "stream": False,
            "options": {
                "num_predict": 100,
                "temperature": 0.1
            }
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer_text = result.get("response", "").strip()
            
            # JSONを抽出（前後のテキストを除去）
            import re
            json_match = re.search(r'\{[^}]+\}', answer_text)
            if json_match:
                scores = json.loads(json_match.group())
                
                # 各スコアを取得（デフォルト5点）
                image_relevance = scores.get("image_relevance", 5)
                specificity = scores.get("specificity", 5)
                actionability = scores.get("actionability", 5)
                structure = scores.get("structure", 5)
                
                # 合計40点満点を0-1に正規化
                total_score = image_relevance + specificity + actionability + structure
                normalized_score = total_score / 40.0
                
                return float(np.clip(normalized_score, 0.0, 1.0))
            else:
                # JSON解析失敗時はデフォルトスコア
                return 0.5
        else:
            # API呼び出し失敗時はデフォルトスコア
            return 0.5
            
    except Exception as e:
        print(f"⚠️ LLM-as-Judge evaluation failed: {e}")
        return 0.5  # エラー時はデフォルトスコア


def calc_combined_score(precision: float,
                        diversity: float,
                        answer_quality: float,
                        w_precision: float = 0.5,
                        w_diversity: float = 0.2,
                        w_quality: float = 0.3) -> float:
    """
    総合評価スコアを計算
    
    Args:
        precision: Precision@K スコア
        diversity: Diversity スコア
        answer_quality: Answer Quality スコア
        w_precision: Precisionの重み
        w_diversity: Diversityの重み
        w_quality: Answer Qualityの重み
    
    Returns:
        総合スコア (0.0-1.0)
    """
    return (
        w_precision * precision +
        w_diversity * diversity +
        w_quality * answer_quality
    )


# ============================================================================
# テスト関数
# ============================================================================

def test_precision_at_k():
    """Precision@K のテスト"""
    print("=" * 70)
    print("Test: Precision@K")
    print("=" * 70)
    
    # テストケース1: 完全一致
    retrieved = ["block_1", "block_2", "block_3", "block_4", "block_5"]
    gold = ["block_1", "block_2", "block_3", "block_4", "block_5"]
    p = calc_precision_at_k(retrieved, gold, k=5)
    print(f"Test 1 (完全一致): {p:.3f} (期待値: 1.000)")
    assert abs(p - 1.0) < 0.01
    
    # テストケース2: 部分一致
    retrieved = ["block_1", "block_2", "block_3", "block_4", "block_5"]
    gold = ["block_1", "block_3", "block_6"]
    p = calc_precision_at_k(retrieved, gold, k=5)
    print(f"Test 2 (2/5一致): {p:.3f} (期待値: 0.400)")
    assert abs(p - 0.4) < 0.01
    
    # テストケース3: 不一致
    retrieved = ["block_1", "block_2", "block_3", "block_4", "block_5"]
    gold = ["block_6", "block_7", "block_8"]
    p = calc_precision_at_k(retrieved, gold, k=5)
    print(f"Test 3 (不一致): {p:.3f} (期待値: 0.000)")
    assert abs(p - 0.0) < 0.01
    
    print("✅ All Precision@K tests passed\n")


def test_diversity():
    """Diversity のテスト"""
    print("=" * 70)
    print("Test: Diversity")
    print("=" * 70)
    
    # テストケース1: BM25支配的（多様性低）
    results = [
        {"block_id": "vol1_page1_table_0", "text_score": 0.2, "bm25_score": 0.95, "triple_score": 0.0},
        {"block_id": "vol1_page2_table_0", "text_score": 0.3, "bm25_score": 0.98, "triple_score": 0.0},
        {"block_id": "vol1_page3_table_0", "text_score": 0.1, "bm25_score": 0.97, "triple_score": 0.0},
        {"block_id": "vol1_page4_table_0", "text_score": 0.2, "bm25_score": 0.96, "triple_score": 0.0},
        {"block_id": "vol1_page5_table_0", "text_score": 0.3, "bm25_score": 0.94, "triple_score": 0.0}
    ]
    d = calc_diversity(results)
    print(f"Test 1 (BM25支配): {d:.3f} (期待値: < 0.5)")
    assert d < 0.5
    
    # テストケース2: バランス良い（多様性高）
    results = [
        {"block_id": "vol1_page1_table_0", "text_score": 0.8, "bm25_score": 0.2, "triple_score": 0.1},
        {"block_id": "vol2_page2_table_0", "text_score": 0.1, "bm25_score": 0.9, "triple_score": 0.0},
        {"block_id": "vol3_page3_table_0", "text_score": 0.3, "bm25_score": 0.3, "triple_score": 0.4},
        {"block_id": "vol4_page4_table_0", "text_score": 0.5, "bm25_score": 0.1, "triple_score": 0.8},
        {"block_id": "vol5_page5_table_0", "text_score": 0.4, "bm25_score": 0.4, "triple_score": 0.2}
    ]
    d = calc_diversity(results)
    print(f"Test 2 (バランス良い): {d:.3f} (期待値: > 0.6)")
    assert d > 0.6
    
    print("✅ All Diversity tests passed\n")


def test_combined_score():
    """総合スコアのテスト"""
    print("=" * 70)
    print("Test: Combined Score")
    print("=" * 70)
    
    precision = 0.8
    diversity = 0.7
    quality = 0.6
    
    score = calc_combined_score(precision, diversity, quality)
    expected = 0.5 * 0.8 + 0.2 * 0.7 + 0.3 * 0.6
    print(f"Combined Score: {score:.3f} (期待値: {expected:.3f})")
    assert abs(score - expected) < 0.01
    
    print("✅ Combined Score test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Evaluation Metrics Test Suite (v0.7.4)")
    print("=" * 70 + "\n")
    
    test_precision_at_k()
    test_diversity()
    test_combined_score()
    
    print("=" * 70)
    print("✅ All tests passed successfully!")
    print("=" * 70)
