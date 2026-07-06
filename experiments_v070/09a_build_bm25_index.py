"""
experiments_v070/09a_build_bm25_index.py
────────────────────────────────────────────────────────────
Phase 1 (v0.7.3): BM25キーワードインデックス構築

layout_blocks_captioned.jsonl (2,430ブロック) から BM25 インデックスを構築。
日本語バイグラムトークナイゼーション（文字 + 2-gram）を使用。

出力:
  experiments_v070/indices/bm25_index.pkl  — BM25オブジェクト + block_id マッピング

Usage:
    python experiments_v070/09a_build_bm25_index.py
    python experiments_v070/09a_build_bm25_index.py --rebuild   # 強制再構築
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

# ─────────────────────────────────────────────────────────
# パス設定
# ─────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
BLOCKS_PATH = Path(__file__).parent / "indices" / "layout_blocks_captioned.jsonl"
OUT_PATH = Path(__file__).parent / "indices" / "bm25_index.pkl"


# ─────────────────────────────────────────────────────────
# BM25トークナイゼーション（v0.6.4と同一）
# ─────────────────────────────────────────────────────────

def bm25_tokenize(text: str) -> list[str]:
    """日本語バイグラムトークナイゼーション
    
    例: '地震被害' → ['地', '震', '被', '害', '地震', '震被', '被害']
    
    Args:
        text: 入力テキスト
    
    Returns:
        文字リスト + 2-gramリスト
    """
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    return chars + bigrams


# ─────────────────────────────────────────────────────────
# ブロック読み込み
# ─────────────────────────────────────────────────────────

def load_blocks() -> tuple[list[str], list[str]]:
    """layout_blocks_captioned.jsonl からテキストコンテンツを抽出
    
    Returns:
        (corpus_texts, corpus_ids)
        - corpus_texts: BM25用のテキストリスト
        - corpus_ids: 対応するblock_idリスト
    """
    corpus_texts = []
    corpus_ids = []
    
    print(f"📖 ブロックデータ読み込み: {BLOCKS_PATH}")
    
    with open(BLOCKS_PATH, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                page_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠️ JSON解析エラー (line {line_num}): {e}")
                continue
            
            # テキストブロック処理
            for text_block in page_data.get("text_blocks", []):
                block_id = text_block.get("block_id")
                text = text_block.get("text", "").strip()
                
                if block_id and text:
                    corpus_texts.append(text)
                    corpus_ids.append(block_id)
            
            # 図表ブロック処理（キャプション + OCRテキスト）
            for figure_block in page_data.get("figure_blocks", []):
                block_id = figure_block.get("block_id")
                caption = figure_block.get("caption", "").strip()
                ocr_text = figure_block.get("ocr_text", "").strip()
                
                # キャプションとOCRテキストを統合
                combined_text = caption
                if ocr_text:
                    combined_text = f"{caption} {ocr_text}" if caption else ocr_text
                
                if block_id and combined_text:
                    corpus_texts.append(combined_text)
                    corpus_ids.append(block_id)
    
    print(f"  ✅ {len(corpus_texts)} ブロック読み込み完了")
    print(f"     - テキストブロック + 図表ブロック（キャプション + OCR）")
    
    return corpus_texts, corpus_ids


# ─────────────────────────────────────────────────────────
# BM25インデックス構築
# ─────────────────────────────────────────────────────────

def build_bm25_index(corpus_texts: list[str]) -> object:
    """BM25Okapiインデックスを構築
    
    Args:
        corpus_texts: ブロックテキストリスト
    
    Returns:
        BM25Okapi インデックスオブジェクト
    """
    from rank_bm25 import BM25Okapi
    
    print(f"🔨 BM25インデックス構築中...")
    print(f"   トークナイゼーション: 文字 + 2-gram")
    
    # 各テキストをトークナイズ
    tokenized_corpus = []
    for i, text in enumerate(corpus_texts):
        tokens = bm25_tokenize(text)
        tokenized_corpus.append(tokens)
        
        if (i + 1) % 500 == 0:
            print(f"   進捗: {i + 1}/{len(corpus_texts)} ブロック処理完了", end="\r")
    
    print(f"   進捗: {len(corpus_texts)}/{len(corpus_texts)} ブロック処理完了")
    
    # BM25インデックス構築
    bm25_index = BM25Okapi(tokenized_corpus)
    
    print(f"  ✅ BM25インデックス構築完了")
    
    return bm25_index


# ─────────────────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────────────────

def save_bm25_index(bm25_index: object, corpus_ids: list[str]):
    """BM25インデックスとblock_idマッピングを保存
    
    Args:
        bm25_index: BM25Okapiオブジェクト
        corpus_ids: block_idリスト
    """
    print(f"💾 保存中: {OUT_PATH}")
    
    # データ構造
    bm25_data = {
        "bm25": bm25_index,
        "corpus_ids": corpus_ids,
        "num_blocks": len(corpus_ids),
    }
    
    with open(OUT_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    
    print(f"  ✅ 保存完了: {OUT_PATH.stat().st_size / 1024:.1f} KB")


# ─────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BM25インデックス構築")
    parser.add_argument("--rebuild", action="store_true", help="既存インデックスを強制再構築")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BM25 Index Builder (v0.7.3)")
    print("=" * 60)
    
    # 既存チェック
    if OUT_PATH.exists() and not args.rebuild:
        print(f"⚠️ BM25インデックスは既に存在します: {OUT_PATH}")
        print(f"   再構築する場合は --rebuild オプションを使用してください。")
        return
    
    # データチェック
    if not BLOCKS_PATH.exists():
        print(f"❌ エラー: ブロックデータが見つかりません: {BLOCKS_PATH}")
        print(f"   Phase 3 (03_multimodal_captioning.py) を先に実行してください。")
        return
    
    # 処理開始
    print()
    corpus_texts, corpus_ids = load_blocks()
    
    if not corpus_texts:
        print("❌ エラー: ブロックが読み込めませんでした。")
        return
    
    print()
    bm25_index = build_bm25_index(corpus_texts)
    
    print()
    save_bm25_index(bm25_index, corpus_ids)
    
    print()
    print("=" * 60)
    print("✅ BM25インデックス構築完了")
    print("=" * 60)
    print(f"📊 統計:")
    print(f"   - ブロック数: {len(corpus_ids)}")
    print(f"   - 出力ファイル: {OUT_PATH}")
    print()
    print("次のステップ:")
    print("  1. experiments_v070/06_multimodal_retriever.py を修正")
    print("  2. experiments_v070/07d_evaluate_multimodal_rag.py で評価実行")


if __name__ == "__main__":
    main()
