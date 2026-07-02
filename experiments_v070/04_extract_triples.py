"""
CoLRAG-TF v0.7.0 - Phase 5: マルチモーダルTriple抽出

図表キャプション（テキスト化済み）からOpenIE形式のtriple（Subject-Predicate-Object）を抽出します。
HippoRAG2スタイルの知識グラフ構築のためのtriple抽出を実装します。

Usage:
    .venv-coltf\\Scripts\\python.exe experiments_v070\\04_extract_triples.py
    .venv-coltf\\Scripts\\python.exe experiments_v070\\04_extract_triples.py --max-blocks 10
"""

import sys
import argparse
import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import time


@dataclass
class Triple:
    """OpenIE Triple (Subject-Predicate-Object)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_block_id: str = ""
    source_type: str = ""  # table, figure, text


class OllamaTripleExtractor:
    """Ollamaを使用したTriple抽出器"""
    
    TRIPLE_EXTRACTION_PROMPT = """あなたは災害教訓文書から知識を抽出する専門家です。

以下のテキストから、重要な事実関係を「主語-述語-目的語」（Subject-Predicate-Object）の3つ組（triple）として抽出してください。

【抽出ルール】
1. 具体的な数値、場所、日付、被害状況などの事実関係を優先
2. 1つのtripleは1つの明確な事実を表現
3. 主語と目的語は名詞句、述語は動詞または関係性を表す表現
4. 災害対策や教訓に関連する情報を重視
5. 各tripleは独立して理解可能であること

【出力形式】
必ず以下のJSON形式で出力してください。各tripleを1行ずつ出力してください。

例:
{{"subject": "台風12号", "predicate": "による", "object": "災害"}}
{{"subject": "全壊家屋", "predicate": "は", "object": "367棟"}}
{{"subject": "死者", "predicate": "は", "object": "56人"}}

【入力テキスト】
{text}

【抽出されたTriples（必ずJSON形式で出力）】
"""

    def __init__(self, model_name: str = "qwen2.5:7b-instruct-q4_k_m", 
                 ollama_url: str = "http://localhost:11434",
                 debug: bool = False):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        self.debug = debug
        print(f"✅ OllamaTripleExtractor initialized: {model_name}")
        if debug:
            print(f"   🐞 Debug mode enabled")
    
    def extract_triples_from_text(self, text: str, 
                                   max_retries: int = 3) -> List[Triple]:
        """テキストからtripleを抽出"""
        if not text or len(text.strip()) < 10:
            return []
        
        prompt = self.TRIPLE_EXTRACTION_PROMPT.format(text=text)
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # 低温度で安定した抽出
                            "num_predict": 512,
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    
                    if self.debug:
                        print(f"\n{'='*60}")
                        print(f"Ollama Response (first 500 chars):")
                        print(response_text[:500])
                        print(f"{'='*60}\n")
                    
                    triples = self._parse_triples_from_response(response_text)
                    
                    if self.debug and len(triples) > 0:
                        print(f"✅ Extracted {len(triples)} triples")
                        for i, triple in enumerate(triples[:3]):
                            print(f"   {i+1}. {triple.subject} → {triple.predicate} → {triple.object}")
                    elif self.debug:
                        print(f"⚠️ No triples extracted from response")
                    
                    return triples
                else:
                    print(f"⚠️ Ollama API error (attempt {attempt+1}/{max_retries}): {response.status_code}")
                    time.sleep(2)
                    
            except requests.exceptions.Timeout:
                print(f"⚠️ Timeout (attempt {attempt+1}/{max_retries})")
                time.sleep(2)
            except Exception as e:
                print(f"❌ Error extracting triples: {e}")
                break
        
        return []
    
    def _parse_triples_from_response(self, response_text: str) -> List[Triple]:
        """LLM応答からtripleをパース"""
        triples = []
        
        # JSON形式の行を探す（複数の方法を試す）
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # 空行やコメント行をスキップ
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # JSONブロックの開始・終了マーカーをスキップ
            if line in ['```json', '```', '【抽出されたTriples】']:
                continue
            
            # JSON形式の行を探す
            # 波括弧で囲まれた部分を探す
            json_matches = re.findall(r'\{[^}]+\}', line)
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if all(k in data for k in ['subject', 'predicate', 'object']):
                        triple = Triple(
                            subject=data['subject'].strip(),
                            predicate=data['predicate'].strip(),
                            object=data['object'].strip(),
                            confidence=data.get('confidence', 1.0)
                        )
                        # 空文字列や短すぎるtripleをフィルタ
                        if all(len(s) > 0 for s in [triple.subject, triple.predicate, triple.object]):
                            triples.append(triple)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return triples


class MultimodalTripleExtractor:
    """マルチモーダルブロックからtriple抽出"""
    
    def __init__(self, extractor: OllamaTripleExtractor):
        self.extractor = extractor
    
    def process_captioned_blocks(self, 
                                  captioned_blocks_path: Path,
                                  max_blocks: Optional[int] = None) -> Dict[str, Any]:
        """キャプション付きブロックからtripleを抽出"""
        print(f"\n{'='*60}")
        print(f"Processing: {captioned_blocks_path}")
        print(f"{'='*60}\n")
        
        # ページレベルのレイアウトを読み込み
        pages = []
        with open(captioned_blocks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pages.append(json.loads(line))
        
        print(f"📄 Loaded {len(pages)} pages")
        
        # ページからブロックを抽出
        all_blocks = []
        for page in pages:
            # figure_blocksを抽出（キャプション付き）
            figure_blocks = page.get('figure_blocks', [])
            for fig in figure_blocks:
                if fig.get('caption'):
                    all_blocks.append({
                        'block_id': fig.get('block_id', 'unknown'),
                        'block_type': fig.get('type', 'figure'),
                        'text': fig.get('caption', '')
                    })
            
            # text_blocksを抽出
            text_blocks = page.get('text_blocks', [])
            for txt in text_blocks:
                if txt.get('text') and len(txt.get('text', '').strip()) > 50:
                    all_blocks.append({
                        'block_id': txt.get('block_id', 'unknown'),
                        'block_type': 'text',
                        'text': txt.get('text', '')
                    })
        
        if max_blocks:
            all_blocks = all_blocks[:max_blocks]
            print(f"📊 Processing {len(all_blocks)} blocks (limited)")
        else:
            print(f"📊 Processing {len(all_blocks)} blocks")
        
        # Triple抽出
        all_triples = []
        block_triple_counts = {}
        
        for block in tqdm(all_blocks, desc="Extracting triples"):
            block_id = block.get('block_id', 'unknown')
            block_type = block.get('block_type', 'unknown')
            text = block.get('text', '')
            
            if not text or len(text.strip()) < 20:
                continue
            
            # Triple抽出
            triples = self.extractor.extract_triples_from_text(text)
            
            # メタデータを追加
            for triple in triples:
                triple.source_block_id = block_id
                triple.source_type = block_type
            
            all_triples.extend(triples)
            block_triple_counts[block_id] = len(triples)
        
        # 統計情報
        stats = {
            "total_pages": len(pages),
            "total_blocks": len(all_blocks),
            "blocks_with_triples": len([c for c in block_triple_counts.values() if c > 0]),
            "total_triples": len(all_triples),
            "avg_triples_per_block": len(all_triples) / len(all_blocks) if all_blocks else 0,
            "triples_by_type": self._count_by_type(all_triples)
        }
        
        return {
            "triples": all_triples,
            "stats": stats,
            "block_counts": block_triple_counts
        }
    
    def _count_by_type(self, triples: List[Triple]) -> Dict[str, int]:
        """タイプ別のtriple数をカウント"""
        counts = {}
        for triple in triples:
            counts[triple.source_type] = counts.get(triple.source_type, 0) + 1
        return counts
    
    def save_triples(self, triples: List[Triple], output_path: Path):
        """Tripleを保存"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for triple in triples:
                f.write(json.dumps(asdict(triple), ensure_ascii=False) + '\n')
        print(f"✅ Saved {len(triples)} triples to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract triples from multimodal captions")
    parser.add_argument("--captioned-blocks", type=str,
                       default="experiments_v070/indices/layout_blocks_captioned.jsonl",
                       help="Path to captioned blocks JSONL")
    parser.add_argument("--output", type=str,
                       default="experiments_v070/indices/mm_triples.jsonl",
                       help="Path to output triples JSONL")
    parser.add_argument("--model", type=str,
                       default="qwen2.5:7b-instruct-q4_k_m",
                       help="Ollama model name")
    parser.add_argument("--max-blocks", type=int, default=None,
                       help="Maximum number of blocks to process (for testing)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output to see Ollama responses")
    
    args = parser.parse_args()
    
    # パスの準備
    captioned_blocks_path = Path(args.captioned_blocks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not captioned_blocks_path.exists():
        print(f"❌ Captioned blocks not found: {captioned_blocks_path}")
        print("⚠️ Run 02_multimodal_caption.py first")
        sys.exit(1)
    
    # Triple抽出
    print("🚀 Starting triple extraction...\n")
    
    ollama_extractor = OllamaTripleExtractor(model_name=args.model, debug=args.debug)
    mm_extractor = MultimodalTripleExtractor(ollama_extractor)
    
    result = mm_extractor.process_captioned_blocks(
        captioned_blocks_path,
        max_blocks=args.max_blocks
    )
    
    # 結果を保存
    mm_extractor.save_triples(result['triples'], output_path)
    
    # 統計情報を表示
    print(f"\n{'='*60}")
    print("📊 Triple Extraction Statistics")
    print(f"{'='*60}")
    stats = result['stats']
    print(f"Total pages processed: {stats['total_pages']}")
    print(f"Total blocks processed: {stats['total_blocks']}")
    print(f"Blocks with triples: {stats['blocks_with_triples']}")
    print(f"Total triples extracted: {stats['total_triples']}")
    print(f"Average triples per block: {stats['avg_triples_per_block']:.2f}")
    print(f"\nTriples by type:")
    for block_type, count in stats['triples_by_type'].items():
        print(f"  - {block_type}: {count}")
    
    # 統計情報をJSON保存
    stats_path = output_path.parent / "mm_triples_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': stats,
            'block_counts': result['block_counts']
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Stats saved to: {stats_path}")
    
    print(f"\n{'='*60}")
    print("✨ Triple extraction completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
