# Copyright 2026 Multimodal CoLRAG-TF contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CoLRAG-TF v0.7.0 - Phase 4: マルチモーダルインデックス構築

テキスト・画像埋め込みを生成し、Qdrantベクトルストアに保存します。
3層階層（Volume→Chapter→Chunk）の代表ベクトルも生成します。

Usage:
    .venv-coltf\\Scripts\\python.exe experiments_v070\\03_build_multimodal_index.py
    .venv-coltf\\Scripts\\python.exe experiments_v070\\03_build_multimodal_index.py --skip-text
"""

import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers imported")
except ImportError as e:
    print(f"❌ sentence-transformers not installed: {e}")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    print("✅ qdrant-client imported")
except ImportError as e:
    print(f"❌ qdrant-client not installed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    import torch
    from tqdm import tqdm
    print("✅ All dependencies loaded")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


class MultimodalEmbeddingGenerator:
    """マルチモーダル埋め込み生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # テキスト埋め込みモデル
        text_model_name = config['embedding']['text']['model']
        print(f"📥 テキスト埋め込みモデル読み込み: {text_model_name}")
        self.text_model = SentenceTransformer(text_model_name)
        self.text_dim = config['embedding']['text']['dimension']
        
        print(f"   ✅ テキストモデル準備完了 (dim={self.text_dim})")
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ
        
        Returns:
            埋め込みベクトル (N, text_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.text_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.config['embedding']['text']['normalize']
        )
        
        return embeddings
    
    def encode_text_single(self, text: str) -> np.ndarray:
        """単一テキストの埋め込み"""
        return self.encode_text([text])[0]


class MultimodalIndexBuilder:
    """マルチモーダルインデックス構築器"""
    
    def __init__(self, config: Dict[str, Any], embedding_generator: MultimodalEmbeddingGenerator):
        self.config = config
        self.embedding_gen = embedding_generator
        
        # Qdrantクライアント初期化
        self.collection_name = "colragtf_v070_multimodal"
        self.client = QdrantClient(":memory:")  # メモリ内で構築
        
        print(f"✅ Qdrantクライアント初期化 (collection: {self.collection_name})")
    
    def create_collection(self):
        """Qdrantコレクションを作成"""
        vector_size = self.embedding_gen.text_dim
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        print(f"✅ コレクション作成完了 (vector_size={vector_size})")
    
    def build_from_layouts(self, layouts: List[Dict]) -> Dict[str, Any]:
        """
        レイアウトブロックからインデックスを構築
        
        Args:
            layouts: レイアウトブロックのリスト
        
        Returns:
            統計情報
        """
        print("\n" + "=" * 60)
        print(" インデックス構築開始")
        print("=" * 60)
        
        # テキストブロックとキャプション付き図表ブロックを抽出
        all_blocks = []
        
        for layout in tqdm(layouts, desc="ブロック抽出"):
            # テキストブロック
            for text_block in layout.get('text_blocks', []):
                block_data = {
                    'id': text_block['block_id'],
                    'type': 'text',
                    'content': text_block.get('text', ''),
                    'page_id': layout['page_id'],
                    'pdf_name': layout['pdf_name'],
                    'page_num': layout['page_num'],
                    'volume': layout.get('volume', ''),
                    'chapter': layout.get('chapter', ''),
                    'bbox': text_block['bbox'],
                    'confidence': text_block.get('confidence', 1.0)
                }
                all_blocks.append(block_data)
            
            # 図表ブロック（キャプション付き）
            for fig_block in layout.get('figure_blocks', []):
                caption = fig_block.get('caption', '')
                if caption and not caption.startswith('[API エラー'):
                    block_data = {
                        'id': fig_block['block_id'],
                        'type': fig_block['type'],
                        'content': caption,
                        'page_id': layout['page_id'],
                        'pdf_name': layout['pdf_name'],
                        'page_num': layout['page_num'],
                        'volume': layout.get('volume', ''),
                        'chapter': layout.get('chapter', ''),
                        'bbox': fig_block['bbox'],
                        'confidence': fig_block.get('confidence', 1.0),
                        'image_path': fig_block.get('image_path', '')
                    }
                    all_blocks.append(block_data)
        
        print(f"\n総ブロック数: {len(all_blocks)}")
        
        # タイプ別カウント
        type_counts = {}
        for block in all_blocks:
            block_type = block['type']
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        print("\nブロックタイプ別統計:")
        for block_type, count in sorted(type_counts.items()):
            print(f"   - {block_type}: {count} 個")
        
        # テキスト埋め込み生成
        print("\n📝 テキスト埋め込み生成中...")
        texts = [block['content'] for block in all_blocks]
        embeddings = self.embedding_gen.encode_text(texts, batch_size=32)
        
        print(f"   ✅ {len(embeddings)} 個の埋め込み生成完了")
        
        # Qdrantに追加
        print("\n💾 Qdrantコレクションに追加中...")
        points = []
        for i, (block, embedding) in enumerate(zip(all_blocks, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=block
            )
            points.append(point)
        
        # バッチで追加
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Qdrant追加"):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"   ✅ {len(points)} ポイント追加完了")
        
        # 階層構造の構築
        hierarchy = self.build_hierarchy(all_blocks, embeddings)
        
        stats = {
            'total_blocks': len(all_blocks),
            'type_counts': type_counts,
            'vector_dim': self.embedding_gen.text_dim,
            'hierarchy': hierarchy
        }
        
        return stats
    
    def build_hierarchy(self, blocks: List[Dict], embeddings: np.ndarray) -> Dict[str, Any]:
        """
        3層階層構造を構築
        
        Volume → Chapter → Chunk (Block)
        各レベルの代表ベクトルを平均ベクトルとして計算
        """
        print("\n🏗️  階層構造構築中...")
        
        # Volume別に集約
        volumes = {}
        for i, block in enumerate(blocks):
            volume = block['volume']
            chapter = block['chapter']
            
            if volume not in volumes:
                volumes[volume] = {
                    'chapters': {},
                    'block_ids': [],
                    'embeddings': []
                }
            
            if chapter not in volumes[volume]['chapters']:
                volumes[volume]['chapters'][chapter] = {
                    'block_ids': [],
                    'embeddings': []
                }
            
            volumes[volume]['block_ids'].append(block['id'])
            volumes[volume]['embeddings'].append(embeddings[i])
            volumes[volume]['chapters'][chapter]['block_ids'].append(block['id'])
            volumes[volume]['chapters'][chapter]['embeddings'].append(embeddings[i])
        
        # 代表ベクトル計算
        hierarchy = {}
        for volume_name, volume_data in volumes.items():
            volume_embedding = np.mean(volume_data['embeddings'], axis=0)
            
            chapters = {}
            for chapter_name, chapter_data in volume_data['chapters'].items():
                chapter_embedding = np.mean(chapter_data['embeddings'], axis=0)
                chapters[chapter_name] = {
                    'block_count': len(chapter_data['block_ids']),
                    'representative_vector': chapter_embedding.tolist()
                }
            
            hierarchy[volume_name] = {
                'block_count': len(volume_data['block_ids']),
                'chapter_count': len(chapters),
                'chapters': chapters,
                'representative_vector': volume_embedding.tolist()
            }
        
        print(f"   ✅ {len(hierarchy)} Volumes, {sum(v['chapter_count'] for v in hierarchy.values())} Chapters")
        
        return hierarchy
    
    def save_index(self, output_dir: Path):
        """インデックスをディスクに保存"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Qdrantスナップショット保存
        snapshot_path = output_dir / "qdrant_collection"
        # Note: インメモリQdrantはスナップショット未対応のため、実装時は永続化設定が必要
        print(f"   ⚠️  インメモリQdrantのため、永続化には別途設定が必要です")
        
        return snapshot_path


def load_config(config_path: Path) -> Dict:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_layouts(input_path: Path) -> List[Dict]:
    """レイアウトブロックJSONLを読み込み"""
    layouts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                layouts.append(json.loads(line))
    return layouts


def save_hierarchy(hierarchy: Dict, output_path: Path):
    """階層構造をJSONで保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="マルチモーダルインデックス構築")
    parser.add_argument('--config', type=str, 
                        default='experiments_v070/configs/mm_llm_config.yaml',
                        help='マルチモーダルLLM設定ファイル')
    parser.add_argument('--input', type=str,
                        default='experiments_v070/indices/layout_blocks_captioned.jsonl',
                        help='入力キャプション付きレイアウトブロックファイル')
    parser.add_argument('--output-dir', type=str,
                        default='experiments_v070/indices',
                        help='出力ディレクトリ')
    parser.add_argument('--skip-text', action='store_true',
                        help='テキストブロックをスキップ（図表のみ）')
    
    args = parser.parse_args()
    
    # パス解決
    config_path = Path(args.config)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print(" マルチモーダルインデックス構築")
    print("=" * 60)
    print()
    
    # 設定読み込み
    print("📋 設定読み込み...")
    config = load_config(config_path)
    
    # レイアウトブロック読み込み
    print(f"📄 レイアウトブロック読み込み: {input_path}")
    layouts = load_layouts(input_path)
    print(f"   ✅ {len(layouts)} ページ読み込み完了")
    
    # 埋め込み生成器初期化
    print("\n🧠 埋め込み生成器初期化...")
    embedding_gen = MultimodalEmbeddingGenerator(config)
    
    # インデックスビルダー初期化
    print("\n🏗️  インデックスビルダー初期化...")
    index_builder = MultimodalIndexBuilder(config, embedding_gen)
    index_builder.create_collection()
    
    # インデックス構築
    stats = index_builder.build_from_layouts(layouts)
    
    # 階層構造保存
    hierarchy_path = output_dir / "mm_hierarchy.json"
    print(f"\n💾 階層構造保存: {hierarchy_path}")
    save_hierarchy(stats['hierarchy'], hierarchy_path)
    
    # インデックス保存
    print(f"\n💾 インデックス保存: {output_dir}")
    index_builder.save_index(output_dir)
    
    print("\n" + "=" * 60)
    print(" インデックス構築完了")
    print("=" * 60)
    print(f"総ブロック数: {stats['total_blocks']}")
    print(f"ベクトル次元: {stats['vector_dim']}")
    print(f"Volume数: {len(stats['hierarchy'])}")
    print(f"Chapter数: {sum(v['chapter_count'] for v in stats['hierarchy'].values())}")
    print()
    print("ブロックタイプ別:")
    for block_type, count in sorted(stats['type_counts'].items()):
        print(f"   - {block_type}: {count} 個")
    print()
    print(f"出力ディレクトリ: {output_dir.absolute()}")
    print()
    print("次のステップ:")
    print("  python 04_extract_mm_triples.py")


if __name__ == "__main__":
    main()
