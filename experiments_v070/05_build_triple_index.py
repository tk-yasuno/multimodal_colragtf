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
CoLRAG-TF v0.7.0 - Phase 5: Triple EmbeddingとFAISSインデックス構築

抽出されたtripleから埋め込みベクトルを生成し、FAISSインデックスを構築します。
HippoRAG2スタイルの知識グラフ検索のためのインデックスを作成します。

Usage:
    .venv-coltf\\Scripts\\python.exe experiments_v070\\05_build_triple_index.py
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers imported")
except ImportError as e:
    print(f"❌ sentence-transformers not installed: {e}")
    sys.exit(1)

try:
    import faiss
    print("✅ FAISS imported")
except ImportError as e:
    print(f"❌ FAISS not installed: {e}")
    print("Install: pip install faiss-cpu  (or faiss-gpu)")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")
    sys.exit(1)


@dataclass
class Triple:
    """Triple data structure"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_block_id: str = ""
    source_type: str = ""


class TripleEmbeddingGenerator:
    """Triple埋め込みベクトル生成器"""
    
    def __init__(self, model_name: str = "hotchpotch/static-embedding-japanese",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        print(f"\n🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded | Dimension: {self.embedding_dim} | Device: {device}")
    
    def triple_to_text(self, triple: Triple) -> str:
        """Tripleをテキスト表現に変換"""
        # 日本語の自然な文として表現
        return f"{triple.subject}は{triple.predicate}{triple.object}"
    
    def generate_embeddings(self, triples: List[Triple], 
                           batch_size: int = 32) -> np.ndarray:
        """Tripleのリストから埋め込みベクトルを生成"""
        print(f"\n🔄 Generating embeddings for {len(triples)} triples...")
        
        # Tripleをテキストに変換
        texts = [self.triple_to_text(triple) for triple in triples]
        
        # バッチ処理で埋め込み生成
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # コサイン類似度のため正規化
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"✅ Generated embeddings: shape {embeddings.shape}")
        
        return embeddings


class FAISSTripleIndex:
    """FAISS Triple インデックス"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Inner Product (IP) インデックス（正規化済みベクトルでコサイン類似度と等価）
        self.index = faiss.IndexFlatIP(dimension)
        print(f"✅ FAISS IndexFlatIP created | Dimension: {dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray):
        """埋め込みベクトルをインデックスに追加"""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        self.index.add(embeddings)
        print(f"✅ Added {embeddings.shape[0]} vectors to index | Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> tuple:
        """類似tripleを検索"""
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
    
    def save(self, index_path: Path):
        """インデックスを保存"""
        faiss.write_index(self.index, str(index_path))
        print(f"✅ FAISS index saved to: {index_path}")
    
    @classmethod
    def load(cls, index_path: Path, dimension: int):
        """インデックスを読み込み"""
        obj = cls(dimension)
        obj.index = faiss.read_index(str(index_path))
        print(f"✅ FAISS index loaded from: {index_path} | Total: {obj.index.ntotal}")
        return obj


def load_triples(triples_path: Path) -> List[Triple]:
    """Tripleを読み込み"""
    triples = []
    with open(triples_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                triples.append(Triple(**data))
    print(f"✅ Loaded {len(triples)} triples from: {triples_path}")
    return triples


def save_triples_metadata(triples: List[Triple], output_path: Path):
    """Triple メタデータを保存（検索結果のマッピング用）"""
    metadata = []
    for i, triple in enumerate(triples):
        metadata.append({
            'triple_id': i,
            'subject': triple.subject,
            'predicate': triple.predicate,
            'object': triple.object,
            'text': f"{triple.subject}は{triple.predicate}{triple.object}",
            'confidence': triple.confidence,
            'source_block_id': triple.source_block_id,
            'source_type': triple.source_type
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Triple metadata saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from triples")
    parser.add_argument("--triples", type=str,
                       default="experiments_v070/indices/mm_triples.jsonl",
                       help="Path to triples JSONL")
    parser.add_argument("--output-index", type=str,
                       default="experiments_v070/indices/mm_triple.index",
                       help="Path to output FAISS index")
    parser.add_argument("--output-metadata", type=str,
                       default="experiments_v070/indices/mm_triples_metadata.json",
                       help="Path to output triple metadata JSON")
    parser.add_argument("--embedding-model", type=str,
                       default="hotchpotch/static-embedding-japanese",
                       help="Sentence Transformer model name")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding generation")
    
    args = parser.parse_args()
    
    # パスの準備
    triples_path = Path(args.triples)
    index_path = Path(args.output_index)
    metadata_path = Path(args.output_metadata)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not triples_path.exists():
        print(f"❌ Triples file not found: {triples_path}")
        print("⚠️ Run 04_extract_triples.py first")
        sys.exit(1)
    
    print("🚀 Starting triple index building...\n")
    print(f"{'='*60}")
    print(f"Input: {triples_path}")
    print(f"Output index: {index_path}")
    print(f"Output metadata: {metadata_path}")
    print(f"{'='*60}\n")
    
    # Tripleを読み込み
    triples = load_triples(triples_path)
    
    if len(triples) == 0:
        print("❌ No triples found. Nothing to index.")
        sys.exit(1)
    
    # 埋め込み生成
    embedding_generator = TripleEmbeddingGenerator(
        model_name=args.embedding_model
    )
    embeddings = embedding_generator.generate_embeddings(
        triples,
        batch_size=args.batch_size
    )
    
    # FAISSインデックス構築
    print("\n🔄 Building FAISS index...")
    faiss_index = FAISSTripleIndex(dimension=embedding_generator.embedding_dim)
    faiss_index.add_embeddings(embeddings)
    
    # インデックス保存
    faiss_index.save(index_path)
    
    # メタデータ保存
    save_triples_metadata(triples, metadata_path)
    
    # 統計情報
    print(f"\n{'='*60}")
    print("📊 Triple Index Statistics")
    print(f"{'='*60}")
    print(f"Total triples: {len(triples)}")
    print(f"Embedding dimension: {embedding_generator.embedding_dim}")
    print(f"Index type: FAISS IndexFlatIP (Inner Product)")
    print(f"Index size: {faiss_index.index.ntotal} vectors")
    
    # タイプ別統計
    type_counts = {}
    for triple in triples:
        type_counts[triple.source_type] = type_counts.get(triple.source_type, 0) + 1
    print(f"\nTriples by source type:")
    for source_type, count in type_counts.items():
        print(f"  - {source_type}: {count}")
    
    print(f"\n{'='*60}")
    print("✨ Triple index building completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
