"""
CoLRAG-TF v0.7.0 - Phase 6: マルチモーダルRetriever実装

HippoRAG2スタイルのcoarse-to-fine検索にマルチモーダル機能を統合します。
- Volume→Chapter→Block の3段階階層検索
- テキスト埋め込み + Triple埋め込み + 画像類似度のスコア融合
- 図表関連キーワード検出によるクエリ解析

Usage:
    .venv-coltf\\Scripts\\python.exe experiments_v070\\06_multimodal_retriever.py --query "台風12号の被害状況"
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    print("✅ sentence-transformers & torch imported")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

try:
    import faiss
    print("✅ FAISS imported")
except ImportError as e:
    print(f"❌ FAISS not installed: {e}")
    sys.exit(1)

try:
    import requests
    from PIL import Image
    import base64
    from io import BytesIO
    print("✅ requests, PIL imported for multimodal LLM")
except ImportError as e:
    print(f"⚠️ Optional dependencies for image understanding: {e}")


@dataclass
class RetrievalResult:
    """検索結果"""
    block_id: str
    block_type: str  # text, table, figure
    content: str  # キャプションまたはテキスト
    score: float
    volume: str = ""
    chapter: str = ""
    page_num: int = 0
    bbox: List[float] = field(default_factory=list)
    image_path: str = ""
    # スコア内訳
    text_score: float = 0.0
    triple_score: float = 0.0
    image_score: float = 0.0
    calibrated_score: float = 0.0


@dataclass
class QueryAnalysis:
    """クエリ解析結果"""
    original_query: str
    is_figure_related: bool = False  # 図表関連クエリかどうか
    figure_keywords: List[str] = field(default_factory=list)
    needs_image: bool = False  # 画像理解が必要か
    entity_mentions: List[str] = field(default_factory=list)


class QueryAnalyzer:
    """クエリ解析器"""
    
    # 図表関連キーワード
    FIGURE_KEYWORDS = [
        "表", "図", "グラフ", "写真", "画像", "地図", "図表",
        "表示", "図示", "チャート", "ダイアグラム", "イラスト"
    ]
    
    # 視覚的情報を必要とするキーワード
    VISUAL_KEYWORDS = [
        "見せて", "表示", "どのような", "様子", "状況", "分布",
        "推移", "変化", "傾向", "パターン", "比較"
    ]
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """クエリを解析して検索戦略を決定"""
        analysis = QueryAnalysis(original_query=query)
        
        # 図表関連キーワード検出
        found_keywords = []
        for keyword in self.FIGURE_KEYWORDS:
            if keyword in query:
                found_keywords.append(keyword)
                analysis.is_figure_related = True
        analysis.figure_keywords = found_keywords
        
        # 視覚的情報の必要性を判定
        for keyword in self.VISUAL_KEYWORDS:
            if keyword in query:
                analysis.needs_image = True
                break
        
        return analysis


class OllamaMultimodalClient:
    """Ollama マルチモーダルLLMクライアント（画像理解用）"""
    
    def __init__(self, 
                 model_name: str = "qwen2.5:7b-instruct-q4_k_m",
                 base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        print(f"✅ OllamaMultimodalClient initialized: {model_name}")
    
    def encode_image(self, image_path: Path) -> Optional[str]:
        """画像をBase64エンコード"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Image encoding error: {e}")
            return None
    
    def understand_image_with_query(self, 
                                     image_path: Path, 
                                     query: str,
                                     caption: str = "") -> str:
        """
        クエリと画像から詳細な回答を生成
        
        Args:
            image_path: 画像パス
            query: ユーザークエリ
            caption: 既存キャプション（コンテキスト）
        
        Returns:
            LLMの回答
        """
        if not image_path.exists():
            return f"[画像が見つかりません: {image_path}]"
        
        image_base64 = self.encode_image(image_path)
        if not image_base64:
            return "[画像の読み込みに失敗しました]"
        
        # プロンプト構築
        prompt = f"""あなたは災害教訓文書の図表を分析する専門家です。

以下の質問に、提供された図表の画像を参照して回答してください。

【質問】
{query}

【既存のキャプション（参考）】
{caption if caption else "（なし）"}

【回答】
図表の内容を確認し、質問に対する具体的な回答を日本語で提供してください。"""
        
        try:
            # Ollama Chat API（マルチモーダル）
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_base64]
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 512
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('message', {}).get('content', '').strip()
                return answer if answer else "[回答を生成できませんでした]"
            else:
                return f"[APIエラー: {response.status_code}]"
        
        except Exception as e:
            return f"[画像理解エラー: {str(e)[:50]}]"


class MultimodalHippoRAG2Retriever:
    """マルチモーダルHippoRAG2スタイルRetriever"""
    
    def __init__(self, 
                 embedding_model_name: str = "hotchpotch/static-embedding-japanese",
                 blocks_path: Optional[Path] = None,
                 triple_index_path: Optional[Path] = None,
                 triple_metadata_path: Optional[Path] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 enable_image_understanding: bool = False,
                 ollama_model: str = "qwen2.5:7b-instruct-q4_k_m"):
        """
        Args:
            embedding_model_name: 埋め込みモデル名
            blocks_path: layout_blocks_captioned.jsonl パス
            triple_index_path: Triple FAISSインデックスパス
            triple_metadata_path: Triple メタデータパス
            device: デバイス (cuda/cpu)
            enable_image_understanding: 画像理解を有効化
            ollama_model: Ollamaマルチモーダルモデル名
        """
        self.device = device
        self.query_analyzer = QueryAnalyzer()
        self.enable_image_understanding = enable_image_understanding
        
        # 埋め込みモデル
        print(f"\n🔄 Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded | Dimension: {self.embedding_dim} | Device: {device}")
        
        # 画像理解クライアント（オプション）
        self.multimodal_client = None
        if enable_image_understanding:
            print(f"\n🔄 Initializing multimodal LLM client...")
            self.multimodal_client = OllamaMultimodalClient(model_name=ollama_model)
        
        # ブロックデータを読み込み
        self.blocks = []
        self.block_embeddings = None
        if blocks_path and blocks_path.exists():
            print(f"\n🔄 Loading blocks from: {blocks_path}")
            self.blocks = self._load_blocks(blocks_path)
            print(f"✅ Loaded {len(self.blocks)} blocks")
            
            # ブロック埋め込み生成
            print(f"🔄 Generating block embeddings...")
            self.block_embeddings = self._generate_block_embeddings()
            print(f"✅ Block embeddings ready: shape {self.block_embeddings.shape}")
        
        # Triple検索インデックス（オプション）
        self.triple_index = None
        self.triple_metadata = []
        if triple_index_path and triple_index_path.exists():
            print(f"\n🔄 Loading triple index: {triple_index_path}")
            self.triple_index = faiss.read_index(str(triple_index_path))
            print(f"✅ Triple index loaded | Total: {self.triple_index.ntotal}")
            
            if triple_metadata_path and triple_metadata_path.exists():
                with open(triple_metadata_path, 'r', encoding='utf-8') as f:
                    self.triple_metadata = json.load(f)
                print(f"✅ Triple metadata loaded | Count: {len(self.triple_metadata)}")
    
    def _load_blocks(self, blocks_path: Path) -> List[Dict[str, Any]]:
        """ページレイアウトからブロックを抽出"""
        all_blocks = []
        with open(blocks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                page = json.loads(line)
                
                # figure_blocks（キャプション付き）
                for fig in page.get('figure_blocks', []):
                    if fig.get('caption'):
                        all_blocks.append({
                            'block_id': fig.get('block_id', ''),
                            'block_type': fig.get('type', 'figure'),
                            'content': fig.get('caption', ''),
                            'volume': page.get('volume', ''),
                            'chapter': page.get('chapter', ''),
                            'page_num': page.get('page_num', 0),
                            'bbox': fig.get('bbox', []),
                            'image_path': fig.get('image_path', '')
                        })
                
                # text_blocks
                for txt in page.get('text_blocks', []):
                    text_content = txt.get('text', '').strip()
                    if text_content and len(text_content) > 50:
                        all_blocks.append({
                            'block_id': txt.get('block_id', ''),
                            'block_type': 'text',
                            'content': text_content,
                            'volume': page.get('volume', ''),
                            'chapter': page.get('chapter', ''),
                            'page_num': page.get('page_num', 0),
                            'bbox': txt.get('bbox', []),
                            'image_path': ''
                        })
        
        return all_blocks
    
    def _generate_block_embeddings(self) -> np.ndarray:
        """ブロックの埋め込みを生成"""
        contents = [block['content'] for block in self.blocks]
        embeddings = self.embedding_model.encode(
            contents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def retrieve(self, 
                 query: str,
                 top_k: int = 10,
                 volume_filter: Optional[str] = None,
                 chapter_filter: Optional[str] = None,
                 alpha_text: float = 0.5,
                 alpha_triple: float = 0.3,
                 alpha_image: float = 0.2) -> List[RetrievalResult]:
        """
        マルチモーダル検索を実行
        
        Args:
            query: 検索クエリ
            top_k: 取得する結果数
            volume_filter: Volumeフィルタ（オプション）
            chapter_filter: Chapterフィルタ（オプション）
            alpha_text: テキストスコアの重み
            alpha_triple: Tripleスコアの重み
            alpha_image: 画像スコアの重み
        
        Returns:
            検索結果のリスト
        """
        # クエリ解析
        analysis = self.query_analyzer.analyze_query(query)
        print(f"\n📊 Query Analysis:")
        print(f"   Original: {analysis.original_query}")
        print(f"   Figure-related: {analysis.is_figure_related}")
        print(f"   Figure keywords: {analysis.figure_keywords}")
        print(f"   Needs image: {analysis.needs_image}")
        
        # クエリ埋め込み生成
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 1. テキスト埋め込みベースの検索（Qdrant）
        text_results = self._search_text_embeddings(
            query_embedding, 
            top_k=top_k * 2,  # 多めに取得してリランク
            volume_filter=volume_filter,
            chapter_filter=chapter_filter
        )
        
        # 2. Triple検索（オプション）
        triple_results = {}
        if self.triple_index and alpha_triple > 0:
            triple_results = self._search_triples(query_embedding, top_k=top_k * 2)
        
        # 3. スコア融合
        fused_results = self._fuse_scores(
            text_results=text_results,
            triple_results=triple_results,
            alpha_text=alpha_text,
            alpha_triple=alpha_triple,
            alpha_image=alpha_image,
            is_figure_query=analysis.is_figure_related
        )
        
        # 4. リランキング（Top-k選択）
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results[:top_k]
    
    def retrieve_with_images(self,
                             query: str,
                             top_k: int = 5,
                             use_image_understanding: bool = True) -> List[Tuple[RetrievalResult, str]]:
        """
        画像理解を統合した検索
        
        Args:
            query: 検索クエリ
            top_k: 取得する結果数
            use_image_understanding: 画像理解LLMを使用するか
        
        Returns:
            (検索結果, 画像理解による回答) のタプルリスト
        """
        # 通常の検索
        results = self.retrieve(query, top_k=top_k)
        
        # 画像理解が無効、またはクライアントがない場合
        if not use_image_understanding or self.multimodal_client is None:
            return [(r, "") for r in results]
        
        # Top結果に対して画像理解を適用
        enhanced_results = []
        for result in results:
            image_answer = ""
            
            # 図表ブロックで画像パスがある場合
            if result.block_type in ['table', 'figure'] and result.image_path:
                image_path = Path(result.image_path)
                if image_path.exists():
                    print(f"🔍 Understanding image: {result.block_id}")
                    image_answer = self.multimodal_client.understand_image_with_query(
                        image_path=image_path,
                        query=query,
                        caption=result.content
                    )
            
            enhanced_results.append((result, image_answer))
        
        return enhanced_results
    
    def generate_answer_multimodal(self,
                                    query: str,
                                    top_k: int = 3) -> str:
        """
        マルチモーダル検索結果から最終回答を生成
        
        Args:
            query: ユーザークエリ
            top_k: 参照する検索結果数
        
        Returns:
            生成された回答
        """
        # 画像理解統合検索
        results_with_images = self.retrieve_with_images(query, top_k=top_k)
        
        # コンテキスト構築
        context_parts = []
        for i, (result, image_answer) in enumerate(results_with_images, 1):
            context_parts.append(f"【参照{i}】 {result.block_id} ({result.block_type})")
            context_parts.append(f"内容: {result.content[:200]}")
            if image_answer:
                context_parts.append(f"画像解析: {image_answer[:300]}")
            context_parts.append("")
        
        context_text = "\n".join(context_parts)
        
        # 最終回答生成
        if self.multimodal_client:
            final_prompt = f"""以下の検索結果を参照して、質問に回答してください。

【質問】
{query}

【検索結果】
{context_text}

【回答】
検索結果の情報を統合し、質問に対する明確で具体的な回答を日本語で提供してください。"""
            
            try:
                response = requests.post(
                    f"{self.multimodal_client.base_url}/api/generate",
                    json={
                        "model": self.multimodal_client.model_name,
                        "prompt": final_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 512
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()
                    return answer if answer else "[回答を生成できませんでした]"
            except Exception as e:
                return f"[回答生成エラー: {str(e)[:50]}]"
        
        # フォールバック: 検索結果をそのまま返す
        return context_text
    
    def _search_text_embeddings(self, 
                                 query_embedding: np.ndarray,
                                 top_k: int = 20,
                                 volume_filter: Optional[str] = None,
                                 chapter_filter: Optional[str] = None) -> List[RetrievalResult]:
        """テキスト埋め込みベースの検索"""
        if self.block_embeddings is None or len(self.blocks) == 0:
            print(f"⚠️ No blocks loaded")
            return []
        
        # コサイン類似度計算（Inner Product、正規化済み）
        similarities = np.dot(self.block_embeddings, query_embedding)
        
        # フィルタリング
        valid_indices = []
        for i, block in enumerate(self.blocks):
            # Volumeフィルタ
            if volume_filter and block['volume'] != volume_filter:
                continue
            # Chapterフィルタ
            if chapter_filter and block['chapter'] != chapter_filter:
                continue
            valid_indices.append(i)
        
        # 有効なインデックスのスコアを取得
        if len(valid_indices) == 0:
            return []
        
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Top-k取得
        results = []
        for idx, score in valid_similarities[:top_k]:
            block = self.blocks[idx]
            result = RetrievalResult(
                block_id=block['block_id'],
                block_type=block['block_type'],
                content=block['content'],
                score=float(score),
                volume=block['volume'],
                chapter=block['chapter'],
                page_num=block['page_num'],
                bbox=block['bbox'],
                image_path=block['image_path']
            )
            results.append(result)
        
        return results
    
    def _search_triples(self, 
                        query_embedding: np.ndarray,
                        top_k: int = 20) -> Dict[str, float]:
        """Triple検索（FAISS）"""
        if self.triple_index is None:
            return {}
        
        # FAISS検索
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.triple_index.search(query_embedding, top_k)
        
        # ブロックIDごとにスコアを集計
        block_scores = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.triple_metadata):
                triple = self.triple_metadata[idx]
                block_id = triple.get('source_block_id', '')
                score = float(dist)  # Inner Product score
                
                # 同じブロックの複数tripleは最大スコアを使用
                if block_id not in block_scores or score > block_scores[block_id]:
                    block_scores[block_id] = score
        
        return block_scores
    
    def _fuse_scores(self,
                     text_results: List[RetrievalResult],
                     triple_results: Dict[str, float],
                     alpha_text: float,
                     alpha_triple: float,
                     alpha_image: float,
                     is_figure_query: bool) -> List[RetrievalResult]:
        """スコア融合"""
        # テキスト結果をベースに、Tripleスコアを追加
        for result in text_results:
            # テキストスコア
            result.text_score = result.score
            
            # Tripleスコア
            if result.block_id in triple_results:
                result.triple_score = triple_results[result.block_id]
            
            # 図表関連クエリの場合、図表ブロックのスコアをブースト
            if is_figure_query and result.block_type in ['table', 'figure']:
                boost = 1.2
            else:
                boost = 1.0
            
            # 融合スコア計算
            result.score = (
                alpha_text * result.text_score +
                alpha_triple * result.triple_score +
                alpha_image * result.image_score  # 画像スコアは将来実装
            ) * boost
        
        return text_results


def demo_retrieval():
    """デモ検索"""
    print("="*60)
    print("Multi-modal HippoRAG2 Retriever Demo")
    print("="*60)
    
    # Retriever初期化
    blocks_path = Path("experiments_v070/indices/layout_blocks_captioned.jsonl")
    triple_index_path = Path("experiments_v070/indices/mm_triple.index")
    triple_metadata_path = Path("experiments_v070/indices/mm_triples_metadata.json")
    
    retriever = MultimodalHippoRAG2Retriever(
        blocks_path=blocks_path,
        triple_index_path=triple_index_path,
        triple_metadata_path=triple_metadata_path
    )
    
    # デモクエリ
    queries = [
        "台風12号の被害状況を教えてください",
        "表で示された被害額はいくらですか",
        "全壊家屋の数は何棟ですか"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = retriever.retrieve(query, top_k=5)
        
        print(f"\n📊 Top-{len(results)} Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Block ID: {result.block_id}")
            print(f"   Type: {result.block_type}")
            print(f"   Score: {result.score:.4f} (text: {result.text_score:.4f}, triple: {result.triple_score:.4f})")
            print(f"   Content: {result.content[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Multimodal HippoRAG2 Retriever")
    parser.add_argument("--query", type=str, 
                       help="Search query")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of results to retrieve")
    parser.add_argument("--blocks", type=str,
                       default="experiments_v070/indices/layout_blocks_captioned.jsonl",
                       help="Path to captioned blocks JSONL")
    parser.add_argument("--triple-index", type=str,
                       default="experiments_v070/indices/mm_triple.index",
                       help="Path to triple FAISS index")
    parser.add_argument("--triple-metadata", type=str,
                       default="experiments_v070/indices/mm_triples_metadata.json",
                       help="Path to triple metadata")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo with sample queries")
    parser.add_argument("--enable-images", action="store_true",
                       help="Enable image understanding with Ollama")
    parser.add_argument("--generate-answer", action="store_true",
                       help="Generate final answer using multimodal context")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_retrieval()
    elif args.query:
        # Retriever初期化
        retriever = MultimodalHippoRAG2Retriever(
            blocks_path=Path(args.blocks),
            triple_index_path=Path(args.triple_index),
            triple_metadata_path=Path(args.triple_metadata),
            enable_image_understanding=args.enable_images
        )
        
        if args.generate_answer and args.enable_images:
            # マルチモーダル回答生成
            print(f"\n{'='*60}")
            print(f"Query: {args.query}")
            print(f"{'='*60}\n")
            
            print("🔄 Generating multimodal answer...")
            answer = retriever.generate_answer_multimodal(args.query, top_k=3)
            
            print(f"\n{'='*60}")
            print("📝 Generated Answer")
            print(f"{'='*60}")
            print(answer)
        else:
            # 通常の検索
            results = retriever.retrieve(args.query, top_k=args.top_k)
            
            # 結果表示
            print(f"\n📊 Retrieved {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.block_id}")
                print(f"   Type: {result.block_type} | Score: {result.score:.4f}")
                print(f"   Content: {result.content[:150]}...")
    else:
        print("Please provide --query or use --demo")
        sys.exit(1)


if __name__ == "__main__":
    main()
