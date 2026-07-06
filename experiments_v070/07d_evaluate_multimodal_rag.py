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
CoLRAG-TF v0.7.0 - Phase 7d: マルチモーダルRAG評価

500QA（1-hop 200Q + multi-hop 200Q + cause-mitigation 100Q）を使用して
マルチモーダルRAGの有用性を評価します。

比較軸:
1. 全体500Q: マルチモーダルRAGの総合性能
2. 1-hop 200Q: シンプルな質問への回答精度
3. multi-hop 300Q: 複雑な推論への対応力（1-hopとの差分）

Usage:
    .venv-coltf\\Scripts\\python.exe experiments_v070\\07d_evaluate_multimodal_rag.py --evaluate-all
"""

import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import time
import numpy as np

try:
    import requests
    from sentence_transformers import SentenceTransformer
    import faiss
    print("✅ All dependencies imported")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


@dataclass
class EvaluationResult:
    """評価結果"""
    question_id: str
    question: str
    question_type: str  # "simple", "disaster_comparison", "phase_transition", etc.
    hop_count: int
    
    predicted_answer: str
    ground_truth: str
    
    retrieved_blocks: List[str] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    
    # 評価スコア
    answer_similarity: float = 0.0  # 回答の類似度
    retrieval_recall: float = 0.0  # 必要ブロックの検索率
    answer_length: int = 0
    
    # エラーフラグ
    retrieval_failed: bool = False
    generation_failed: bool = False


class MultimodalRetriever:
    """マルチモーダルリトリーバー（簡易版 - v0.7.4: Bayesian-optimized weights）"""
    
    def __init__(self, 
                 triple_index_path: Path,
                 bm25_index_path: Optional[Path] = None,
                 embedding_model_name: str = "hotchpotch/static-embedding-japanese",
                 alpha_triple: float = 0.604,  # v0.7.4: Bayesian-optimized (0.4422/0.7325)
                 alpha_bm25: float = 0.396):   # v0.7.4: Bayesian-optimized (0.2903/0.7325)
        """
        Args:
            triple_index_path: FAISS triple indexのパス
            bm25_index_path: BM25 indexのパス
            embedding_model_name: 埋め込みモデル名
            alpha_triple: Triple軸の重み (default: 0.604, v0.7.4最適化値)
            alpha_bm25: BM25軸の重み (default: 0.396, v0.7.4最適化値)
        """
        self.alpha_triple = alpha_triple
        self.alpha_bm25 = alpha_bm25
        print(f"   Fusion weights: alpha_triple={alpha_triple:.3f}, alpha_bm25={alpha_bm25:.3f}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Triple indexを読み込み
        print(f"   Loading triple index from {triple_index_path}...")
        self.triple_index = faiss.read_index(str(triple_index_path))
        
        # Triple metadataを読み込み
        metadata_path = triple_index_path.parent / "mm_triples_metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.triple_metadata = json.load(f)
        
        print(f"   ✅ Loaded {self.triple_index.ntotal} triples")
        
        # BM25 indexを読み込み (v0.7.3)
        self.bm25_index = None
        self.bm25_corpus_ids = []
        if bm25_index_path and bm25_index_path.exists():
            print(f"   Loading BM25 index from {bm25_index_path}...")
            with open(bm25_index_path, 'rb') as f:
                bm25_data = pickle.load(f)
                self.bm25_index = bm25_data['bm25']
                self.bm25_corpus_ids = bm25_data['corpus_ids']
            print(f"   ✅ Loaded BM25 index with {len(self.bm25_corpus_ids)} blocks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        クエリから関連ブロックを検索 (v0.7.3: BM25融合)
        
        Returns:
            List[Dict]: 検索結果 [{"block_id": ..., "score": ..., "content": ...}, ...]
        """
        # クエリを埋め込み
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # 1. FAISS検索 (Triple埋め込み)
        scores, indices = self.triple_index.search(query_embedding.astype('float32'), top_k * 2)
        
        # Tripleスコアをblock_idごとに集約
        triple_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.triple_metadata):
                metadata = self.triple_metadata[idx]
                block_id = metadata.get("source_block_id", "unknown")
                if block_id not in triple_scores or score > triple_scores[block_id]:
                    triple_scores[block_id] = float(score)
        
        # 2. BM25検索 (v0.7.3)
        bm25_scores = {}
        if self.bm25_index:
            tokens = self._bm25_tokenize(query)
            bm25_raw = self.bm25_index.get_scores(tokens)
            bm25_max = bm25_raw.max()
            if bm25_max > 0:
                bm25_norm = bm25_raw / bm25_max
            else:
                bm25_norm = bm25_raw
            
            # Top candidatesを取得
            top_indices = np.argsort(-bm25_norm)[:top_k * 2]
            for idx in top_indices:
                if idx < len(self.bm25_corpus_ids):
                    block_id = self.bm25_corpus_ids[idx]
                    bm25_scores[block_id] = float(bm25_norm[idx])
        
        # 3. スコア融合 (v0.7.4: Bayesian-optimized weights)
        fused_scores = {}
        all_block_ids = set(triple_scores.keys()) | set(bm25_scores.keys())
        
        for block_id in all_block_ids:
            triple_score = triple_scores.get(block_id, 0.0)
            bm25_score = bm25_scores.get(block_id, 0.0)
            
            # 融合スコア (v0.7.4: alpha_triple=0.604, alpha_bm25=0.396)
            if self.bm25_index:
                fused_scores[block_id] = self.alpha_triple * triple_score + self.alpha_bm25 * bm25_score
            else:
                fused_scores[block_id] = triple_score
        
        # 4. Top-k選択
        sorted_blocks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 結果を整形
        results = []
        for block_id, score in sorted_blocks:
            # metadataからtriple情報を取得
            triple_info = ""
            for metadata in self.triple_metadata:
                if metadata.get("source_block_id") == block_id:
                    triple_info = metadata.get("triple", "")
                    break
            
            results.append({
                "block_id": block_id,
                "score": float(score),
                "triple": triple_info,
                "content": triple_info
            })
        
        return results
    
    @staticmethod
    def _bm25_tokenize(text: str) -> List[str]:
        """日本語バイグラムトークン化 (v0.7.3)"""
        chars = list(text)
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        return chars + bigrams


class OllamaAnswerGenerator:
    """Ollamaを使用した回答生成器"""
    
    ANSWER_PROMPT = """以下の質問に対して、検索された災害教訓文書の情報を活用し、詳細かつ具体的な回答を生成してください。

【質問】
{question}

【参考情報（災害教訓文書からの抽出）】
{context}

【回答作成の指示】
1. 参考情報に含まれる具体的な数値、事例、地名、時期などを必ず引用してください
2. 複数の参考情報がある場合は、それらを統合して包括的に説明してください
3. 災害の要因、影響、対策について段階的に説明してください
4. 回答は200-350文字程度で、具体的かつ詳細に記述してください
5. 参考情報に含まれない内容は推測せず、根拠のある情報のみを記載してください

回答（200-350文字で具体的に記述）:"""

    def __init__(self, model_name: str = "qwen2.5:7b-instruct-q4_k_m",
                 base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        print(f"✅ OllamaAnswerGenerator initialized: {model_name}")
    
    def generate_answer(self, question: str, context_blocks: List[Dict[str, Any]]) -> str:
        """
        質問とコンテキストから回答を生成
        
        Args:
            question: 質問
            context_blocks: 検索されたブロック
        
        Returns:
            生成された回答
        """
        # コンテキストを結合（より多くの情報を含める）
        context_parts = []
        for i, block in enumerate(context_blocks[:5], 1):  # Top-5のみ使用
            content = block.get('content', '')
            if content:
                # より長いコンテキストを提供（400文字まで）
                context_parts.append(f"【参考{i}】{content[:400]}")
        
        context = "\n\n".join(context_parts) if context_parts else "（参考情報が見つかりませんでした）"
        
        prompt = self.ANSWER_PROMPT.format(question=question, context=context)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 600,  # 400→600に増加
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=150  # 120→150に増加
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                # 回答が極端に短い場合の警告（デバッグ用）
                if len(answer) < 100:
                    print(f"\n⚠️ Short answer generated: {len(answer)} chars")
                
                return answer
        except Exception as e:
            print(f"\n⚠️ Answer generation error: {e}")
        
        return "回答の生成に失敗しました。"


class MultimodalRAGEvaluator:
    """マルチモーダルRAG評価器"""
    
    def __init__(self,
                 retriever: MultimodalRetriever,
                 generator: OllamaAnswerGenerator):
        self.retriever = retriever
        self.generator = generator
        self.embedding_model = retriever.embedding_model
    
    def evaluate_qa(self, qa: Dict[str, Any]) -> EvaluationResult:
        """1つのQAペアを評価"""
        
        question = qa.get('question', '')
        ground_truth = qa.get('answer', '')
        question_id = qa.get('question_id', '')
        question_type = qa.get('question_type', 'simple')
        hop_count = qa.get('hop_count', 1)
        
        result = EvaluationResult(
            question_id=question_id,
            question=question,
            question_type=question_type,
            hop_count=hop_count,
            predicted_answer="",
            ground_truth=ground_truth
        )
        
        # 1. Retrieval
        retrieval_start = time.time()
        try:
            retrieved_blocks = self.retriever.retrieve(question, top_k=5)
            result.retrieved_blocks = [b['block_id'] for b in retrieved_blocks]
            result.retrieval_time_ms = (time.time() - retrieval_start) * 1000
        except Exception as e:
            print(f"\n⚠️ Retrieval error for {question_id}: {e}")
            result.retrieval_failed = True
            return result
        
        # 2. Answer Generation
        generation_start = time.time()
        try:
            predicted_answer = self.generator.generate_answer(question, retrieved_blocks)
            result.predicted_answer = predicted_answer
            result.answer_length = len(predicted_answer)
            result.generation_time_ms = (time.time() - generation_start) * 1000
        except Exception as e:
            print(f"\n⚠️ Generation error for {question_id}: {e}")
            result.generation_failed = True
            return result
        
        # 3. Answer Similarity（埋め込みベース）
        try:
            pred_embedding = self.embedding_model.encode([predicted_answer], normalize_embeddings=True)
            gt_embedding = self.embedding_model.encode([ground_truth], normalize_embeddings=True)
            similarity = float(np.dot(pred_embedding[0], gt_embedding[0]))
            result.answer_similarity = max(0.0, min(1.0, similarity))  # [0, 1]にクリップ
        except:
            result.answer_similarity = 0.0
        
        # 4. Retrieval Recall（正解ブロックIDとの一致）
        expected_block_ids = qa.get('block_ids', [])
        if expected_block_ids:
            retrieved_set = set(result.retrieved_blocks)
            expected_set = set(expected_block_ids)
            recall = len(retrieved_set & expected_set) / len(expected_set)
            result.retrieval_recall = recall
        
        return result
    
    def evaluate_dataset(self, qa_list: List[Dict[str, Any]],
                         max_samples: Optional[int] = None) -> List[EvaluationResult]:
        """データセット全体を評価"""
        
        if max_samples:
            qa_list = qa_list[:max_samples]
        
        results = []
        for i, qa in enumerate(qa_list, 1):
            print(f"   Evaluating {i}/{len(qa_list)}... ", end='\r')
            result = self.evaluate_qa(qa)
            results.append(result)
            time.sleep(0.5)  # 0.3→0.5に増加（より丁寧な生成のため）
        
        print()  # 改行
        return results


def load_qa_datasets(qa_paths: Dict[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """複数のQAデータセットを読み込み"""
    datasets = {}
    
    for name, path in qa_paths.items():
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                datasets[name] = json.load(f)
            print(f"   ✅ Loaded {name}: {len(datasets[name])} QA pairs")
        else:
            print(f"   ⚠️ {name} not found: {path}")
            datasets[name] = []
    
    return datasets


def compute_statistics(results: List[EvaluationResult]) -> Dict[str, float]:
    """評価結果から統計を計算"""
    
    if not results:
        return {}
    
    # フィルタリング（失敗を除外）
    valid_results = [r for r in results if not r.retrieval_failed and not r.generation_failed]
    
    if not valid_results:
        return {"error": "No valid results"}
    
    stats = {
        "total_questions": len(results),
        "valid_questions": len(valid_results),
        "retrieval_failed": sum(1 for r in results if r.retrieval_failed),
        "generation_failed": sum(1 for r in results if r.generation_failed),
        
        "avg_answer_similarity": np.mean([r.answer_similarity for r in valid_results]),
        "avg_retrieval_recall": np.mean([r.retrieval_recall for r in valid_results if r.retrieval_recall > 0]),
        "avg_answer_length": np.mean([r.answer_length for r in valid_results]),
        "avg_retrieval_time_ms": np.mean([r.retrieval_time_ms for r in valid_results]),
        "avg_generation_time_ms": np.mean([r.generation_time_ms for r in valid_results]),
        
        "median_answer_similarity": np.median([r.answer_similarity for r in valid_results]),
        "std_answer_similarity": np.std([r.answer_similarity for r in valid_results]),
    }
    
    return stats


def print_statistics(stats: Dict[str, float], title: str):
    """統計を表示"""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    if "error" in stats:
        print(f"   ❌ {stats['error']}")
        return
    
    print(f"   Total Questions: {stats['total_questions']}")
    print(f"   Valid Questions: {stats['valid_questions']}")
    print(f"   Failed (Retrieval): {stats['retrieval_failed']}")
    print(f"   Failed (Generation): {stats['generation_failed']}")
    print(f"\n   📈 Performance Metrics:")
    print(f"      Answer Similarity: {stats['avg_answer_similarity']:.4f} (±{stats.get('std_answer_similarity', 0):.4f})")
    print(f"      Median Similarity: {stats.get('median_answer_similarity', 0):.4f}")
    print(f"      Retrieval Recall: {stats['avg_retrieval_recall']:.4f}")
    print(f"      Answer Length: {stats['avg_answer_length']:.1f} chars (target: 200-350)")
    
    # 改善度の評価
    avg_len = stats['avg_answer_length']
    if avg_len < 150:
        print(f"         ⚠️ Too short - need improvement")
    elif avg_len > 350:
        print(f"         ⚠️ Too long - consider trimming")
    else:
        print(f"         ✅ Good length")
    
    print(f"\n   ⏱️ Timing:")
    print(f"      Retrieval: {stats['avg_retrieval_time_ms']:.1f} ms")
    print(f"      Generation: {stats['avg_generation_time_ms']:.1f} ms")
    print(f"      Total: {stats['avg_retrieval_time_ms'] + stats['avg_generation_time_ms']:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG Evaluation")
    parser.add_argument("--evaluate-all", action="store_true",
                       help="Evaluate all QA datasets")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset (for quick test)")
    parser.add_argument("--triple-index", type=str,
                       default="experiments_v070/indices/mm_triple.index",
                       help="Path to triple FAISS index")
    parser.add_argument("--bm25-index", type=str,
                       default="experiments_v070/indices/bm25_index.pkl",
                       help="Path to BM25 index (v0.7.3)")
    parser.add_argument("--output", type=str,
                       default="experiments_v070/indices/eval_results_full.json",
                       help="Output path for evaluation results")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Multimodal RAG Evaluation")
    print(f"{'='*60}")
    
    # QAデータセットを読み込み（クリーン版を使用）
    qa_paths = {
        "1-hop": Path("experiments_v070/indices/qa_multimodal_clean.json"),
        "multi-hop": Path("experiments_v070/indices/qa_multihop_clean.json"),
        "cause-mitigation": Path("experiments_v070/indices/qa_cause_mitigation.json")
    }
    
    datasets = load_qa_datasets(qa_paths)
    
    # 統合データセット
    all_qa = []
    for name, qa_list in datasets.items():
        all_qa.extend(qa_list)
    
    print(f"\n   📦 Total QA pairs: {len(all_qa)}")
    
    # リトリーバーと生成器を初期化 (v0.7.3: BM25追加)
    print(f"\n🔄 Initializing retriever and generator...")
    retriever = MultimodalRetriever(
        triple_index_path=Path(args.triple_index),
        bm25_index_path=Path(args.bm25_index)  # v0.7.3
    )
    generator = OllamaAnswerGenerator()
    evaluator = MultimodalRAGEvaluator(retriever, generator)
    
    # 評価実行
    all_results = {}
    
    if args.evaluate_all:
        # 1. 全体評価（500Q）
        print(f"\n{'='*60}")
        print("🔍 Evaluating: ALL 500Q")
        print(f"{'='*60}")
        results_all = evaluator.evaluate_dataset(all_qa, max_samples=args.max_samples)
        stats_all = compute_statistics(results_all)
        print_statistics(stats_all, "All 500Q Results")
        all_results['all_500q'] = {
            'stats': stats_all,
            'results': [asdict(r) for r in results_all]
        }
        
        # 2. 1-hop評価（200Q）
        if datasets.get("1-hop"):
            print(f"\n{'='*60}")
            print("🔍 Evaluating: 1-hop 200Q")
            print(f"{'='*60}")
            results_1hop = evaluator.evaluate_dataset(datasets["1-hop"], max_samples=args.max_samples)
            stats_1hop = compute_statistics(results_1hop)
            print_statistics(stats_1hop, "1-hop 200Q Results")
            all_results['1hop_200q'] = {
                'stats': stats_1hop,
                'results': [asdict(r) for r in results_1hop]
            }
        
        # 3. Multi-hop評価（300Q = multi-hop 200Q + cause-mitigation 100Q）
        multihop_combined = datasets.get("multi-hop", []) + datasets.get("cause-mitigation", [])
        if multihop_combined:
            print(f"\n{'='*60}")
            print("🔍 Evaluating: Multi-hop 300Q (multihop + cause-mitigation)")
            print(f"{'='*60}")
            results_multihop = evaluator.evaluate_dataset(multihop_combined, max_samples=args.max_samples)
            stats_multihop = compute_statistics(results_multihop)
            print_statistics(stats_multihop, "Multi-hop 300Q Results")
            all_results['multihop_300q'] = {
                'stats': stats_multihop,
                'results': [asdict(r) for r in results_multihop]
            }
        
        # 4. 比較分析
        if 'stats' in all_results.get('1hop_200q', {}) and 'stats' in all_results.get('multihop_300q', {}):
            print(f"\n{'='*60}")
            print("📊 Comparison: 1-hop vs Multi-hop")
            print(f"{'='*60}")
            
            sim_1hop = all_results['1hop_200q']['stats']['avg_answer_similarity']
            sim_multihop = all_results['multihop_300q']['stats']['avg_answer_similarity']
            improvement = ((sim_multihop - sim_1hop) / sim_1hop) * 100
            
            len_1hop = all_results['1hop_200q']['stats']['avg_answer_length']
            len_multihop = all_results['multihop_300q']['stats']['avg_answer_length']
            len_diff = len_multihop - len_1hop
            
            print(f"   Answer Similarity:")
            print(f"      1-hop:      {sim_1hop:.4f}")
            print(f"      Multi-hop:  {sim_multihop:.4f}")
            print(f"      Improvement: {improvement:+.2f}%")
            
            print(f"\n   Answer Length:")
            print(f"      1-hop:      {len_1hop:.1f} chars")
            print(f"      Multi-hop:  {len_multihop:.1f} chars")
            print(f"      Difference: {len_diff:+.1f} chars")
            
            if improvement > 0:
                print(f"\n   ✅ Multi-hop questions show BETTER performance (+{improvement:.2f}%)")
            else:
                print(f"\n   ⚠️ Multi-hop questions show WORSE performance ({improvement:.2f}%)")
            
            if len_1hop >= 200 and len_multihop >= 200:
                print(f"   ✅ Both achieve target answer length (200-350 chars)")
            elif len_multihop >= 200:
                print(f"   ⚠️ Multi-hop achieves target, but 1-hop is too short")
            else:
                print(f"   ⚠️ Both need longer answers to reach target (200-350 chars)")
    
    # 結果を保存
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"💾 Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
