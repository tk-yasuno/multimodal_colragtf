#!/usr/bin/env python3
# Copyright 2026 Multimodal CoLRAG-TF contributors
# Licensed under the Apache License, Version 2.0

"""
CoLRAG-TF v0.7.4 - Phase 10b: Optunaベイズ最適化

α_text・α_bm25・α_triple の3変数をベイズ最適化でキャリブレーション。
制約条件: α_text + α_bm25 + α_triple = 1.0

Usage:
    python experiments_v070/10b_optimize_fusion_weights.py --n-trials 50
    python experiments_v070/10b_optimize_fusion_weights.py --n-trials 50 --visualize
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pickle

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

# 評価指標のインポート
import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluation_metrics_v074 import (
    calc_precision_at_k,
    calc_diversity,
    calc_answer_quality,
    calc_combined_score
)


class MultimodalRetrieverForOptimization:
    """
    最適化用のMultimodalリトリーバー
    
    v0.7.3のMultimodalHippoRAG2Retrieverをベースに、
    重みをパラメータとして受け取れるように簡略化
    """
    
    def __init__(self,
                 blocks_path: Path,
                 triple_index_path: Path,
                 triple_metadata_path: Path,
                 bm25_index_path: Path,
                 embedding_model_name: str = "hotchpotch/static-embedding-japanese"):
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load blocks
        self.blocks = []
        with open(blocks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                page = json.loads(line)
                
                # Extract figure blocks
                for block in page.get("figure_blocks", []):
                    if "caption" in block:
                        self.blocks.append({
                            "block_id": block["block_id"],
                            "type": block["type"],
                            "caption": block["caption"]
                        })
                
                # Extract text blocks
                text_full = page.get("text_full", "").strip()
                if len(text_full) > 50:
                    self.blocks.append({
                        "block_id": f"{page['pdf_name']}_page{page['page_number']:03d}_text_full",
                        "type": "text",
                        "text": text_full[:1000]
                    })
        
        # Generate block embeddings
        texts = []
        for block in self.blocks:
            if block["type"] in ["table", "figure"]:
                texts.append(block.get("caption", ""))
            else:
                texts.append(block.get("text", ""))
        
        batch_size = 32
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_emb = self.embedding_model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_emb)
        
        self.block_embeddings = np.vstack(embeddings_list).astype("float32")
        
        # Load triple index
        self.triple_index = faiss.read_index(str(triple_index_path))
        with open(triple_metadata_path, "r", encoding="utf-8") as f:
            self.triple_metadata = json.load(f)
        
        # Load BM25 index
        with open(bm25_index_path, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25_index = bm25_data["bm25"]
            self.bm25_corpus_ids = bm25_data["corpus_ids"]
    
    @staticmethod
    def _bm25_tokenize(text: str) -> List[str]:
        """Japanese bigram tokenization"""
        chars = list(text)
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        return chars + bigrams
    
    def _search_bm25(self, query: str, top_k: int = 50) -> Dict[str, float]:
        """BM25 keyword search"""
        tokens = self._bm25_tokenize(query)
        bm25_scores = self.bm25_index.get_scores(tokens)
        
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_norm = bm25_scores / bm25_max
        else:
            bm25_norm = bm25_scores
        
        top_indices = np.argsort(-bm25_norm)[:top_k]
        
        results = {}
        for idx in top_indices:
            if idx < len(self.bm25_corpus_ids):
                block_id = self.bm25_corpus_ids[idx]
                results[block_id] = float(bm25_norm[idx])
        
        return results
    
    def retrieve(self,
                 query: str,
                 alpha_text: float,
                 alpha_bm25: float,
                 alpha_triple: float,
                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve with specified fusion weights
        """
        # Encode query
        query_emb = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        
        # 1. Text search
        text_scores = np.dot(self.block_embeddings, query_emb.T).flatten()
        
        # 2. BM25 search
        bm25_results = self._search_bm25(query, top_k=top_k*2)
        bm25_scores = np.zeros(len(self.blocks))
        for i, block in enumerate(self.blocks):
            bm25_scores[i] = bm25_results.get(block["block_id"], 0.0)
        
        # 3. Triple search
        triple_scores = np.zeros(len(self.blocks))
        if self.triple_index.ntotal > 0:
            D, I = self.triple_index.search(query_emb, k=20)
            
            for idx, score in zip(I[0], D[0]):
                if idx < len(self.triple_metadata):
                    block_id = self.triple_metadata[idx]["source_block_id"]
                    
                    for i, block in enumerate(self.blocks):
                        if block["block_id"] == block_id:
                            triple_scores[i] = max(triple_scores[i], score)
        
        # 4. Score fusion
        final_scores = (
            alpha_text * text_scores +
            alpha_bm25 * bm25_scores +
            alpha_triple * triple_scores
        )
        
        # Get top-k
        top_indices = np.argsort(final_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            block = self.blocks[idx]
            results.append({
                "block_id": block["block_id"],
                "type": block["type"],
                "score": float(final_scores[idx]),
                "text_score": float(text_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "triple_score": float(triple_scores[idx]),
                "image_score": 0.0,
                "content": block.get("caption", block.get("text", ""))[:500]
            })
        
        return results


def load_gold_standard(gold_path: Path) -> Dict[str, List[str]]:
    """
    Load gold standard annotations
    
    Returns:
        {image_id: [relevant_block_ids]}
    """
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    
    gold_standard = {}
    for image_info in gold_data["images"]:
        image_id = image_info["image_id"]
        gold_standard[image_id] = image_info["relevant_block_ids"]
    
    return gold_standard


def load_demo_results(indices_dir: Path) -> Dict[str, Dict]:
    """
    Load v0.7.3 demo results
    
    Returns:
        {image_id: demo_result_dict}
    """
    demo_results = {}
    
    for i in range(1, 13):
        image_id = f"{i:02d}"
        
        # Find corresponding JSON file
        json_files = list(indices_dir.glob(f"demo_result_v073_{image_id}_*.json"))
        
        if json_files:
            with open(json_files[0], "r", encoding="utf-8") as f:
                demo_results[image_id] = json.load(f)
    
    return demo_results


def create_objective_function(retriever: MultimodalRetrieverForOptimization,
                               gold_standard: Dict[str, List[str]],
                               demo_results: Dict[str, Dict],
                               enable_answer_quality: bool = False):
    """
    Create Optuna objective function
    """
    
    def objective(trial: optuna.Trial) -> float:
        # Sample weights
        alpha_text = trial.suggest_float("alpha_text", 0.2, 0.6)
        alpha_bm25 = trial.suggest_float("alpha_bm25", 0.1, 0.4)
        alpha_triple = trial.suggest_float("alpha_triple", 0.2, 0.6)
        
        # Normalize to sum=1.0
        total = alpha_text + alpha_bm25 + alpha_triple
        alpha_text /= total
        alpha_bm25 /= total
        alpha_triple /= total
        
        # Evaluate on all images
        precisions = []
        diversities = []
        answer_qualities = []
        
        for image_id, demo_result in demo_results.items():
            # Get query from image description
            image_desc = demo_result.get("image_description", "")
            query = f"災害の教訓 {image_desc[:200]}"
            
            # Retrieve with trial weights
            results = retriever.retrieve(
                query=query,
                alpha_text=alpha_text,
                alpha_bm25=alpha_bm25,
                alpha_triple=alpha_triple,
                top_k=5
            )
            
            # Extract block IDs
            retrieved_ids = [r["block_id"] for r in results]
            gold_ids = gold_standard.get(image_id, [])
            
            # Calculate Precision@5
            precision = calc_precision_at_k(retrieved_ids, gold_ids, k=5)
            precisions.append(precision)
            
            # Calculate Diversity
            diversity = calc_diversity(results)
            diversities.append(diversity)
            
            # Calculate Answer Quality (optional, expensive)
            if enable_answer_quality:
                lessons = demo_result.get("lessons_learned", "")
                answer_quality = calc_answer_quality(image_desc, lessons)
                answer_qualities.append(answer_quality)
        
        # Calculate average scores
        avg_precision = np.mean(precisions)
        avg_diversity = np.mean(diversities)
        avg_answer_quality = np.mean(answer_qualities) if answer_qualities else 0.5
        
        # Combined score
        if enable_answer_quality:
            score = calc_combined_score(
                precision=avg_precision,
                diversity=avg_diversity,
                answer_quality=avg_answer_quality,
                w_precision=0.5,
                w_diversity=0.2,
                w_quality=0.3
            )
        else:
            # Without answer quality (faster)
            score = 0.7 * avg_precision + 0.3 * avg_diversity
        
        # Log intermediate results
        trial.set_user_attr("avg_precision", avg_precision)
        trial.set_user_attr("avg_diversity", avg_diversity)
        trial.set_user_attr("avg_answer_quality", avg_answer_quality)
        
        return score
    
    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna optimization for fusion weights")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--enable-answer-quality", action="store_true", help="Enable answer quality evaluation (slower)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization after optimization")
    parser.add_argument("--study-name", type=str, default="fusion_weights_v074", help="Optuna study name")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Optuna Bayesian Optimization for Fusion Weights (v0.7.4)")
    print("=" * 70)
    print()
    
    # Paths
    base_dir = Path(__file__).parent
    indices_dir = base_dir / "indices"
    
    blocks_path = indices_dir / "layout_blocks_captioned.jsonl"
    triple_index_path = indices_dir / "mm_triple.index"
    triple_metadata_path = indices_dir / "mm_triples_metadata.json"
    bm25_index_path = indices_dir / "bm25_index.pkl"
    gold_path = indices_dir / "gold_standard_image_retrieval.json"
    
    # Initialize retriever
    print("🔄 Initializing retriever...")
    retriever = MultimodalRetrieverForOptimization(
        blocks_path=blocks_path,
        triple_index_path=triple_index_path,
        triple_metadata_path=triple_metadata_path,
        bm25_index_path=bm25_index_path
    )
    print(f"✅ Retriever initialized | Blocks: {len(retriever.blocks)}")
    print()
    
    # Load gold standard and demo results
    print("🔄 Loading gold standard and demo results...")
    gold_standard = load_gold_standard(gold_path)
    demo_results = load_demo_results(indices_dir)
    print(f"✅ Loaded {len(gold_standard)} images with gold standard")
    print(f"✅ Loaded {len(demo_results)} demo results")
    print()
    
    # Create objective function
    objective = create_objective_function(
        retriever=retriever,
        gold_standard=gold_standard,
        demo_results=demo_results,
        enable_answer_quality=args.enable_answer_quality
    )
    
    # Run optimization
    print(f"🔍 Starting Bayesian optimization ({args.n_trials} trials)...")
    print(f"   Answer Quality: {'Enabled (slower)' if args.enable_answer_quality else 'Disabled (faster)'}")
    print()
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Results
    print()
    print("=" * 70)
    print("Optimization Results")
    print("=" * 70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print()
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    print()
    
    # Normalize best params
    best_text = study.best_params["alpha_text"]
    best_bm25 = study.best_params["alpha_bm25"]
    best_triple = study.best_params["alpha_triple"]
    total = best_text + best_bm25 + best_triple
    
    print("Normalized best parameters:")
    print(f"  alpha_text: {best_text/total:.4f}")
    print(f"  alpha_bm25: {best_bm25/total:.4f}")
    print(f"  alpha_triple: {best_triple/total:.4f}")
    print()
    
    # User attributes
    best_trial = study.best_trial
    print("Best trial metrics:")
    print(f"  Avg Precision@5: {best_trial.user_attrs.get('avg_precision', 0):.4f}")
    print(f"  Avg Diversity: {best_trial.user_attrs.get('avg_diversity', 0):.4f}")
    print(f"  Avg Answer Quality: {best_trial.user_attrs.get('avg_answer_quality', 0):.4f}")
    print()
    
    # Save results
    results = {
        "version": "v0.7.4",
        "optimization_method": "Optuna TPESampler",
        "n_trials": args.n_trials,
        "best_score": study.best_value,
        "best_params": {
            "alpha_text": best_text / total,
            "alpha_bm25": best_bm25 / total,
            "alpha_triple": best_triple / total,
            "alpha_image": 0.0
        },
        "best_trial_number": study.best_trial.number,
        "metrics": {
            "avg_precision_at_5": best_trial.user_attrs.get("avg_precision", 0),
            "avg_diversity": best_trial.user_attrs.get("avg_diversity", 0),
            "avg_answer_quality": best_trial.user_attrs.get("avg_answer_quality", 0)
        }
    }
    
    output_path = indices_dir / "best_weights_v074.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    
    # Save study database
    db_path = indices_dir / "optuna_study_v074.db"
    study.trials_dataframe().to_csv(indices_dir / "optuna_trials_v074.csv", index=False)
    print(f"✅ Study data saved to: {indices_dir / 'optuna_trials_v074.csv'}")
    
    # Visualization
    if args.visualize:
        print()
        print("🎨 Generating visualizations...")
        
        try:
            import plotly.io as pio
            
            # Optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_html(indices_dir / "optuna_history_v074.html")
            
            # Parameter importances
            fig2 = plot_param_importances(study)
            fig2.write_html(indices_dir / "optuna_importances_v074.html")
            
            # Parallel coordinate
            fig3 = plot_parallel_coordinate(study)
            fig3.write_html(indices_dir / "optuna_parallel_v074.html")
            
            print(f"✅ Visualizations saved to: {indices_dir}")
        except ImportError:
            print("⚠️  Plotly not installed. Skipping visualizations.")
    
    print()
    print("=" * 70)
    print("✅ Optimization completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
