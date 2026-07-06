#!/usr/bin/env python3
# Copyright 2026 Multimodal CoLRAG-TF contributors
# Licensed under the Apache License, Version 2.0

"""
Integrated Demo: Image Understanding + Multimodal Retriever

Demonstrates the full pipeline:
1. Input: Disaster image
2. Query: "この画像に移っている災害の教訓を整理したい。"
3. Image Understanding: llava:7b extracts disaster information
4. Retrieval: MultimodalHippoRAG2Retriever finds relevant blocks
5. Answer Generation: qwen2.5:7b synthesizes lessons learned
"""

import sys
import json
import base64
import requests
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class QueryAnalysis:
    """Query analysis result"""
    original_query: str
    is_figure_related: bool
    figure_keywords: List[str]
    needs_image: bool


class OllamaMultimodalClient:
    """Client for Ollama multimodal LLM"""
    
    def __init__(self, text_model: str = "qwen2.5:7b-instruct-q4_k_m", 
                 vision_model: str = "llava:7b",
                 base_url: str = "http://localhost:11434"):
        self.text_model = text_model
        self.vision_model = vision_model
        self.base_url = base_url
        print(f"✅ OllamaMultimodalClient initialized:")
        print(f"   Text model: {text_model}")
        print(f"   Vision model: {vision_model}")
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def understand_image(self, image_path: Path, query: str, max_tokens: int = 500) -> str:
        """
        Understand image content with vision model
        """
        try:
            image_b64 = self.encode_image(image_path)
            
            payload = {
                "model": self.vision_model,
                "prompt": query,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_answer(self, query: str, context: str, max_tokens: int = 500) -> str:
        """
        Generate answer using text model
        """
        try:
            prompt = f"""以下の情報に基づいて、質問に答えてください。

【参考情報】
{context}

【質問】
{query}

【回答】
"""
            
            payload = {
                "model": self.text_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.5
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"


class MultimodalHippoRAG2Retriever:
    """Multimodal HippoRAG2-style retriever with triple search"""
    
    def __init__(self, 
                 embedding_model_name: str = "hotchpotch/static-embedding-japanese",
                 alpha_text: float = 0.6,
                 alpha_triple: float = 0.4,
                 figure_boost: float = 1.2):
        self.alpha_text = alpha_text
        self.alpha_triple = alpha_triple
        self.figure_boost = figure_boost
        
        # Load embedding model
        print(f"🔄 Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded | Dimension: {self.embedding_dim} | Device: {self.embedding_model.device}")
        
        self.blocks = []
        self.block_embeddings = None
        self.triple_index = None
        self.triple_metadata = []
    
    def _load_blocks(self, blocks_path: Path):
        """Load captioned blocks"""
        print(f"🔄 Loading blocks from: {blocks_path}")
        
        with open(blocks_path, "r", encoding="utf-8") as f:
            for line in f:
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
                        "text": text_full[:1000]  # Limit length
                    })
        
        print(f"✅ Loaded {len(self.blocks)} blocks")
    
    def _generate_block_embeddings(self):
        """Generate embeddings for all blocks"""
        print(f"🔄 Generating block embeddings...")
        
        texts = []
        for block in self.blocks:
            if block["type"] in ["table", "figure"]:
                texts.append(block.get("caption", ""))
            else:
                texts.append(block.get("text", ""))
        
        # Batch encoding
        batch_size = 32
        embeddings_list = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
            batch = texts[i:i+batch_size]
            batch_emb = self.embedding_model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_emb)
        
        self.block_embeddings = np.vstack(embeddings_list).astype("float32")
        print(f"✅ Block embeddings ready: shape {self.block_embeddings.shape}")
    
    def _load_triple_index(self, index_path: Path, metadata_path: Path):
        """Load FAISS triple index"""
        print(f"🔄 Loading triple index: {index_path}")
        self.triple_index = faiss.read_index(str(index_path))
        print(f"✅ Triple index loaded | Total: {self.triple_index.ntotal}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.triple_metadata = json.load(f)
        print(f"✅ Triple metadata loaded | Count: {len(self.triple_metadata)}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant blocks with HippoRAG2-style fusion
        """
        # Encode query
        query_emb = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        
        # Text search (Inner Product)
        text_scores = np.dot(self.block_embeddings, query_emb.T).flatten()
        
        # Triple search
        triple_scores = np.zeros(len(self.blocks))
        if self.triple_index and self.triple_index.ntotal > 0:
            D, I = self.triple_index.search(query_emb, k=20)
            
            # Map triple scores to blocks
            for idx, score in zip(I[0], D[0]):
                if idx < len(self.triple_metadata):
                    block_id = self.triple_metadata[idx]["source_block_id"]
                    
                    for i, block in enumerate(self.blocks):
                        if block["block_id"] == block_id:
                            triple_scores[i] = max(triple_scores[i], score)
        
        # Score fusion
        final_scores = self.alpha_text * text_scores + self.alpha_triple * triple_scores
        
        # Figure boost
        for i, block in enumerate(self.blocks):
            if block["type"] in ["table", "figure"]:
                final_scores[i] *= self.figure_boost
        
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
                "triple_score": float(triple_scores[idx]),
                "content": block.get("caption", block.get("text", ""))[:500]
            })
        
        return results


def run_integrated_demo(image_path: Path):
    """
    Run integrated demo: Image Understanding + Retrieval + Answer Generation
    """
    print("=" * 70)
    print("Integrated Demo: Image-based Disaster Lesson Extraction")
    print("=" * 70)
    print()
    
    # Initialize components
    llm_client = OllamaMultimodalClient()
    retriever = MultimodalHippoRAG2Retriever()
    
    # Load data
    data_dir = Path(__file__).parent / "indices"
    retriever._load_blocks(data_dir / "layout_blocks_captioned.jsonl")
    retriever._generate_block_embeddings()
    retriever._load_triple_index(
        data_dir / "mm_triple.index",
        data_dir / "mm_triples_metadata.json"
    )
    
    print()
    print("=" * 70)
    print(f"Input Image: {image_path.name}")
    print("=" * 70)
    print()
    
    # Step 1: Image Understanding
    print("🔍 Step 1: Image Understanding with Vision LLM")
    print("-" * 70)
    
    understanding_query = "この画像に写っている災害の種類、被害状況、特徴を詳しく説明してください。"
    print(f"Query: {understanding_query}\n")
    
    image_description = llm_client.understand_image(image_path, understanding_query, max_tokens=300)
    print(f"💬 Image Description:\n{image_description}\n")
    
    # Step 2: Retrieval based on image understanding
    print("🔍 Step 2: Retrieving Relevant Blocks")
    print("-" * 70)
    
    retrieval_query = f"災害の教訓 {image_description[:200]}"
    print(f"Query: {retrieval_query[:100]}...\n")
    
    results = retriever.retrieve(retrieval_query, top_k=5)
    
    print(f"📊 Top-{len(results)} Results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Block ID: {result['block_id']}")
        print(f"   Type: {result['type']}")
        print(f"   Score: {result['score']:.4f} (text: {result['text_score']:.4f}, triple: {result['triple_score']:.4f})")
        print(f"   Content: {result['content'][:150]}...\n")
    
    # Step 3: Generate lessons learned
    print("🔍 Step 3: Generating Disaster Lessons")
    print("-" * 70)
    
    context = "\n\n".join([
        f"【参考資料{i+1}】\n{r['content']}"
        for i, r in enumerate(results)
    ])
    
    lesson_query = f"""画像に写っている災害について、以下の観点から教訓を整理してください：

【画像から読み取った情報】
{image_description}

【整理すべき教訓の観点】
1. 災害の特徴と被害の実態
2. 事前の備えとして必要なこと
3. 災害発生時の対応策
4. 復旧・復興における重要事項

上記の観点から、具体的な教訓を整理してください。
"""
    
    print(f"Query: {lesson_query[:150]}...\n")
    
    lessons = llm_client.generate_answer(lesson_query, context, max_tokens=600)
    
    print("=" * 70)
    print("📋 Disaster Lessons Learned")
    print("=" * 70)
    print(lessons)
    print()
    
    # Save results
    output = {
        "image_path": str(image_path),
        "image_description": image_description,
        "retrieval_results": results,
        "lessons_learned": lessons
    }
    
    output_path = Path(__file__).parent / "indices" / f"demo_result_{image_path.stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("=" * 70)
    print(f"✅ Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    # Test with sample images
    sample_images_dir = Path(__file__).parent.parent / "data" / "sample_images_n12"
    
    # Demo with all 12 images for diversity and robustness evaluation
    test_images = [
        "01_令和２年７月豪雨被害_熊本写真.jpg",
        "02_宮城県津波災害_フリー写真素材.jpg",
        "03_熊本地震_フリー画像_日本防災士会.jpg",
        "04_建設システム_総合防災アプリ_大雨防災.jpg",
        "05_阪神淡路大震災から25年_文春オンライン.jpg",
        "06_土砂災害フリー画像.jpg",
        "07_熱海土砂崩れ_建設システム_綜合防災アプリ.jpeg",
        "08_熱海土石流災害_静岡新聞.jpg",
        "09_箱根山噴火_Rescue4th.jpg",
        "10_ベネズエラ首都土砂災害_AFPBB_News.jpg",
        "11_2022年10月ベネズエラ、北部アラグア州郊外_Matias_AP通信.jpg",
        "12_ベネズエラM7地震1分揺れ-venezuela-quake_CNN.jpg"
    ]
    
    for image_name in test_images:
        image_path = sample_images_dir / image_name
        
        if not image_path.exists():
            print(f"⚠ Image not found: {image_path}")
            continue
        
        run_integrated_demo(image_path)
        print("\n\n")
