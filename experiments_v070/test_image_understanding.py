#!/usr/bin/env python3
# Copyright 2026 Multimodal CoLRAG-TF contributors
# Licensed under the Apache License, Version 2.0

"""
Test script for Phase 6c: Image Understanding with Multimodal LLM

Tests the image understanding capability using sample disaster images
with llava:7b model via Ollama.
"""

import sys
import json
import base64
import requests
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class OllamaMultimodalClient:
    """Client for Ollama multimodal LLM"""
    
    def __init__(self, model: str = "llava:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        print(f"✅ OllamaMultimodalClient initialized: {model}")
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def understand_image_with_query(self, image_path: Path, query: str, max_tokens: int = 300) -> str:
        """
        Send image and query to multimodal LLM for understanding
        
        Args:
            image_path: Path to image file
            query: Question or instruction about the image
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM's answer about the image
        """
        try:
            image_b64 = self.encode_image(image_path)
            
            payload = {
                "model": self.model,
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
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"


def test_image_understanding():
    """Test image understanding on 3 sample disaster images"""
    
    print("=" * 60)
    print("Phase 6c: Image Understanding Test")
    print("=" * 60)
    print()
    
    # Initialize client
    client = OllamaMultimodalClient(model="llava:7b")
    
    # Define test images and queries
    data_dir = Path(__file__).parent.parent / "data" / "sample_images_n12"
    
    test_cases = [
        {
            "image": "01_令和２年７月豪雨被害_熊本写真.jpg",
            "queries": [
                "この画像に写っている災害の種類を教えてください。",
                "被害の状況を詳しく説明してください。",
                "この災害による主な被害は何ですか？"
            ]
        },
        {
            "image": "03_熊本地震_フリー画像_日本防災士会.jpg",
            "queries": [
                "この画像に写っている災害の種類を教えてください。",
                "建物の被害状況はどのようですか？",
                "この災害の特徴的な被害を説明してください。"
            ]
        },
        {
            "image": "06_土砂災害フリー画像.jpg",
            "queries": [
                "この画像に写っている災害の種類を教えてください。",
                "土砂災害の規模や範囲について説明してください。",
                "この災害による主な危険性は何ですか？"
            ]
        }
    ]
    
    results = []
    
    for idx, test_case in enumerate(test_cases, 1):
        image_path = data_dir / test_case["image"]
        
        if not image_path.exists():
            print(f"⚠ Image not found: {image_path}")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Test Case {idx}: {test_case['image']}")
        print(f"{'=' * 60}\n")
        
        case_result = {
            "image": test_case["image"],
            "qa_pairs": []
        }
        
        for q_idx, query in enumerate(test_case["queries"], 1):
            print(f"📝 Query {q_idx}: {query}")
            print(f"🔄 Processing...")
            
            answer = client.understand_image_with_query(image_path, query, max_tokens=300)
            
            print(f"💬 Answer:\n{answer}\n")
            
            case_result["qa_pairs"].append({
                "query": query,
                "answer": answer
            })
        
        results.append(case_result)
    
    # Save results
    output_path = Path(__file__).parent / "indices" / "image_understanding_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"✅ Test completed! Results saved to:")
    print(f"   {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_image_understanding()
