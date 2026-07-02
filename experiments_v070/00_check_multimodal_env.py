#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
環境チェックスクリプト - CoLRAG-TF v0.7.0
GPU、マルチモーダルLLM、依存パッケージの動作確認

Usage:
    python 00_check_multimodal_env.py
"""

import sys
import json
from pathlib import Path

def check_python_version():
    """Python バージョン確認"""
    print("=" * 60)
    print("1. Python バージョン")
    print("=" * 60)
    version = sys.version_info
    print(f"   {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        print("   ✅ OK (Python 3.10+)")
    else:
        print("   ⚠️  推奨: Python 3.10 以上")
    print()

def check_gpu():
    """GPU 確認"""
    print("=" * 60)
    print("2. GPU / CUDA")
    print("=" * 60)
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   Device {i}: {props.name}")
                print(f"      Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"      Compute capability: {props.major}.{props.minor}")
            print("   ✅ GPU OK")
        else:
            print("   ⚠️  CUDA not available (CPU mode)")
    except ImportError:
        print("   ❌ PyTorch not installed")
        print("      Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print()

def check_core_packages():
    """コアパッケージ確認"""
    print("=" * 60)
    print("3. コアパッケージ")
    print("=" * 60)
    
    packages = {
        "numpy": "数値計算",
        "pandas": "データ処理",
        "sentence_transformers": "埋め込みモデル",
        "faiss": "ベクトル検索",
        "rank_bm25": "BM25検索",
        "httpx": "HTTP クライアント",
    }
    
    for pkg, desc in packages.items():
        try:
            if pkg == "faiss":
                import faiss
                version = faiss.__version__ if hasattr(faiss, '__version__') else "unknown"
            else:
                module = __import__(pkg)
                version = module.__version__ if hasattr(module, '__version__') else "unknown"
            print(f"   ✅ {pkg:25s} {version:10s} ({desc})")
        except ImportError:
            print(f"   ❌ {pkg:25s} {'N/A':10s} ({desc})")
    print()

def check_multimodal_packages():
    """マルチモーダルパッケージ確認"""
    print("=" * 60)
    print("4. マルチモーダルパッケージ")
    print("=" * 60)
    
    packages = {
        "llama_index": "LlamaIndex コア",
        "qdrant_client": "Qdrant クライアント",
        "transformers": "HuggingFace Transformers",
        "PIL": "画像処理 (Pillow)",
        "fitz": "PDF処理 (PyMuPDF)",
        "layoutparser": "レイアウト解析",
        "pytesseract": "OCR",
    }
    
    for pkg, desc in packages.items():
        try:
            if pkg == "PIL":
                from PIL import Image
                import PIL
                version = PIL.__version__
            elif pkg == "fitz":
                import fitz
                version = fitz.version[0] if hasattr(fitz, 'version') else "unknown"
            else:
                module = __import__(pkg)
                version = module.__version__ if hasattr(module, '__version__') else "unknown"
            print(f"   ✅ {pkg:25s} {version:10s} ({desc})")
        except ImportError:
            print(f"   ❌ {pkg:25s} {'N/A':10s} ({desc})")
    print()

def check_ollama():
    """Ollama サービス確認"""
    print("=" * 60)
    print("5. Ollama サービス")
    print("=" * 60)
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"   ✅ Ollama running (http://localhost:11434)")
            print(f"   利用可能モデル: {len(models)}個")
            
            target_models = ["qwen2.5", "qwen2.5-omni", "llama-4-scout", "gemma"]
            found_models = []
            for model in models:
                name = model.get("name", "")
                for target in target_models:
                    if target in name.lower():
                        found_models.append(name)
                        break
            
            if found_models:
                print("   マルチモーダル対応モデル:")
                for model in found_models[:5]:
                    print(f"      - {model}")
            else:
                print("   ⚠️  推奨モデルが見つかりません")
                print("      インストール: ollama pull qwen2.5-omni:7b-instruct-q4_k_m")
        else:
            print(f"   ⚠️  Ollama response error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ollama not running")
        print(f"      Error: {e}")
        print("      起動: ollama serve")
    print()

def check_data_directories():
    """データディレクトリ確認"""
    print("=" * 60)
    print("6. データディレクトリ")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    data_dirs = {
        "災害教訓PDF": base_dir / "data" / "disaster_visual_docus_rn",
        "橋梁診断データ": base_dir / "data" / "doken_bridge_diagnosis_logic",
        "Markdown テキスト": base_dir / "data" / "kasensabo_markdown_text",
        "既存実験結果": base_dir / "0_LogBAK" / "2b_kasensabo_colrag-tf" / "experiments",
    }
    
    for name, path in data_dirs.items():
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.rglob("*")))
                print(f"   ✅ {name:20s} {file_count:5d} files")
            else:
                print(f"   ✅ {name:20s} (file)")
        else:
            print(f"   ⚠️  {name:20s} not found")
    print()

def check_volume_mapping():
    """Volume マッピング確認"""
    print("=" * 60)
    print("7. Volume マッピング")
    print("=" * 60)
    
    mapping_file = Path(__file__).parent / "disaster_volume_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        volumes = mapping.get("volumes", {})
        metadata = mapping.get("metadata", {})
        
        print(f"   ✅ マッピングファイル: {mapping_file.name}")
        print(f"   Volume 数: {len(volumes)}")
        print(f"   推定 PDF 数: {metadata.get('total_pdfs', 'N/A')}")
        print(f"   推定ページ数: {metadata.get('estimated_pages', 'N/A')}")
        print()
        print("   Volume 一覧:")
        for vol_name, vol_data in volumes.items():
            chapter_count = len(vol_data.get("chapters", []))
            print(f"      - {vol_name}: {chapter_count} PDFs")
    else:
        print(f"   ⚠️  マッピングファイル未作成: {mapping_file}")
    print()

def main():
    print("\n" + "=" * 60)
    print(" CoLRAG-TF v0.7.0 環境チェック")
    print("=" * 60)
    print()
    
    check_python_version()
    check_gpu()
    check_core_packages()
    check_multimodal_packages()
    check_ollama()
    check_data_directories()
    check_volume_mapping()
    
    print("=" * 60)
    print(" チェック完了")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  1. 不足パッケージをインストール: pip install -r requirements_v070.txt")
    print("  2. Ollama モデルをダウンロード: ollama pull qwen2.5-omni:7b-instruct-q4_k_m")
    print("  3. レイアウト解析を実行: python 01_layout_analysis.py")
    print()

if __name__ == "__main__":
    main()
