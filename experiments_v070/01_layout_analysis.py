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

r"""
レイアウト解析 - Table Transformer による図表検出
CoLRAG-TF v0.7.0

Usage:
    .\.venv-coltf\Scripts\python.exe experiments_v070\01_layout_analysis.py --config configs/layout_config.yaml
    .\.venv-coltf\Scripts\python.exe experiments_v070\01_layout_analysis.py --sample 10
"""

import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import fitz  # PyMuPDF
    print("✅ PyMuPDF imported")
except ImportError as e:
    print(f"❌ PyMuPDF not installed: {e}")
    print("実行: .venv-coltf\\Scripts\\pip.exe install PyMuPDF")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch imported: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")
    print("実行: .venv-coltf\\Scripts\\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)

try:
    from PIL import Image
    print("✅ Pillow imported")
except ImportError as e:
    print(f"❌ Pillow not installed: {e}")
    print("実行: .venv-coltf\\Scripts\\pip.exe install Pillow")
    sys.exit(1)

try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    print("✅ Transformers imported")
except ImportError as e:
    print(f"❌ Transformers not installed: {e}")
    print("実行: .venv-coltf\\Scripts\\pip.exe install transformers")
    sys.exit(1)

try:
    from tqdm import tqdm
    import numpy as np
    print("✅ All dependencies loaded")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


@dataclass
class BoundingBox:
    """バウンディングボックス"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def expand(self, margin: int) -> 'BoundingBox':
        """マージン拡張"""
        return BoundingBox(
            max(0, self.x1 - margin),
            max(0, self.y1 - margin),
            self.x2 + margin,
            self.y2 + margin
        )


@dataclass
class LayoutElement:
    """レイアウト要素"""
    element_type: str  # "text", "table", "figure", "image"
    bbox: BoundingBox
    confidence: float
    block_id: str


@dataclass
class PageLayout:
    """ページレイアウト"""
    page_id: str
    pdf_name: str
    page_num: int
    width: int
    height: int
    text_blocks: List[Dict]
    figure_blocks: List[Dict]
    
    def to_dict(self) -> Dict:
        """辞書に変換"""
        return {
            "page_id": self.page_id,
            "pdf_name": self.pdf_name,
            "page_num": int(self.page_num),
            "width": int(self.width),
            "height": int(self.height),
            "text_blocks": self.text_blocks,
            "figure_blocks": self.figure_blocks
        }


class TableTransformerDetector:
    """Table Transformer による図表検出"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Table Transformer をロード中... (device: {self.device})")
        model_name = config['model']['name']
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # FP16混合精度
        if config['model']['precision'] == 'fp16' and self.device.type == 'cuda':
            self.model = self.model.half()
            print("   ✅ FP16混合精度モード")
        
        self.confidence_threshold = config['detection']['confidence_threshold']
        self.target_types = set(config['detection']['target_types'])
        print(f"   ✅ 検出対象: {self.target_types}")
    
    def detect(self, image: Image.Image) -> List[LayoutElement]:
        """画像から図表検出"""
        # 前処理
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # FP16変換
        if self.model.dtype == torch.float16:
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].half()
        
        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 後処理
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # LayoutElement に変換
        elements = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()]
            
            # 対象タイプのみ
            if label_name not in self.target_types:
                continue
            
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            bbox = BoundingBox(x1, y1, x2, y2)
            
            # 最小面積フィルタ
            if bbox.area < self.config['detection']['min_box_area']:
                continue
            
            block_id = f"{label_name}_{len(elements)}"
            elements.append(LayoutElement(
                element_type=label_name,
                bbox=bbox,
                confidence=score.item(),
                block_id=block_id
            ))
        
        return elements
    
    def apply_nms(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Non-Maximum Suppression (重複除去)"""
        if len(elements) == 0:
            return []
        
        # IoU計算用
        def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
            x1 = max(box1.x1, box2.x1)
            y1 = max(box1.y1, box2.y1)
            x2 = min(box1.x2, box2.x2)
            y2 = min(box1.y2, box2.y2)
            
            if x2 < x1 or y2 < y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = box1.area
            area2 = box2.area
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # 信頼度でソート
        sorted_elements = sorted(elements, key=lambda e: e.confidence, reverse=True)
        
        keep = []
        iou_threshold = self.config['detection']['nms_iou_threshold']
        
        while sorted_elements:
            current = sorted_elements.pop(0)
            keep.append(current)
            
            # 残りの要素とIoU計算
            sorted_elements = [
                e for e in sorted_elements
                if compute_iou(current.bbox, e.bbox) < iou_threshold
            ]
        
        return keep


class PDFLayoutAnalyzer:
    """PDF レイアウト解析"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detector = TableTransformerDetector(config)
        self.output_dir = Path(config['output']['block_image_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def pdf_to_images(self, pdf_path: Path) -> List[Tuple[int, Image.Image]]:
        """PDF → PIL Image 変換"""
        doc = fitz.open(pdf_path)
        images = []
        dpi = self.config['pdf']['dpi']
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            
            # PIL Image に変換
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # 最大サイズ制限
            max_dim = self.config['pdf']['max_image_dimension']
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            images.append((page_num + 1, img))
        
        doc.close()
        return images
    
    def extract_text_from_page(self, pdf_path: Path, page_num: int) -> str:
        """ページからテキスト抽出"""
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        text = page.get_text()
        doc.close()
        return text.strip()
    
    def analyze_page(self, pdf_path: Path, page_num: int, image: Image.Image) -> PageLayout:
        """ページレイアウト解析"""
        pdf_name = pdf_path.stem
        page_id = f"{pdf_name}_page{page_num:03d}"
        
        # 図表検出
        elements = self.detector.detect(image)
        elements = self.detector.apply_nms(elements)
        
        # テキストブロックと図表ブロックに分離
        text_blocks = []
        figure_blocks = []
        
        expand_margin = self.config['detection']['expand_margin']
        
        for elem in elements:
            bbox_expanded = elem.bbox.expand(expand_margin)
            
            # バウンディングボックス内の画像を切り出し
            crop_box = (bbox_expanded.x1, bbox_expanded.y1, bbox_expanded.x2, bbox_expanded.y2)
            block_image = image.crop(crop_box)
            
            block_data = {
                "block_id": f"{page_id}_{elem.block_id}",
                "type": elem.element_type,
                "bbox": [int(elem.bbox.x1), int(elem.bbox.y1), int(elem.bbox.x2), int(elem.bbox.y2)],
                "confidence": float(elem.confidence)
            }
            
            if elem.element_type in ["table", "figure", "image"]:
                # 図表ブロック
                if self.config['output']['save_block_images']:
                    image_filename = f"{page_id}_{elem.block_id}.png"
                    image_path = self.output_dir / image_filename
                    block_image.save(image_path)
                    # 相対パスとして保存（プロジェクトルートからの相対パス）
                    block_data["image_path"] = str(image_path).replace('\\', '/')
                
                figure_blocks.append(block_data)
            else:
                # テキストブロック（OCR）
                text = self.extract_text_from_bbox(pdf_path, page_num, elem.bbox)
                if len(text) >= self.config['ocr']['min_text_length']:
                    block_data["text"] = text
                    text_blocks.append(block_data)
        
        # 図表が検出されない場合、ページ全体をテキストブロックとして扱う
        if len(figure_blocks) == 0 and len(text_blocks) == 0:
            full_text = self.extract_text_from_page(pdf_path, page_num)
            if len(full_text) >= self.config['ocr']['min_text_length']:
                text_blocks.append({
                    "block_id": f"{page_id}_text_full",
                    "type": "text",
                    "bbox": [0, 0, int(image.width), int(image.height)],
                    "confidence": 1.0,
                    "text": full_text
                })
        
        return PageLayout(
            page_id=page_id,
            pdf_name=pdf_name,
            page_num=page_num,
            width=int(image.width),
            height=int(image.height),
            text_blocks=text_blocks,
            figure_blocks=figure_blocks
        )
    
    def extract_text_from_bbox(self, pdf_path: Path, page_num: int, bbox: BoundingBox) -> str:
        """バウンディングボックス内のテキスト抽出"""
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        
        # PyMuPDFの座標系に変換（DPI補正）
        dpi = self.config['pdf']['dpi']
        zoom = dpi / 72.0
        rect = fitz.Rect(
            bbox.x1 / zoom, 
            bbox.y1 / zoom, 
            bbox.x2 / zoom, 
            bbox.y2 / zoom
        )
        
        text = page.get_text(clip=rect)
        doc.close()
        return text.strip()
    
    def analyze_pdf(self, pdf_path: Path, max_pages: int = None) -> List[PageLayout]:
        """PDF全体のレイアウト解析"""
        print(f"\n📄 {pdf_path.name}")
        
        # PDF → 画像
        images = self.pdf_to_images(pdf_path)
        if max_pages:
            images = images[:max_pages]
        
        layouts = []
        for page_num, image in tqdm(images, desc="  ページ解析"):
            layout = self.analyze_page(pdf_path, page_num, image)
            layouts.append(layout)
        
        print(f"   ✅ {len(layouts)} ページ解析完了")
        print(f"      テキストブロック: {sum(len(l.text_blocks) for l in layouts)}")
        print(f"      図表ブロック: {sum(len(l.figure_blocks) for l in layouts)}")
        
        return layouts


def load_volume_mapping(mapping_file: Path) -> Dict:
    """Volumeマッピングロード"""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_pdf_files(data_dir: Path, volume_mapping: Dict) -> List[Tuple[str, Path]]:
    """PDFファイル検索"""
    pdf_files = []
    
    for volume_name, volume_data in volume_mapping['volumes'].items():
        for chapter_name in volume_data['chapters']:
            # PDFファイル検索
            pdf_path = data_dir / chapter_name
            if not pdf_path.exists():
                # images/ ディレクトリ内も検索
                pdf_path = data_dir / "images" / chapter_name
            
            if pdf_path.exists():
                pdf_files.append((volume_name, pdf_path))
            else:
                print(f"⚠️  PDF not found: {chapter_name}")
    
    return pdf_files


def main():
    parser = argparse.ArgumentParser(description="レイアウト解析 - CoLRAG-TF v0.7.0")
    parser.add_argument("--config", type=str, default="configs/layout_config.yaml",
                        help="設定ファイルパス")
    parser.add_argument("--sample", type=int, default=None,
                        help="サンプルページ数（デバッグ用）")
    parser.add_argument("--pdf", type=str, default=None,
                        help="特定のPDFのみ処理")
    args = parser.parse_args()
    
    # 設定ロード
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print(" レイアウト解析 - Table Transformer")
    print("=" * 60)
    print(f"設定ファイル: {config_path}")
    print(f"検出信頼度閾値: {config['detection']['confidence_threshold']}")
    print(f"PDF DPI: {config['pdf']['dpi']}")
    print()
    
    # Volumeマッピングロード
    mapping_file = Path(__file__).parent / "disaster_volume_mapping.json"
    volume_mapping = load_volume_mapping(mapping_file)
    
    # データディレクトリ
    data_dir = Path(volume_mapping['metadata']['data_directory'])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir
    
    # PDFファイル検索
    pdf_files = find_pdf_files(data_dir, volume_mapping)
    print(f"📚 検出されたPDF: {len(pdf_files)}個")
    
    if args.pdf:
        # 特定PDFのみ
        pdf_files = [(v, p) for v, p in pdf_files if args.pdf in p.name]
        print(f"   フィルタ適用: {len(pdf_files)}個")
    
    # レイアウト解析器初期化
    analyzer = PDFLayoutAnalyzer(config)
    
    # 解析実行
    all_layouts = []
    for volume_name, pdf_path in pdf_files:
        layouts = analyzer.analyze_pdf(pdf_path, max_pages=args.sample)
        
        # Volumeメタデータ追加
        for layout in layouts:
            layout_dict = layout.to_dict()
            layout_dict['volume'] = volume_name
            layout_dict['chapter'] = pdf_path.stem
            all_layouts.append(layout_dict)
    
    # 結果保存
    output_file = Path(__file__).parent / config['output']['layout_json_path']
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for layout in all_layouts:
            f.write(json.dumps(layout, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 60)
    print(" 解析完了")
    print("=" * 60)
    print(f"総ページ数: {len(all_layouts)}")
    print(f"総テキストブロック: {sum(len(l['text_blocks']) for l in all_layouts)}")
    print(f"総図表ブロック: {sum(len(l['figure_blocks']) for l in all_layouts)}")
    print(f"出力ファイル: {output_file}")
    print()
    print("次のステップ:")
    print("  python 02_multimodal_caption.py")


if __name__ == "__main__":
    main()
