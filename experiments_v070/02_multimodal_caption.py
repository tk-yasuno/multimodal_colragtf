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
CoLRAG-TF v0.7.0 - Phase 3: マルチモーダルキャプション生成

表に対してはOCR（PaddleOCR/Tesseract）でテキスト抽出後、
テキストLLMで構造化キャプションを生成します。
図・画像はマルチモーダルLLM（llava:7b）で直接キャプション生成します。

Usage:
    .venv-coltf\\Scripts\\python.exe experiments_v070\\02_multimodal_caption.py
    .venv-coltf\\Scripts\\python.exe experiments_v070\\02_multimodal_caption.py --batch-size 4
    .venv-coltf\\Scripts\\python.exe experiments_v070\\02_multimodal_caption.py --ocr-engine tesseract
"""

import sys
import argparse
import json
import yaml
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import base64
from io import BytesIO

try:
    from PIL import Image
    print("✅ Pillow imported")
except ImportError as e:
    print(f"❌ Pillow not installed: {e}")
    sys.exit(1)

try:
    import httpx
    print("✅ httpx imported")
except ImportError as e:
    print(f"❌ httpx not installed: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    print("✅ All dependencies loaded")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# OCR関連のインポート（オプション）
try:
    import pytesseract
    # Tesseract実行ファイルのパスを設定
    tesseract_path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    if tesseract_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
    TESSERACT_AVAILABLE = True
    print("✅ Tesseract OCR available")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  Tesseract OCR not available")

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    print("✅ PaddleOCR available")
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️  PaddleOCR not available")


class TableOCREngine:
    """表専用OCRエンジン"""
    
    def __init__(self, engine: str = "pymupdf"):
        self.engine = engine
        
        if engine == "tesseract" and not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract OCR is not available")
        elif engine == "paddle":
            # PaddleOCRはサブプロセスで実行（PyTorch競合回避）
            ocr_venv_python = Path(".venv-ocr/Scripts/python.exe")
            if not ocr_venv_python.exists():
                raise RuntimeError(f".venv-ocr environment not found: {ocr_venv_python}")
            self.ocr_python = ocr_venv_python
            self.ocr_script = Path("experiments_v070/ocr_subprocess.py")
            if not self.ocr_script.exists():
                raise RuntimeError(f"OCR subprocess script not found: {self.ocr_script}")
            print(f"✅ PaddleOCR初期化完了 (subprocess mode: {self.ocr_python})")
        elif engine == "pymupdf":
            # PyMuPDFでPDFから直接テキスト抽出（OCRより高精度）
            import fitz
            print(f"✅ PyMuPDF初期化完了 (direct PDF text extraction)")
        else:
            print(f"✅ Tesseract OCR初期化完了")
    
    def extract_text_from_table(self, figure_block: Dict[str, Any], pdf_dir: Path) -> str:
        """
        表からテキストを抽出
        
        Args:
            figure_block: 図表ブロック情報（image_path, pdf_name, page_num, bbox含む）
            pdf_dir: PDFファイルのディレクトリ
        
        Returns:
            抽出されたテキスト
        """
        try:
            if self.engine == "pymupdf":
                return self._extract_with_pymupdf(figure_block, pdf_dir)
            elif self.engine == "paddle":
                image_path = Path(figure_block.get("image_path", ""))
                return self._extract_with_paddle(image_path)
            else:
                image_path = Path(figure_block.get("image_path", ""))
                return self._extract_with_tesseract(image_path)
        except Exception as e:
            print(f"⚠️  テキスト抽出エラー: {figure_block.get('block_id', 'unknown')} - {e}")
            return ""
    
    def _extract_with_pymupdf(self, figure_block: Dict[str, Any], pdf_dir: Path) -> str:
        """PyMuPDFでPDFから直接テキスト抽出"""
        import fitz
        
        pdf_name = figure_block.get("pdf_name", "")
        page_num = figure_block.get("page_num", 0)
        bbox = figure_block.get("bbox", [])
        
        if not pdf_name or not bbox:
            return ""
        
        pdf_path = pdf_dir / f"{pdf_name}.pdf"
        if not pdf_path.exists():
            print(f"⚠️  PDF not found: {pdf_path}")
            return ""
        
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return ""
        
        page = doc[page_num - 1]  # 0-indexed
        
        # バウンディングボックスを取得（x1, y1, x2, y2）
        x1, y1, x2, y2 = bbox
        
        # PDFページの実際のサイズを取得
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # layout_blocks.jsonlのbboxは画像座標（150 DPI）なので、PDF座標に変換
        dpi = 150
        zoom = dpi / 72.0
        x1_pdf = x1 / zoom
        y1_pdf = y1 / zoom
        x2_pdf = x2 / zoom
        y2_pdf = y2 / zoom
        
        # テキスト抽出領域を指定
        rect = fitz.Rect(x1_pdf, y1_pdf, x2_pdf, y2_pdf)
        text = page.get_text("text", clip=rect)
        
        doc.close()
        return text.strip()
    
    def _extract_with_tesseract(self, image_path: Path) -> str:
        """Tesseractでテキスト抽出"""
        img = Image.open(image_path)
        # 日本語+英語の設定
        text = pytesseract.image_to_string(img, lang='jpn+eng')
        return text.strip()
    
    def _extract_with_paddle(self, image_path: Path) -> str:
        """PaddleOCRでテキスト抽出（サブプロセス実行）"""
        try:
            # OCR専用環境でサブプロセスを実行
            result = subprocess.run(
                [str(self.ocr_python), str(self.ocr_script), str(image_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"⚠️  OCRサブプロセスエラー: {result.stderr}")
                return ""
            
            # JSON結果をパース
            ocr_result = json.loads(result.stdout)
            
            if ocr_result.get("success"):
                return ocr_result.get("text", "")
            else:
                print(f"⚠️  OCR失敗: {ocr_result.get('error', 'Unknown error')}")
                return ""
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  OCRタイムアウト: {image_path}")
            return ""
        except json.JSONDecodeError as e:
            print(f"⚠️  OCR結果のパースエラー: {e}")
            return ""
        except Exception as e:
            print(f"⚠️  OCR実行エラー: {e}")
            return ""


class OllamaMultimodalClient:
    """Ollama マルチモーダル API クライアント"""
    
    def __init__(self, config: Dict[str, Any], ocr_engine: TableOCREngine = None):
        self.config = config
        self.base_url = config['ollama']['base_url']
        self.model = config['models']['primary']['ollama_model']
        self.timeout = config['ollama'].get('timeout', 120)
        self.client = httpx.Client(timeout=self.timeout)
        self.ocr_engine = ocr_engine
        
        # プロンプトテンプレート
        self.caption_prompt = config['prompts']['caption_generation']
        self.table_caption_prompt = config['prompts'].get('table_caption_from_text', 
            "以下は表から抽出されたテキストです。このテキストを基に、表の内容を簡潔に説明してください（100-200文字）：\n\n{text}")
        
        print(f"✅ Ollama クライアント初期化")
        print(f"   Model: {self.model}")
        print(f"   Base URL: {self.base_url}")
        if ocr_engine:
            print(f"   OCR Engine: {ocr_engine.engine}")
    
    def encode_image(self, image_path: Path) -> str:
        """画像をBase64エンコード"""
        try:
            with Image.open(image_path) as img:
                # RGB変換（透明度チャンネル削除）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # メモリ上でPNG形式に変換
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                
                # Base64エンコード
                return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"⚠️  画像エンコードエラー: {image_path} - {e}")
            return None
    
    def generate_caption(self, image_path: Path, element_type: str = "table") -> str:
        """
        図表に対するキャプションを生成
        
        Args:
            image_path: 図表画像のパス
            element_type: 要素タイプ (table, figure, image)
        
        Returns:
            生成されたキャプション（日本語）
        """
        # 画像をBase64エンコード
        image_base64 = self.encode_image(image_path)
        if not image_base64:
            return f"[{element_type}の説明を生成できませんでした]"
        
        # タイプに応じたプロンプト調整
        type_text = {
            "table": "表",
            "figure": "図",
            "image": "画像"
        }.get(element_type, "図表")
        
        prompt = self.caption_prompt.format(type=type_text)
        
        try:
            # Ollama API リクエスト（マルチモーダル対応: /api/chat）
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_base64]
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.config['models']['primary'].get('temperature', 0.2),
                        "num_predict": self.config['models']['primary'].get('max_tokens', 256)
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # chatエンドポイントは message.content にレスポンスが入る
                caption = result.get('message', {}).get('content', '').strip()
                return caption if caption else f"[{type_text}の説明なし]"
            else:
                print(f"⚠️  API エラー: {response.status_code} - {response.text[:200]}")
                return f"[API エラー: {response.status_code}]"
        
        except Exception as e:
            print(f"⚠️  キャプション生成エラー: {e}")
            return f"[生成エラー: {str(e)[:50]}]"
    
    def generate_caption_from_ocr_text(self, ocr_text: str, element_type: str = "table") -> str:
        """
        OCR抽出テキストから表のキャプションを生成
        
        Args:
            ocr_text: OCRで抽出されたテキスト
            element_type: 要素タイプ
        
        Returns:
            生成されたキャプション（日本語）
        """
        if not ocr_text or len(ocr_text) < 10:
            return f"[{element_type}のテキストが抽出できませんでした]"
        
        # テキストLLMモデル使用（qwen2.5:7b等）
        text_model = "qwen2.5:7b-instruct-q4_k_m"
        
        prompt = self.table_caption_prompt.format(text=ocr_text[:2000])  # 最大2000文字
        
        try:
            # テキスト専用エンドポイント: /api/generate
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": text_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 256
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                caption = result.get('response', '').strip()
                return caption if caption else f"[テキストから{element_type}の説明を生成できませんでした]"
            else:
                print(f"⚠️  テキストLLM APIエラー: {response.status_code}")
                return f"[テキストLLM APIエラー: {response.status_code}]"
        
        except Exception as e:
            print(f"⚠️  テキストからのキャプション生成エラー: {e}")
            return f"[テキスト生成エラー: {str(e)[:50]}]"
    
    def generate_captions_batch(self, blocks: List[Dict], batch_size: int = 8, pdf_dir: Path = None) -> List[Dict]:
        """
        バッチ処理でキャプション生成
        
        表の場合: OCR→テキストLLM
        図・画像の場合: マルチモーダルLLM
        
        Args:
            blocks: 図表ブロックのリスト
            batch_size: バッチサイズ
            pdf_dir: PDFファイルのディレクトリ（pymupdf OCR用）
        
        Returns:
            キャプション付き図表ブロックのリスト
        """
        results = []
        
        for i in tqdm(range(0, len(blocks), batch_size), desc="キャプション生成"):
            batch = blocks[i:i+batch_size]
            
            for block in batch:
                image_path = Path(block['image_path'])
                
                if not image_path.exists():
                    print(f"⚠️  画像が見つかりません: {image_path}")
                    block['caption'] = f"[画像ファイルが見つかりません]"
                elif block['type'] == 'table' and self.ocr_engine:
                    # 表の場合: OCRでテキスト抽出→テキストLLMでキャプション生成
                    ocr_text = self.ocr_engine.extract_text_from_table(block, pdf_dir)
                    caption = self.generate_caption_from_ocr_text(ocr_text, block['type'])
                    block['caption'] = caption
                    block['ocr_text'] = ocr_text  # OCRテキストも保存
                else:
                    # 図・画像の場合: マルチモーダルLLMで直接キャプション生成
                    caption = self.generate_caption(image_path, block['type'])
                    block['caption'] = caption
                
                results.append(block)
        
        return results


def load_config(config_path: Path) -> Dict:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_layout_blocks(input_path: Path) -> List[Dict]:
    """レイアウトブロックJSONLを読み込み"""
    layouts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                layouts.append(json.loads(line))
    return layouts


def extract_figure_blocks(layouts: List[Dict]) -> List[Dict]:
    """全ページから図表ブロックを抽出"""
    figure_blocks = []
    
    for layout in layouts:
        for fig_block in layout.get('figure_blocks', []):
            # ページ情報を追加
            fig_block['page_id'] = layout['page_id']
            fig_block['pdf_name'] = layout['pdf_name']
            fig_block['page_num'] = layout['page_num']
            fig_block['volume'] = layout.get('volume', '')
            fig_block['chapter'] = layout.get('chapter', '')
            figure_blocks.append(fig_block)
    
    return figure_blocks


def merge_captions_to_layouts(layouts: List[Dict], captioned_blocks: List[Dict]) -> List[Dict]:
    """キャプション付き図表ブロックを元のレイアウトにマージ"""
    # block_id -> caption のマッピングを作成
    caption_map = {block['block_id']: block['caption'] for block in captioned_blocks}
    
    # 各レイアウトの figure_blocks にキャプションを追加
    for layout in layouts:
        for fig_block in layout.get('figure_blocks', []):
            block_id = fig_block['block_id']
            if block_id in caption_map:
                fig_block['caption'] = caption_map[block_id]
    
    return layouts


def save_layout_blocks(layouts: List[Dict], output_path: Path):
    """更新されたレイアウトブロックを保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for layout in layouts:
            f.write(json.dumps(layout, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="マルチモーダルキャプション生成")
    parser.add_argument('--config', type=str, 
                        default='experiments_v070/configs/mm_llm_config.yaml',
                        help='マルチモーダルLLM設定ファイル')
    parser.add_argument('--input', type=str,
                        default='experiments_v070/indices/layout_blocks.jsonl',
                        help='入力レイアウトブロックファイル')
    parser.add_argument('--output', type=str,
                        default='experiments_v070/indices/layout_blocks_captioned.jsonl',
                        help='出力キャプション付きレイアウトブロックファイル')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='バッチサイズ')
    parser.add_argument('--ocr-engine', type=str, default='pymupdf', choices=['pymupdf', 'tesseract', 'paddle', 'none'],
                        help='表のテキスト抽出エンジン (pymupdf/tesseract/paddle/none)')
    
    args = parser.parse_args()
    
    # パス解決
    config_path = Path(args.config)
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print("=" * 60)
    print(" マルチモーダルキャプション生成")
    print("=" * 60)
    print()
    
    # 設定読み込み
    print("📋 設定読み込み...")
    config = load_config(config_path)
    
    # レイアウトブロック読み込み
    print(f"📄 レイアウトブロック読み込み: {input_path}")
    layouts = load_layout_blocks(input_path)
    print(f"   ✅ {len(layouts)} ページ読み込み完了")
    
    # 図表ブロック抽出
    print("🖼️  図表ブロック抽出中...")
    figure_blocks = extract_figure_blocks(layouts)
    print(f"   ✅ {len(figure_blocks)} 個の図表ブロック検出")
    
    if len(figure_blocks) == 0:
        print("\n⚠️  図表ブロックが見つかりません。処理を終了します。")
        return
    
    # タイプ別の統計
    type_counts = {}
    for block in figure_blocks:
        block_type = block['type']
        type_counts[block_type] = type_counts.get(block_type, 0) + 1
    
    print("\n図表タイプ別統計:")
    for block_type, count in sorted(type_counts.items()):
        print(f"   - {block_type}: {count} 個")
    
    # OCRエンジン初期化（表がある場合のみ）
    ocr_engine = None
    table_count = type_counts.get('table', 0)
    if table_count > 0 and args.ocr_engine != 'none':
        print(f"\n🔍 OCRエンジン初期化 ({args.ocr_engine})...")
        try:
            ocr_engine = TableOCREngine(args.ocr_engine)
        except RuntimeError as e:
            print(f"⚠️  {e}")
            print(f"   表は画像ベースのキャプション生成にフォールバックします")
            ocr_engine = None
    
    # Ollama クライアント初期化
    print("\n🤖 Ollama マルチモーダルクライアント初期化...")
    client = OllamaMultimodalClient(config, ocr_engine)
    
    # PDFディレクトリを取得（pymupdf OCR用）
    pdf_dir = None
    if ocr_engine and ocr_engine.engine == 'pymupdf':
        volume_mapping_path = Path('experiments_v070/disaster_volume_mapping.json')
        if volume_mapping_path.exists():
            with open(volume_mapping_path, 'r', encoding='utf-8') as f:
                volume_mapping = json.load(f)
                data_directory = volume_mapping.get('metadata', {}).get('data_directory')
                if data_directory:
                    pdf_dir = Path(data_directory)
                    print(f"   📁 PDFディレクトリ: {pdf_dir}")
    
    # キャプション生成
    print(f"\n📝 キャプション生成開始 (batch_size={args.batch_size})...")
    captioned_blocks = client.generate_captions_batch(figure_blocks, args.batch_size, pdf_dir)
    print(f"   ✅ {len(captioned_blocks)} 個のキャプション生成完了")
    
    # レイアウトにマージ
    print("\n🔄 キャプションをレイアウトにマージ中...")
    updated_layouts = merge_captions_to_layouts(layouts, captioned_blocks)
    
    # 保存
    print(f"\n💾 更新されたレイアウトブロックを保存: {output_path}")
    save_layout_blocks(updated_layouts, output_path)
    
    print("\n" + "=" * 60)
    print(" キャプション生成完了")
    print("=" * 60)
    print(f"総ページ数: {len(updated_layouts)}")
    print(f"総図表ブロック: {len(captioned_blocks)}")
    print(f"出力ファイル: {output_path.absolute()}")
    print()
    print("次のステップ:")
    print("  python 03_build_multimodal_index.py")


if __name__ == "__main__":
    main()
