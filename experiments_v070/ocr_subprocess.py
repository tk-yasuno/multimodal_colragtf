"""
OCR専用サブプロセス
PaddleOCRとPyTorchのGPU競合を避けるため、別プロセスで実行
"""
import sys
import json
from pathlib import Path
from paddleocr import PaddleOCR

def extract_text_with_paddle(image_path: str, lang: str = 'japan') -> str:
    """
    PaddleOCRでテキスト抽出
    
    Args:
        image_path: 画像ファイルパス
        lang: 言語設定（デフォルト: japan）
    
    Returns:
        抽出されたテキスト
    """
    try:
        # PaddleOCR初期化（GPU使用）
        ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
        
        # OCR実行
        result = ocr.ocr(image_path, cls=True)
        
        # 結果を整形（行ごとにテキストをまとめる）
        lines = []
        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    text = line[1][0]  # テキスト部分
                    conf = line[1][1]  # 信頼度
                    if conf > 0.5:  # 信頼度50%以上のみ
                        lines.append(text)
        
        return '\n'.join(lines)
    
    except Exception as e:
        return f"ERROR: {str(e)}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: ocr_subprocess.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else 'japan'
    
    if not Path(image_path).exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)
    
    # OCR実行
    text = extract_text_with_paddle(image_path, lang)
    
    # JSON形式で出力
    result = {
        "image_path": image_path,
        "text": text,
        "success": not text.startswith("ERROR:")
    }
    
    print(json.dumps(result, ensure_ascii=False))
