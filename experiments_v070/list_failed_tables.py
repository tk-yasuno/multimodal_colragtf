"""失敗した表画像のリストを作成"""
import json
from pathlib import Path

jsonl_path = Path("experiments_v070/indices/layout_blocks_captioned.jsonl")
lines = jsonl_path.read_text(encoding='utf-8').splitlines()

# 失敗した表を収集
failed_tables = []
for line in lines:
    data = json.loads(line)
    for fb in data.get('figure_blocks', []):
        if fb.get('caption', '').startswith('[table'):
            failed_tables.append(fb)

print(f"失敗した表: {len(failed_tables)}件\n")
print("="*70)

for i, table in enumerate(failed_tables, 1):
    print(f"\n{i}. Page: {table['page_id']}")
    print(f"   Image: {table['image_path']}")
    print(f"   Confidence: {table.get('confidence', 0):.3f}")
    print(f"   OCR text length: {len(table.get('ocr_text', ''))}")
    print(f"   Bbox: {table.get('bbox', [])}")

print("\n" + "="*70)
print(f"\n画像確認コマンド例:")
print(f"Start-Process '{failed_tables[0]['image_path']}'")
