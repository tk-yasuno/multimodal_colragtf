import json

with open('experiments_v070/indices/layout_blocks_captioned.jsonl', 'r', encoding='utf-8') as f:
    layouts = [json.loads(line) for line in f if line.strip()]

print("=" * 80)
print("生成されたキャプション一覧")
print("=" * 80)
print()

count = 0
for layout in layouts:
    for fig_block in layout.get('figure_blocks', []):
        count += 1
        caption = fig_block.get('caption', '[キャプションなし]')
        print(f"{count}. {fig_block['block_id']}")
        print(f"   タイプ: {fig_block['type']}")
        print(f"   信頼度: {fig_block['confidence']:.2%}")
        print(f"   キャプション: {caption[:200]}{'...' if len(caption) > 200 else ''}")
        print()

print(f"総キャプション数: {count}")
