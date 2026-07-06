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
CoLRAG-TF v0.7.0 - Phase 7c: 要因→教訓→防災対策 QA生成器

災害事例を振り返り、大規模化の要因・教訓・防災対策を問う2-hop QAを生成します。

質問例:
- この地震災害が大規模化した要因と教訓を踏まえ、今後の防災・減災の取り組みとしてどんな課題がありますか？

Usage:
    .venv-coltf\\Scripts\\python.exe experiments_v070\\07c_generate_cause_mitigation_qa.py --target-count 100
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import random
import time

try:
    import requests
    print("✅ requests imported")
except ImportError as e:
    print(f"❌ requests not installed: {e}")
    sys.exit(1)


@dataclass
class CauseMitigationQA:
    """要因→教訓→防災対策QA質問"""
    question_id: str
    question: str
    answer: str
    question_type: str = "cause_and_mitigation"
    block_id: str = ""
    hop_count: int = 2
    requires_image: bool = True
    difficulty: str = "hard"
    
    # メタデータ
    disaster_type: str = "不明"
    pdf_name: str = ""


# 災害種別キーワード
DISASTER_KEYWORDS = {
    "地震": ["地震", "震災", "余震", "液状化", "耐震"],
    "洪水": ["洪水", "豪雨", "氾濫", "水害", "浸水", "堤防"],
    "土砂災害": ["土砂", "崩壊", "がけ崩れ", "斜面", "地すべり"],
    "台風": ["台風", "暴風", "高潮"],
    "津波": ["津波", "津波被害"],
    "噴火": ["噴火", "火山", "降灰"]
}

# 要因分析に有用なキーワード（被害規模や課題を示す）
CAUSE_KEYWORDS = [
    "全壊", "半壊", "死者", "行方不明", "被災者", "避難者",
    "甚大", "大規模", "壊滅", "孤立", "寸断",
    "遅れ", "不足", "機能不全", "想定外", "課題"
]


class ConceptExtractor:
    """キャプションから概念を抽出"""
    
    @staticmethod
    def extract_disaster_type(text: str) -> Optional[str]:
        """災害種別を抽出"""
        for disaster_type, keywords in DISASTER_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return disaster_type
        return None
    
    @staticmethod
    def has_cause_keywords(text: str) -> bool:
        """要因分析に適したキーワードを含むか"""
        return any(kw in text for kw in CAUSE_KEYWORDS)


class CauseMitigationQAGenerator:
    """要因→教訓→防災対策QA生成器"""
    
    PROMPT_TEMPLATE = """以下は災害事例の図表キャプションです。

【災害事例】
{caption}

【タスク】
この災害事例を振り返り、以下の形式で2-hop質問と回答を1つ生成してください：

質問: この{disaster_type}災害が大規模化した要因と教訓を踏まえ、今後の防災・減災の取り組みとしてどんな課題がありますか？

回答の構成（200文字以上で記述）:
1. 要因分析: 「～が不足していた」「～の準備が遅れた」「～の想定が甘かった」など、災害が拡大した具体的な要因を2-3点
2. 得られた教訓: 要因から学んだ具体的な教訓を1-2点
3. 防災・減災対策の課題: 「～の事前整備が必要」「～の体制強化が課題」「～の見直しが急務」など、今後取り組むべき具体的な対策を2-3点

生成してください："""

    def __init__(self, model_name: str = "qwen2.5:7b-instruct-q4_k_m",
                 base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.extractor = ConceptExtractor()
        print(f"✅ CauseMitigationQAGenerator initialized: {model_name}")
    
    def generate_qa(self, block: Dict[str, Any]) -> Optional[CauseMitigationQA]:
        """ブロックから要因→防災対策QAを生成"""
        
        caption = block.get('caption', '')
        
        if len(caption) < 80:
            return None
        
        # 災害種別を抽出
        disaster_type = self.extractor.extract_disaster_type(caption)
        
        if not disaster_type:
            return None
        
        # プロンプト生成
        prompt = self.PROMPT_TEMPLATE.format(
            caption=caption,
            disaster_type=disaster_type
        )
        
        # LLMに質問生成を依頼
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 600  # 長い回答用
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # QAペアをパース
                question, answer = self._parse_qa_response(response_text)
                
                if question and answer and len(answer) >= 150:
                    return CauseMitigationQA(
                        question_id=f"{block.get('block_id', '')}_cause_mitigation",
                        question=question,
                        answer=answer,
                        block_id=block.get('block_id', ''),
                        disaster_type=disaster_type,
                        pdf_name=block.get('pdf_name', '')
                    )
        except Exception as e:
            print(f"\n⚠️ QA generation error: {e}")
        
        return None
    
    def _parse_qa_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """LLM応答からQAペアをパース"""
        lines = response_text.strip().split('\n')
        question = None
        answer_lines = []
        in_answer = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('質問:') or line.startswith('質問：'):
                question = line.split(':', 1)[-1].split('：', 1)[-1].strip()
            elif line.startswith('回答:') or line.startswith('回答：'):
                answer_part = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                if answer_part:
                    answer_lines.append(answer_part)
                in_answer = True
            elif in_answer and line:
                answer_lines.append(line)
        
        answer = ' '.join(answer_lines) if answer_lines else None
        
        return question, answer


def select_candidate_blocks(blocks: List[Dict[str, Any]],
                            target_count: int = 150) -> List[Dict[str, Any]]:
    """
    要因分析に適したブロックを選択
    
    優先順位:
    1. 被害規模が大きい（CAUSE_KEYWORDSを含む）
    2. 災害種別が明確
    3. キャプションが詳細（100文字以上）
    """
    print(f"\n🔄 Selecting candidate blocks for cause-mitigation QA...")
    print(f"   Total blocks: {len(blocks)}")
    
    extractor = ConceptExtractor()
    scored_blocks = []
    
    for block in blocks:
        caption = block.get('caption', '')
        
        if len(caption) < 80:
            continue
        
        score = 0
        
        # 災害種別が明確か
        disaster_type = extractor.extract_disaster_type(caption)
        if disaster_type:
            score += 10
        else:
            continue  # 災害種別不明はスキップ
        
        # 要因分析キーワードを含むか
        if extractor.has_cause_keywords(caption):
            score += 20
        
        # キャプション長（詳細度）
        if len(caption) > 150:
            score += 5
        if len(caption) > 200:
            score += 5
        
        scored_blocks.append((score, block))
    
    # スコアでソート
    scored_blocks.sort(key=lambda x: x[0], reverse=True)
    
    # 上位を選択
    selected = [block for score, block in scored_blocks[:target_count]]
    
    # 災害種別分布
    disaster_dist = {}
    for block in selected:
        caption = block.get('caption', '')
        disaster_type = extractor.extract_disaster_type(caption)
        if disaster_type:
            disaster_dist[disaster_type] = disaster_dist.get(disaster_type, 0) + 1
    
    print(f"   ✅ Selected {len(selected)} candidate blocks")
    print(f"      Disaster type distribution: {disaster_dist}")
    
    return selected


def generate_cause_mitigation_qa_dataset(blocks_path: Path,
                                         output_path: Path,
                                         target_count: int = 100,
                                         candidate_count: int = 150) -> List[CauseMitigationQA]:
    """要因→教訓→防災対策QAデータセットを生成"""
    print(f"\n{'='*60}")
    print("Cause → Lesson → Mitigation QA Generation")
    print(f"{'='*60}")
    print(f"   Target: {target_count} QA pairs")
    print(f"   Candidates: {candidate_count} blocks")
    
    # ブロックを読み込み
    pages = []
    with open(blocks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    
    # 図表ブロックを収集
    figure_blocks = []
    for page in pages:
        pdf_name = page.get('pdf_name', '')
        for fig in page.get('figure_blocks', []):
            if fig.get('caption') and len(fig.get('caption', '')) > 50:
                fig['pdf_name'] = pdf_name
                figure_blocks.append(fig)
    
    print(f"   Found {len(figure_blocks)} figure blocks with captions")
    
    # 候補ブロックを選択
    candidates = select_candidate_blocks(figure_blocks, target_count=candidate_count)
    
    # QA生成
    generator = CauseMitigationQAGenerator()
    qa_pairs = []
    
    for i, block in enumerate(candidates, 1):
        print(f"   Generating QA {i}/{len(candidates)}... (Current: {len(qa_pairs)} valid)", end='\r')
        
        qa = generator.generate_qa(block)
        
        if qa:
            qa_pairs.append(qa)
        
        # 目標数に達したら終了
        if len(qa_pairs) >= target_count:
            print(f"\n   ✅ Reached target count: {target_count}")
            break
        
        time.sleep(0.5)  # レート制限
    
    print(f"\n   ✅ Generated {len(qa_pairs)} cause-mitigation QA pairs")
    
    # 統計情報
    disaster_dist = {}
    avg_answer_length = 0
    
    for qa in qa_pairs:
        disaster_dist[qa.disaster_type] = disaster_dist.get(qa.disaster_type, 0) + 1
        avg_answer_length += len(qa.answer)
    
    avg_answer_length = avg_answer_length / len(qa_pairs) if qa_pairs else 0
    
    print(f"\n   📊 Statistics:")
    print(f"      Disaster types: {disaster_dist}")
    print(f"      Average answer length: {avg_answer_length:.1f} chars")
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(qa) for qa in qa_pairs], f, ensure_ascii=False, indent=2)
    
    print(f"\n   💾 Saved to: {output_path}")
    
    return qa_pairs


def main():
    parser = argparse.ArgumentParser(description="Cause-Mitigation QA Generation")
    parser.add_argument("--blocks", type=str,
                       default="experiments_v070/indices/layout_blocks_captioned.jsonl",
                       help="Path to captioned blocks")
    parser.add_argument("--output", type=str,
                       default="experiments_v070/indices/qa_cause_mitigation.json",
                       help="Output path for generated QA")
    parser.add_argument("--target-count", type=int, default=100,
                       help="Target number of QA pairs to generate")
    parser.add_argument("--candidate-count", type=int, default=150,
                       help="Number of candidate blocks to select")
    
    args = parser.parse_args()
    
    qa_pairs = generate_cause_mitigation_qa_dataset(
        blocks_path=Path(args.blocks),
        output_path=Path(args.output),
        target_count=args.target_count,
        candidate_count=args.candidate_count
    )
    
    print(f"\n{'='*60}")
    print(f"✨ Generated {len(qa_pairs)} cause-mitigation QA pairs!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
