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
CoLRAG-TF v0.7.0 - Phase 7b: マルチホップQA生成器

2-hop質問を生成して、複数の災害事例・フェーズをまたいだ推論能力を評価します。

戦略:
1. 災害種別でグループ化 (地震, 洪水, 土砂災害, 台風, 噴火)
2. フェーズでグループ化 (被害把握, 救命活動, 応急復旧, 復興事業)
3. ペアリング:
   - 同じ災害種別の異なる事例 (事例A vs 事例B)
   - 異なるフェーズの関連図表 (被害状況 → 応急復旧)
   - 異なる災害の類似フェーズ (地震避難 vs 洪水避難)

Usage:
    .venv-coltf\\Scripts\\python.exe experiments_v070\\07b_generate_multihop_qa.py --target-count 200
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import random
import time
import re

try:
    import requests
    print("✅ requests imported")
except ImportError as e:
    print(f"❌ requests not installed: {e}")
    sys.exit(1)


@dataclass
class MultiHopQA:
    """マルチホップQA質問 (2-hop)"""
    question_id: str
    question: str
    answer: str
    question_type: str  # "disaster_comparison", "phase_transition", "cross_disaster"
    block_ids: List[str] = field(default_factory=list)  # 2つのブロックID
    hop_count: int = 2
    requires_image: bool = True
    difficulty: str = "hard"
    
    # メタデータ
    disaster_types: List[str] = field(default_factory=list)  # ["地震", "洪水"]
    phases: List[str] = field(default_factory=list)  # ["被害把握", "応急復旧"]


# 災害種別キーワード
DISASTER_KEYWORDS = {
    "地震": ["地震", "震災", "余震", "液状化", "耐震"],
    "洪水": ["洪水", "豪雨", "氾濫", "水害", "浸水", "堤防"],
    "土砂災害": ["土砂", "崩壊", "がけ崩れ", "斜面", "地すべり"],
    "台風": ["台風", "暴風", "高潮"],
    "津波": ["津波", "津波被害"],
    "噴火": ["噴火", "火山", "降灰"]
}

# フェーズキーワード
PHASE_KEYWORDS = {
    "被害把握": ["被害", "被災", "損壊", "倒壊", "全壊", "半壊", "犠牲者"],
    "救命活動": ["救助", "救命", "避難", "避難所", "救援"],
    "応急復旧": ["応急", "復旧", "仮設", "ライフライン", "道路啓開"],
    "復興事業": ["復興", "再建", "まちづくり", "都市計画", "移転"]
}


class ConceptExtractor:
    """キャプションから概念（災害種別、フェーズ）を抽出"""
    
    @staticmethod
    def extract_disaster_type(text: str) -> Optional[str]:
        """災害種別を抽出"""
        for disaster_type, keywords in DISASTER_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return disaster_type
        return None
    
    @staticmethod
    def extract_phase(text: str) -> Optional[str]:
        """フェーズを抽出"""
        for phase, keywords in PHASE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return phase
        return None
    
    @staticmethod
    def extract_location(text: str) -> Optional[str]:
        """地名を抽出（簡易版）"""
        # 都道府県名パターン
        prefectures = [
            "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島",
            "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
            "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜",
            "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
            "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口",
            "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎",
            "熊本", "大分", "宮崎", "鹿児島", "沖縄"
        ]
        for pref in prefectures:
            if pref in text:
                return pref
        return None


class MultiHopQAGenerator:
    """2-hopマルチホップQA生成器"""
    
    # 質問タイプ別プロンプトテンプレート
    DISASTER_COMPARISON_PROMPT = """以下は2つの災害事例の図表キャプションです。

【事例A】
{caption_a}

【事例B】
{caption_b}

【タスク】
この2つの災害事例を比較し、以下の形式で2-hop質問と回答を1つ生成してください：

質問: 事例Aと事例Bを比較して、今後の{disaster_type}災害への備えとして何が重要ですか？
回答: （2つの事例の共通点・相違点を踏まえ、具体的な教訓や対策を150文字程度で記述）

生成してください："""

    PHASE_TRANSITION_PROMPT = """以下は災害対応の異なるフェーズの図表キャプションです。

【{phase_a}】
{caption_a}

【{phase_b}】
{caption_b}

【タスク】
この2つのフェーズの情報を統合し、以下の形式で2-hop質問と回答を1つ生成してください：

質問: {phase_a}から{phase_b}へ移行する際、どのような対応手順が重要ですか？
回答: （2つのフェーズの情報を統合し、具体的な対応手順のコツを150文字程度で記述）

生成してください："""

    CROSS_DISASTER_PROMPT = """以下は異なる種類の災害事例の図表キャプションです。

【{disaster_a}災害】
{caption_a}

【{disaster_b}災害】
{caption_b}

【タスク】
この2つの異なる災害事例から、以下の形式で2-hop質問と回答を1つ生成してください：

質問: {disaster_a}災害と{disaster_b}災害の経験を統合すると、複合災害への備えとして何が重要ですか？
回答: （2つの異なる災害の教訓を統合し、共通する対策や複合災害への備えを150文字程度で記述）

生成してください："""

    CAUSE_MITIGATION_PROMPT = """以下は災害事例の図表キャプションです。

【災害事例】
{caption}

【タスク】
この災害事例を振り返り、以下の形式で2-hop質問と回答を1つ生成してください：

質問: この{disaster_type}災害が大規模化した要因と教訓を踏まえ、今後の防災・減災の取り組みとしてどんな課題がありますか？
回答: （災害の要因分析→得られた教訓→具体的な防災・減災対策の課題、という流れで200文字程度で記述。要因には「○○が不足していた」「××の準備が遅れた」などを含め、対策には「△△の事前整備」「◇◇の体制強化」などを明記すること）

生成してください："""

    def __init__(self, model_name: str = "qwen2.5:7b-instruct-q4_k_m",
                 base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.extractor = ConceptExtractor()
        print(f"✅ MultiHopQAGenerator initialized: {model_name}")
    
    def generate_multihop_qa(self,
                             block_a: Dict[str, Any],
                             block_b: Optional[Dict[str, Any]],
                             qa_type: str) -> Optional[MultiHopQA]:
        """2つのブロック（または1つ）から2-hop QAを生成"""
        
        caption_a = block_a.get('caption', '')
        
        if len(caption_a) < 50:
            return None
        
        # cause_and_mitigationタイプは1ブロックのみ使用
        if qa_type == "cause_and_mitigation":
            disaster_a = self.extractor.extract_disaster_type(caption_a)
            
            if not disaster_a:
                return None
            
            prompt = self.CAUSE_MITIGATION_PROMPT.format(
                caption=caption_a,
                disaster_type=disaster_a
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
                            "num_predict": 512
                        }
                    },
                    timeout=90
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '').strip()
                    
                    # QAペアをパース
                    question, answer = self._parse_qa_response(response_text)
                    
                    if question and answer and len(answer) >= 100:
                        return MultiHopQA(
                            question_id=f"{block_a.get('block_id', '')}_cause_mitigation",
                            question=question,
                            answer=answer,
                            question_type=qa_type,
                            block_ids=[block_a.get('block_id', '')],
                            hop_count=2,
                            requires_image=True,
                            difficulty="hard",
                            disaster_types=[disaster_a],
                            phases=["不明"]
                        )
            except Exception as e:
                print(f"\n⚠️ QA generation error: {e}")
            
            return None
        
        # 2ブロック使用タイプ
        if not block_b:
            return None
        
        caption_b = block_b.get('caption', '')
        
        if len(caption_b) < 50:
            return None
        
        # 災害種別・フェーズを抽出
        disaster_a = self.extractor.extract_disaster_type(caption_a)
        disaster_b = self.extractor.extract_disaster_type(caption_b)
        phase_a = self.extractor.extract_phase(caption_a)
        phase_b = self.extractor.extract_phase(caption_b)
        
        # プロンプトを選択
        if qa_type == "disaster_comparison" and disaster_a:
            prompt = self.DISASTER_COMPARISON_PROMPT.format(
                caption_a=caption_a,
                caption_b=caption_b,
                disaster_type=disaster_a
            )
        elif qa_type == "phase_transition" and phase_a and phase_b:
            prompt = self.PHASE_TRANSITION_PROMPT.format(
                phase_a=phase_a,
                phase_b=phase_b,
                caption_a=caption_a,
                caption_b=caption_b
            )
        elif qa_type == "cross_disaster" and disaster_a and disaster_b:
            prompt = self.CROSS_DISASTER_PROMPT.format(
                disaster_a=disaster_a,
                disaster_b=disaster_b,
                caption_a=caption_a,
                caption_b=caption_b
            )
        else:
            return None
        
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
                        "num_predict": 512
                    }
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # QAペアをパース
                question, answer = self._parse_qa_response(response_text)
                
                if question and answer and len(answer) >= 50:
                    return MultiHopQA(
                        question_id=f"{block_a.get('block_id', '')}_{block_b.get('block_id', '')}_multihop",
                        question=question,
                        answer=answer,
                        question_type=qa_type,
                        block_ids=[block_a.get('block_id', ''), block_b.get('block_id', '')],
                        hop_count=2,
                        requires_image=True,
                        difficulty="hard",
                        disaster_types=[disaster_a, disaster_b] if disaster_a and disaster_b else [disaster_a or disaster_b or "不明"],
                        phases=[phase_a, phase_b] if phase_a and phase_b else [phase_a or phase_b or "不明"]
                    )
        except Exception as e:
            print(f"\n⚠️ QA generation error: {e}")
        
        return None
    
    def _parse_qa_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """LLM応答からQAペアをパース"""
        lines = response_text.strip().split('\n')
        question = None
        answer = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('質問:') or line.startswith('質問：'):
                question = line.split(':', 1)[-1].split('：', 1)[-1].strip()
            elif line.startswith('回答:') or line.startswith('回答：'):
                answer = line.split(':', 1)[-1].split('：', 1)[-1].strip()
        
        return question, answer


def pair_blocks_for_multihop(blocks: List[Dict[str, Any]],
                              target_count: int = 300) -> List[Tuple[Dict, Optional[Dict], str]]:
    """
    ブロックをペアリングして2-hop質問候補を生成
    
    Returns:
        List[(block_a, block_b_or_None, qa_type)]
        block_b_or_Noneは、cause_and_mitigationタイプの場合Noneになる
    """
    print(f"\n🔄 Pairing blocks for multihop QA...")
    print(f"   Total blocks: {len(blocks)}")
    
    extractor = ConceptExtractor()
    
    # ブロックをメタデータで分類
    blocks_by_disaster = {}
    blocks_by_phase = {}
    blocks_by_pdf = {}
    
    for block in blocks:
        caption = block.get('caption', '')
        pdf_name = block.get('pdf_name', '')
        
        # 災害種別で分類
        disaster_type = extractor.extract_disaster_type(caption)
        if disaster_type:
            if disaster_type not in blocks_by_disaster:
                blocks_by_disaster[disaster_type] = []
            blocks_by_disaster[disaster_type].append(block)
        
        # フェーズで分類
        phase = extractor.extract_phase(caption)
        if phase:
            if phase not in blocks_by_phase:
                blocks_by_phase[phase] = []
            blocks_by_phase[phase].append(block)
        
        # PDFで分類
        if pdf_name:
            if pdf_name not in blocks_by_pdf:
                blocks_by_pdf[pdf_name] = []
            blocks_by_pdf[pdf_name].append(block)
    
    print(f"   Disaster types: {list(blocks_by_disaster.keys())}")
    print(f"   Phases: {list(blocks_by_phase.keys())}")
    print(f"   PDFs: {len(blocks_by_pdf)}")
    
    pairs = []
    
    # 1. 災害比較ペア (同じ災害種別の異なるPDF)
    for disaster_type, disaster_blocks in blocks_by_disaster.items():
        # PDFでグループ化
        pdf_groups = {}
        for block in disaster_blocks:
            pdf = block.get('pdf_name', '')
            if pdf not in pdf_groups:
                pdf_groups[pdf] = []
            pdf_groups[pdf].append(block)
        
        # 異なるPDFからペアを作成
        pdf_list = list(pdf_groups.keys())
        for i in range(len(pdf_list)):
            for j in range(i + 1, min(i + 3, len(pdf_list))):  # 最大2つのPDFとペア
                pdf_a, pdf_b = pdf_list[i], pdf_list[j]
                blocks_a = pdf_groups[pdf_a]
                blocks_b = pdf_groups[pdf_b]
                
                # ランダムにペアを作成
                for _ in range(min(5, len(blocks_a), len(blocks_b))):
                    block_a = random.choice(blocks_a)
                    block_b = random.choice(blocks_b)
                    pairs.append((block_a, block_b, "disaster_comparison"))
    
    # 2. フェーズ遷移ペア (異なるフェーズの同じPDF)
    phase_order = ["被害把握", "救命活動", "応急復旧", "復興事業"]
    for pdf_name, pdf_blocks in blocks_by_pdf.items():
        # フェーズでグループ化
        phase_groups = {}
        for block in pdf_blocks:
            caption = block.get('caption', '')
            phase = extractor.extract_phase(caption)
            if phase:
                if phase not in phase_groups:
                    phase_groups[phase] = []
                phase_groups[phase].append(block)
        
        # 隣接フェーズからペアを作成
        for i in range(len(phase_order) - 1):
            phase_a = phase_order[i]
            phase_b = phase_order[i + 1]
            
            if phase_a in phase_groups and phase_b in phase_groups:
                blocks_a = phase_groups[phase_a]
                blocks_b = phase_groups[phase_b]
                
                for _ in range(min(3, len(blocks_a), len(blocks_b))):
                    block_a = random.choice(blocks_a)
                    block_b = random.choice(blocks_b)
                    pairs.append((block_a, block_b, "phase_transition"))
    
    # 3. 異種災害ペア (異なる災害種別、同じフェーズ)
    disaster_types = list(blocks_by_disaster.keys())
    for i in range(len(disaster_types)):
        for j in range(i + 1, min(i + 3, len(disaster_types))):
            disaster_a = disaster_types[i]
            disaster_b = disaster_types[j]
            
            blocks_a = blocks_by_disaster[disaster_a]
            blocks_b = blocks_by_disaster[disaster_b]
            
            # 同じフェーズのものを優先
            for phase in PHASE_KEYWORDS.keys():
                phase_blocks_a = [b for b in blocks_a if extractor.extract_phase(b.get('caption', '')) == phase]
                phase_blocks_b = [b for b in blocks_b if extractor.extract_phase(b.get('caption', '')) == phase]
                
                if phase_blocks_a and phase_blocks_b:
                    for _ in range(min(3, len(phase_blocks_a), len(phase_blocks_b))):
                        block_a = random.choice(phase_blocks_a)
                        block_b = random.choice(phase_blocks_b)
                        pairs.append((block_a, block_b, "cross_disaster"))
    
    # 4. 要因と防災対策ペア (単一ブロック、被害や復興フェーズ優先)
    # 被害把握や復興事業フェーズのブロックから要因分析に適したものを選択
    cause_mitigation_candidates = []
    for disaster_type, disaster_blocks in blocks_by_disaster.items():
        # 被害規模が大きそうなキーワードを含むブロックを優先
        severity_keywords = ["全壊", "半壊", "死者", "行方不明", "被災", "甚大", "大規模", "壊滅"]
        for block in disaster_blocks:
            caption = block.get('caption', '')
            # 被害規模や復興フェーズに関連するキーワードがあれば候補に
            if any(kw in caption for kw in severity_keywords):
                cause_mitigation_candidates.append(block)
            # または、被害把握・復興事業フェーズのブロック
            phase = extractor.extract_phase(caption)
            if phase in ["被害把握", "復興事業"]:
                cause_mitigation_candidates.append(block)
    
    # 重複を削除
    cause_mitigation_candidates = list({b.get('block_id'): b for b in cause_mitigation_candidates}.values())
    
    # ランダムサンプリング（目標数の30%程度）
    cause_mitigation_count = min(len(cause_mitigation_candidates), int(target_count * 0.3))
    sampled_cause_mitigation = random.sample(cause_mitigation_candidates, cause_mitigation_count)
    
    for block in sampled_cause_mitigation:
        pairs.append((block, None, "cause_and_mitigation"))
    
    # ランダムシャッフル
    random.shuffle(pairs)
    
    # 目標数にトリミング
    pairs = pairs[:target_count]
    
    print(f"   ✅ Created {len(pairs)} candidate pairs")
    print(f"      - disaster_comparison: {sum(1 for p in pairs if p[2] == 'disaster_comparison')}")
    print(f"      - phase_transition: {sum(1 for p in pairs if p[2] == 'phase_transition')}")
    print(f"      - cross_disaster: {sum(1 for p in pairs if p[2] == 'cross_disaster')}")
    print(f"      - cause_and_mitigation: {sum(1 for p in pairs if p[2] == 'cause_and_mitigation')}")
    
    return pairs


def generate_multihop_qa_dataset(blocks_path: Path,
                                  output_path: Path,
                                  target_count: int = 200,
                                  candidate_count: int = 300) -> List[MultiHopQA]:
    """マルチホップQAデータセットを生成"""
    print(f"\n{'='*60}")
    print("Multi-Hop QA Dataset Generation")
    print(f"{'='*60}")
    print(f"   Target: {target_count} QA pairs")
    print(f"   Candidates: {candidate_count} pairs")
    
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
                fig['pdf_name'] = pdf_name  # PDF名を追加
                figure_blocks.append(fig)
    
    print(f"   Found {len(figure_blocks)} figure blocks with captions")
    
    # ペアリング
    pairs = pair_blocks_for_multihop(figure_blocks, target_count=candidate_count)
    
    # QA生成
    generator = MultiHopQAGenerator()
    qa_pairs = []
    
    for i, (block_a, block_b, qa_type) in enumerate(pairs, 1):
        print(f"   Generating QA {i}/{len(pairs)}... (Current: {len(qa_pairs)} valid)", end='\r')
        
        qa = generator.generate_multihop_qa(block_a, block_b, qa_type)
        
        if qa:
            qa_pairs.append(qa)
        
        # 目標数に達したら終了
        if len(qa_pairs) >= target_count:
            print(f"\n   ✅ Reached target count: {target_count}")
            break
        
        time.sleep(0.5)  # レート制限
    
    print(f"\n   ✅ Generated {len(qa_pairs)} multi-hop QA pairs")
    
    # 統計情報
    disaster_dist = {}
    phase_dist = {}
    type_dist = {}
    
    for qa in qa_pairs:
        # 災害種別分布
        for disaster in qa.disaster_types:
            disaster_dist[disaster] = disaster_dist.get(disaster, 0) + 1
        
        # フェーズ分布
        for phase in qa.phases:
            phase_dist[phase] = phase_dist.get(phase, 0) + 1
        
        # 質問タイプ分布
        type_dist[qa.question_type] = type_dist.get(qa.question_type, 0) + 1
    
    print(f"\n   📊 Statistics:")
    print(f"      Disaster types: {disaster_dist}")
    print(f"      Phases: {phase_dist}")
    print(f"      Question types: {type_dist}")
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(qa) for qa in qa_pairs], f, ensure_ascii=False, indent=2)
    
    print(f"\n   💾 Saved to: {output_path}")
    
    return qa_pairs


def main():
    parser = argparse.ArgumentParser(description="Multi-Hop QA Generation")
    parser.add_argument("--blocks", type=str,
                       default="experiments_v070/indices/layout_blocks_captioned.jsonl",
                       help="Path to captioned blocks")
    parser.add_argument("--output", type=str,
                       default="experiments_v070/indices/qa_multihop.json",
                       help="Output path for generated QA")
    parser.add_argument("--target-count", type=int, default=200,
                       help="Target number of QA pairs to generate")
    parser.add_argument("--candidate-count", type=int, default=300,
                       help="Number of candidate pairs to generate")
    
    args = parser.parse_args()
    
    qa_pairs = generate_multihop_qa_dataset(
        blocks_path=Path(args.blocks),
        output_path=Path(args.output),
        target_count=args.target_count,
        candidate_count=args.candidate_count
    )
    
    print(f"\n{'='*60}")
    print(f"✨ Generated {len(qa_pairs)} multi-hop QA pairs!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
