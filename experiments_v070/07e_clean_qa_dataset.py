"""
QAデータセットから文字化けや品質の低いQAを検出・除外するスクリプト
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class QualityMetrics:
    """QAペアの品質メトリクス"""
    has_mojibake: bool = False
    has_replacement_char: bool = False
    has_excessive_repetition: bool = False
    is_too_short: bool = False
    is_too_long: bool = False
    has_invalid_chars: bool = False
    is_non_japanese: bool = False
    quality_score: float = 1.0
    issues: List[str] = field(default_factory=list)


class QAQualityChecker:
    """QAペアの品質をチェック"""
    
    def __init__(self):
        # 文字化けパターン
        self.replacement_char = '\ufffd'  # �
        self.invalid_chars_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        
        # 繰り返しパターン（同じ文字が5回以上連続）
        self.repetition_pattern = re.compile(r'(.)\1{4,}')
        
        # 品質基準
        self.min_question_length = 10
        self.max_question_length = 500
        self.min_answer_length = 20
        self.max_answer_length = 1000
        
        # 日本語文字の範囲
        self.hiragana_pattern = re.compile(r'[\u3040-\u309f]')
        self.katakana_pattern = re.compile(r'[\u30a0-\u30ff]')
        self.kanji_pattern = re.compile(r'[\u4e00-\u9fff]')
        
        # 中国語簡体字特有の文字（日本語では使わない）
        self.chinese_simplified_chars = set([
            '为', '务', '国', '图', '圆', '场', '坏', '块', '坚', '垒', '执', '扩', 
            '担', '据', '挤', '挥', '损', '换', '报', '护', '抢', '拟',
            '择', '拨', '拥', '拦', '挂', '构', '标', '栋', '树', '档', '检', 
            '业', '丛', '东', '举', '义', '乌', '乐', '习', '乡', '书', '买', 
            '乱', '争', '于', '亏', '云', '亚', '产', '亩', '享', '亲', '亿', 
            '仅', '从', '仑', '仓', '仪', '们', '价', '众', '优', '会', '伛', 
            '伞', '伟', '传', '伤', '伦', '伪', '伫', '体', '余', '佣', '侠', 
            '侣', '侥', '侦', '侧', '侨', '侩', '侪', '促', '俣'
        ])
        
    def check_mojibake(self, text: str) -> Tuple[bool, List[str]]:
        """文字化けをチェック"""
        issues = []
        has_mojibake = False
        
        # 1. Replacement character の検出
        if self.replacement_char in text:
            has_mojibake = True
            count = text.count(self.replacement_char)
            issues.append(f"Replacement character found: {count} times")
        
        # 2. 不正な制御文字の検出
        invalid_matches = self.invalid_chars_pattern.findall(text)
        if invalid_matches:
            has_mojibake = True
            issues.append(f"Invalid control characters found: {len(invalid_matches)}")
        
        # 3. 過度な繰り返しの検出
        repetition_matches = self.repetition_pattern.findall(text)
        if repetition_matches:
            has_mojibake = True
            issues.append(f"Excessive repetition found: {''.join(set(repetition_matches))}")
        
        # 4. 不自然な文字列パターン
        if re.search(r'[^\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\w\s\.,!?;:()「」『』【】。、]', text):
            unusual_chars = re.findall(r'[^\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\w\s\.,!?;:()「」『』【】。、]', text)
            if len(unusual_chars) > 5:
                issues.append(f"Unusual characters found: {len(unusual_chars)} chars")
        
        return has_mojibake, issues
    
    def check_length(self, text: str, text_type: str) -> Tuple[bool, List[str]]:
        """長さをチェック"""
        issues = []
        is_invalid = False
        
        length = len(text)
        
        if text_type == "question":
            if length < self.min_question_length:
                is_invalid = True
                issues.append(f"Question too short: {length} chars (min: {self.min_question_length})")
            elif length > self.max_question_length:
                is_invalid = True
                issues.append(f"Question too long: {length} chars (max: {self.max_question_length})")
        else:  # answer
            if length < self.min_answer_length:
                is_invalid = True
                issues.append(f"Answer too short: {length} chars (min: {self.min_answer_length})")
            elif length > self.max_answer_length:
                is_invalid = True
                issues.append(f"Answer too long: {length} chars (max: {self.max_answer_length})")
        
        return is_invalid, issues
    
    def check_language(self, text: str) -> Tuple[bool, List[str]]:
        """日本語かどうかをチェック"""
        issues = []
        is_non_japanese = False
        
        # 空文字チェック
        if not text or len(text.strip()) == 0:
            return False, []
        
        # ひらがな・カタカナの出現頻度
        hiragana_count = len(self.hiragana_pattern.findall(text))
        katakana_count = len(self.katakana_pattern.findall(text))
        kanji_count = len(self.kanji_pattern.findall(text))
        
        # 総文字数（空白・記号を除く）
        text_clean = re.sub(r'[\s\W\d]', '', text)
        total_chars = len(text_clean)
        
        if total_chars == 0:
            return False, []
        
        # 日本語特有の文字（ひらがな・カタカナ）の割合
        japanese_chars = hiragana_count + katakana_count
        japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
        
        # 中国語簡体字の検出
        chinese_simplified_count = sum(1 for char in text if char in self.chinese_simplified_chars)
        chinese_ratio = chinese_simplified_count / total_chars if total_chars > 0 else 0
        
        # 判定基準
        # 1. ひらがな・カタカナが5%未満 → 日本語ではない可能性
        if japanese_ratio < 0.05 and total_chars > 20:
            is_non_japanese = True
            issues.append(f"Very few Japanese kana: {japanese_ratio*100:.1f}% (hiragana: {hiragana_count}, katakana: {katakana_count})")
        
        # 2. 中国語簡体字が3%以上 → 中国語の可能性
        if chinese_ratio > 0.03:
            is_non_japanese = True
            issues.append(f"Chinese simplified chars detected: {chinese_ratio*100:.1f}% ({chinese_simplified_count} chars)")
        
        # 3. 漢字のみで構成されている（ひらがな・カタカナが全くない）→ 不自然
        if japanese_chars == 0 and kanji_count > 10:
            is_non_japanese = True
            issues.append(f"Only Kanji, no hiragana/katakana (total: {kanji_count} kanji)")
        
        # 4. 英語のみの文章（アルファベットが80%以上）
        ascii_chars = len(re.findall(r'[a-zA-Z]', text))
        ascii_ratio = ascii_chars / len(text) if len(text) > 0 else 0
        if ascii_ratio > 0.8 and len(text) > 20:
            is_non_japanese = True
            issues.append(f"Mostly English text: {ascii_ratio*100:.1f}% ASCII")
        
        return is_non_japanese, issues
    
    def evaluate_qa_pair(self, qa: Dict) -> QualityMetrics:
        """QAペアを評価"""
        metrics = QualityMetrics()
        quality_score = 1.0
        
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        
        # 1. 質問の文字化けチェック
        q_mojibake, q_issues = self.check_mojibake(question)
        if q_mojibake:
            metrics.has_mojibake = True
            metrics.issues.extend([f"[Q] {issue}" for issue in q_issues])
            quality_score -= 0.5
        
        # 2. 回答の文字化けチェック
        a_mojibake, a_issues = self.check_mojibake(answer)
        if a_mojibake:
            metrics.has_mojibake = True
            metrics.issues.extend([f"[A] {issue}" for issue in a_issues])
            quality_score -= 0.5
        
        # 3. Replacement character の特定
        if self.replacement_char in question or self.replacement_char in answer:
            metrics.has_replacement_char = True
        
        # 4. 繰り返しパターンの特定
        if self.repetition_pattern.search(question) or self.repetition_pattern.search(answer):
            metrics.has_excessive_repetition = True
        
        # 5. 質問の長さチェック
        q_length_invalid, q_length_issues = self.check_length(question, "question")
        if q_length_invalid:
            metrics.is_too_short = len(question) < self.min_question_length
            metrics.is_too_long = len(question) > self.max_question_length
            metrics.issues.extend(q_length_issues)
            quality_score -= 0.2
        
        # 6. 回答の長さチェック
        a_length_invalid, a_length_issues = self.check_length(answer, "answer")
        if a_length_invalid:
            metrics.is_too_short = len(answer) < self.min_answer_length
            metrics.is_too_long = len(answer) > self.max_answer_length
            metrics.issues.extend(a_length_issues)
            quality_score -= 0.2
        
        # 7. 質問の言語チェック（日本語かどうか）
        q_non_japanese, q_lang_issues = self.check_language(question)
        if q_non_japanese:
            metrics.is_non_japanese = True
            metrics.issues.extend([f"[Q] {issue}" for issue in q_lang_issues])
            quality_score -= 0.8  # 言語不適合は重大な問題
        
        # 8. 回答の言語チェック（日本語かどうか）
        a_non_japanese, a_lang_issues = self.check_language(answer)
        if a_non_japanese:
            metrics.is_non_japanese = True
            metrics.issues.extend([f"[A] {issue}" for issue in a_lang_issues])
            quality_score -= 0.8  # 言語不適合は重大な問題
        
        metrics.quality_score = max(0.0, quality_score)
        
        return metrics
    
    def filter_dataset(self, qa_list: List[Dict], quality_threshold: float = 0.5) -> Tuple[List[Dict], List[Dict], Dict]:
        """データセットをフィルタリング"""
        clean_qa = []
        rejected_qa = []
        stats = {
            "total": len(qa_list),
            "clean": 0,
            "rejected": 0,
            "mojibake": 0,
            "replacement_char": 0,
            "excessive_repetition": 0,
            "too_short": 0,
            "too_long": 0,
            "non_japanese": 0,
        }
        
        for qa in qa_list:
            metrics = self.evaluate_qa_pair(qa)
            
            if metrics.quality_score >= quality_threshold:
                clean_qa.append(qa)
                stats["clean"] += 1
            else:
                rejected_qa.append({
                    "qa": qa,
                    "metrics": metrics,
                    "reason": "; ".join(metrics.issues)
                })
                stats["rejected"] += 1
                
                # 統計を更新
                if metrics.has_mojibake:
                    stats["mojibake"] += 1
                if metrics.has_replacement_char:
                    stats["replacement_char"] += 1
                if metrics.has_excessive_repetition:
                    stats["excessive_repetition"] += 1
                if metrics.is_too_short:
                    stats["too_short"] += 1
                if metrics.is_too_long:
                    stats["too_long"] += 1
                if metrics.is_non_japanese:
                    stats["non_japanese"] += 1
        
        return clean_qa, rejected_qa, stats


def print_statistics(stats: Dict, dataset_name: str):
    """統計を表示"""
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name}")
    print(f"{'='*60}")
    print(f"   Total QA pairs: {stats['total']}")
    print(f"   ✅ Clean: {stats['clean']} ({stats['clean']/stats['total']*100:.1f}%)")
    print(f"   ❌ Rejected: {stats['rejected']} ({stats['rejected']/stats['total']*100:.1f}%)")
    
    if stats['rejected'] > 0:
        print(f"\n   📋 Rejection Reasons:")
        print(f"      Non-Japanese (Chinese/English): {stats['non_japanese']}")
        print(f"      Mojibake: {stats['mojibake']}")
        print(f"      Replacement char: {stats['replacement_char']}")
        print(f"      Excessive repetition: {stats['excessive_repetition']}")
        print(f"      Too short: {stats['too_short']}")
        print(f"      Too long: {stats['too_long']}")


def main():
    """メイン処理"""
    print("="*60)
    print("QA Dataset Quality Check & Cleaning")
    print("="*60)
    
    # ファイルパス
    indices_dir = Path("experiments_v070/indices")
    
    qa_files = {
        "1-hop (qa_multimodal.json)": indices_dir / "qa_multimodal.json",
        "Multi-hop (qa_multihop.json)": indices_dir / "qa_multihop.json",
        "Cause-Mitigation (qa_cause_mitigation.json)": indices_dir / "qa_cause_mitigation.json",
    }
    
    # 品質チェッカー
    checker = QAQualityChecker()
    
    all_stats = {}
    all_rejected = {}
    
    # 各データセットを処理
    for dataset_name, file_path in qa_files.items():
        print(f"\n🔍 Processing: {dataset_name}")
        
        # データ読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_list = json.load(f)
        
        # フィルタリング
        clean_qa, rejected_qa, stats = checker.filter_dataset(qa_list, quality_threshold=0.5)
        
        # 統計を保存
        all_stats[dataset_name] = stats
        all_rejected[dataset_name] = rejected_qa
        
        # 統計を表示
        print_statistics(stats, dataset_name)
        
        # クリーンなデータを保存
        if stats['rejected'] > 0:
            clean_file = file_path.stem + "_clean.json"
            clean_path = indices_dir / clean_file
            with open(clean_path, 'w', encoding='utf-8') as f:
                json.dump(clean_qa, f, ensure_ascii=False, indent=2)
            print(f"   💾 Clean data saved to: {clean_path.name}")
    
    # 全体統計
    print(f"\n{'='*60}")
    print("📊 Overall Statistics")
    print(f"{'='*60}")
    
    total_qa = sum(s['total'] for s in all_stats.values())
    total_clean = sum(s['clean'] for s in all_stats.values())
    total_rejected = sum(s['rejected'] for s in all_stats.values())
    
    print(f"   Total QA pairs: {total_qa}")
    print(f"   ✅ Clean: {total_clean} ({total_clean/total_qa*100:.1f}%)")
    print(f"   ❌ Rejected: {total_rejected} ({total_rejected/total_qa*100:.1f}%)")
    
    # 拒否されたQAの詳細をファイルに保存
    if total_rejected > 0:
        rejected_file = indices_dir / "qa_rejected_report.json"
        with open(rejected_file, 'w', encoding='utf-8') as f:
            json.dump(all_rejected, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n   📄 Rejected QA report saved to: {rejected_file.name}")
        
        # サンプル表示（最初の5つ）
        print(f"\n{'='*60}")
        print("📋 Sample Rejected QA (first 5)")
        print(f"{'='*60}")
        
        sample_count = 0
        for dataset_name, rejected_list in all_rejected.items():
            if rejected_list and sample_count < 5:
                for item in rejected_list[:min(5-sample_count, len(rejected_list))]:
                    sample_count += 1
                    print(f"\n[{dataset_name}]")
                    print(f"Question ID: {item['qa'].get('question_id', 'N/A')}")
                    print(f"Quality Score: {item['metrics'].quality_score:.2f}")
                    print(f"Issues: {item['reason']}")
                    print(f"Question: {item['qa'].get('question', '')[:100]}...")
                    print(f"Answer: {item['qa'].get('answer', '')[:100]}...")
                    
                    if sample_count >= 5:
                        break


if __name__ == "__main__":
    main()
