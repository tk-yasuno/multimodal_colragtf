#!/usr/bin/env python3
"""
Analyze 12 demo results for diversity and robustness analysis
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_demo_results():
    indices_dir = Path("experiments_v070/indices")
    
    results_summary = []
    score_stats = defaultdict(list)
    disaster_types = defaultdict(int)
    
    for i in range(1, 13):
        json_file = list(indices_dir.glob(f"demo_result_{i:02d}_*.json"))
        
        if not json_file:
            print(f"⚠ File not found for image {i:02d}")
            continue
        
        json_file = json_file[0]
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract disaster type from image description
        description = data["image_description"]
        
        # Classify disaster type
        disaster_type = "Unknown"
        if "地震" in description:
            disaster_type = "地震"
        elif "津波" in description:
            disaster_type = "津波"
        elif "洪水" in description or "水流" in description or "浸水" in description:
            disaster_type = "洪水"
        elif "土砂" in description or "崩" in description:
            disaster_type = "土砂災害"
        elif "噴火" in description or "火山" in description:
            disaster_type = "噴火"
        elif "台風" in description or "大雨" in description:
            disaster_type = "台風/豪雨"
        
        disaster_types[disaster_type] += 1
        
        # Extract retrieval scores
        retrieval_results = data["retrieval_results"]
        
        if retrieval_results:
            top1_score = retrieval_results[0]["score"]
            top1_text_score = retrieval_results[0]["text_score"]
            top1_triple_score = retrieval_results[0]["triple_score"]
            
            avg_score = sum(r["score"] for r in retrieval_results) / len(retrieval_results)
            avg_text_score = sum(r["text_score"] for r in retrieval_results) / len(retrieval_results)
            avg_triple_score = sum(r["triple_score"] for r in retrieval_results) / len(retrieval_results)
            
            score_stats["top1_score"].append(top1_score)
            score_stats["top1_text_score"].append(top1_text_score)
            score_stats["top1_triple_score"].append(top1_triple_score)
            score_stats["avg_score"].append(avg_score)
            score_stats["avg_text_score"].append(avg_text_score)
            score_stats["avg_triple_score"].append(avg_triple_score)
        else:
            top1_score = 0.0
            avg_score = 0.0
            top1_text_score = 0.0
            top1_triple_score = 0.0
        
        # Count lesson length
        lessons = data["lessons_learned"]
        lesson_length = len(lessons)
        
        results_summary.append({
            "id": i,
            "filename": json_file.stem.replace("demo_result_", ""),
            "disaster_type": disaster_type,
            "top1_score": top1_score,
            "top1_text_score": top1_text_score,
            "top1_triple_score": top1_triple_score,
            "avg_score": avg_score,
            "lesson_length": lesson_length
        })
    
    # Print summary table
    print("=" * 120)
    print("12-Image Demo Results Summary")
    print("=" * 120)
    print(f"{'ID':<4} {'Disaster Type':<15} {'Top-1 Score':<12} {'Text':<10} {'Triple':<10} {'Avg Score':<12} {'Lesson Len':<12}")
    print("-" * 120)
    
    for result in results_summary:
        print(f"{result['id']:<4} {result['disaster_type']:<15} {result['top1_score']:<12.4f} {result['top1_text_score']:<10.4f} {result['top1_triple_score']:<10.4f} {result['avg_score']:<12.4f} {result['lesson_length']:<12}")
    
    print("-" * 120)
    
    # Statistics
    print("\n" + "=" * 80)
    print("Statistical Summary")
    print("=" * 80)
    
    if score_stats["top1_score"]:
        print(f"\nTop-1 Score:")
        print(f"  Mean: {sum(score_stats['top1_score'])/len(score_stats['top1_score']):.4f}")
        print(f"  Min:  {min(score_stats['top1_score']):.4f}")
        print(f"  Max:  {max(score_stats['top1_score']):.4f}")
        
        print(f"\nTop-1 Text Score:")
        print(f"  Mean: {sum(score_stats['top1_text_score'])/len(score_stats['top1_text_score']):.4f}")
        
        print(f"\nTop-1 Triple Score:")
        print(f"  Mean: {sum(score_stats['top1_triple_score'])/len(score_stats['top1_triple_score']):.4f}")
        
        print(f"\nAverage Score (Top-5):")
        print(f"  Mean: {sum(score_stats['avg_score'])/len(score_stats['avg_score']):.4f}")
        print(f"  Min:  {min(score_stats['avg_score']):.4f}")
        print(f"  Max:  {max(score_stats['avg_score']):.4f}")
    
    print(f"\nDisaster Type Distribution:")
    for dtype, count in sorted(disaster_types.items(), key=lambda x: -x[1]):
        print(f"  {dtype:<15} {count:>2} ({count/len(results_summary)*100:.1f}%)")
    
    print(f"\nTotal processed: {len(results_summary)} images")
    
    # Save to JSON
    output = {
        "summary": results_summary,
        "statistics": {
            "top1_score": {
                "mean": sum(score_stats['top1_score'])/len(score_stats['top1_score']) if score_stats['top1_score'] else 0,
                "min": min(score_stats['top1_score']) if score_stats['top1_score'] else 0,
                "max": max(score_stats['top1_score']) if score_stats['top1_score'] else 0
            },
            "disaster_types": dict(disaster_types)
        }
    }
    
    with open("experiments_v070/indices/demo_12images_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Analysis saved to: experiments_v070/indices/demo_12images_analysis.json")

if __name__ == "__main__":
    analyze_demo_results()
