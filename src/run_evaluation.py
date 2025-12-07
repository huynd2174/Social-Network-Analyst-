"""
Script để đánh giá chatbot trên evaluation dataset

Chạy: python src/run_evaluation.py
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import KpopChatbot
from chatbot.comparison import ChatbotComparison


def main():
    """Chạy đánh giá chatbot trên evaluation dataset."""
    print("\n" + "="*70)
    print("  📊 ĐÁNH GIÁ CHATBOT TRÊN EVALUATION DATASET")
    print("="*70)
    
    # Check dataset
    dataset_path = "data/evaluation_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset không tồn tại: {dataset_path}")
        print("   Chạy: python src/run_chatbot.py --mode eval")
        return
    
    # Load dataset info
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get('questions', [])
    metadata = data.get('metadata', {})
    
    print(f"\n📂 Dataset: {dataset_path}")
    print(f"   ✅ Tổng số câu hỏi: {len(questions)}")
    print(f"   ✅ Yêu cầu: ≥ 2000 câu hỏi")
    print(f"   ✅ Kết quả: {'✅ ĐẠT' if len(questions) >= 2000 else '❌ CHƯA ĐẠT'}")
    
    print(f"\n   Phân bố theo số hop:")
    for hop, count in metadata.get('by_hops', {}).items():
        print(f"      - {hop}-hop: {count} câu")
    
    print(f"\n   Phân bố theo loại:")
    for qtype, count in metadata.get('by_type', {}).items():
        print(f"      - {qtype}: {count} câu")
    
    # Initialize chatbot
    print(f"\n🔄 Đang khởi tạo chatbot...")
    chatbot = KpopChatbot(verbose=False)  # Set verbose=False để không in quá nhiều
    
    # Initialize comparison (chỉ cần để dùng evaluate_kpop_chatbot)
    comparison = ChatbotComparison(kpop_chatbot=chatbot)
    
    # Ask user how many questions to evaluate
    print(f"\n💡 Bạn muốn đánh giá bao nhiêu câu hỏi?")
    print(f"   - Nhấn Enter để đánh giá TẤT CẢ ({len(questions)} câu) - có thể mất nhiều thời gian")
    print(f"   - Hoặc nhập số (ví dụ: 100, 500, 1000)")
    
    try:
        user_input = input("\n   Số câu hỏi (Enter = tất cả): ").strip()
        if user_input:
            max_questions = int(user_input)
            if max_questions <= 0 or max_questions > len(questions):
                print(f"   ⚠️  Số không hợp lệ, dùng tất cả {len(questions)} câu")
                max_questions = None
        else:
            max_questions = None
    except ValueError:
        print(f"   ⚠️  Input không hợp lệ, dùng tất cả {len(questions)} câu")
        max_questions = None
    
    if max_questions:
        print(f"\n🔄 Đang đánh giá trên {max_questions} câu hỏi...")
    else:
        print(f"\n🔄 Đang đánh giá trên TẤT CẢ {len(questions)} câu hỏi...")
        print(f"   ⚠️  Có thể mất nhiều thời gian (ước tính: {len(questions) * 2 / 60:.1f} phút)")
    
    # Run evaluation
    start_time = datetime.now()
    result = comparison.evaluate_kpop_chatbot(
        questions,
        max_questions=max_questions
    )
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print results
    print("\n" + "="*70)
    print("  📊 KẾT QUẢ ĐÁNH GIÁ")
    print("="*70)
    
    print(f"\n✅ Chatbot: {result.chatbot_name}")
    print(f"   📈 Accuracy: {result.accuracy:.2%} ({result.correct}/{result.total})")
    print(f"   ⏱️  Thời gian trung bình: {result.avg_response_time:.2f}s/câu")
    print(f"   ⏱️  Tổng thời gian: {duration/60:.1f} phút ({duration:.0f} giây)")
    
    if result.accuracy_by_hops:
        print(f"\n   📊 Accuracy theo số hop:")
        for hop in sorted(result.accuracy_by_hops.keys(), key=int):
            acc = result.accuracy_by_hops[hop]
            print(f"      - {hop}-hop: {acc:.2%}")
    
    if result.accuracy_by_type:
        print(f"\n   📊 Accuracy theo loại câu hỏi:")
        for qtype, acc in result.accuracy_by_type.items():
            print(f"      - {qtype}: {acc:.2%}")
    
    if result.accuracy_by_category:
        print(f"\n   📊 Accuracy theo category:")
        for category, acc in result.accuracy_by_category.items():
            print(f"      - {category}: {acc:.2%}")
    
    # Save results
    output_path = "data/evaluation_results.json"
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "chatbot_name": result.chatbot_name,
        "total_questions": result.total,
        "correct": result.correct,
        "accuracy": result.accuracy,
        "avg_response_time": result.avg_response_time,
        "total_time_seconds": duration,
        "accuracy_by_hops": result.accuracy_by_hops,
        "accuracy_by_type": result.accuracy_by_type,
        "accuracy_by_category": result.accuracy_by_category,
        "errors": result.errors[:20] if result.errors else []  # Save first 20 errors
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Kết quả đã được lưu vào: {output_path}")
    
    # Show some errors if any
    if result.errors:
        print(f"\n⚠️  Một số lỗi (hiển thị 5 lỗi đầu):")
        for i, error in enumerate(result.errors[:5], 1):
            print(f"   {i}. {error}")
        if len(result.errors) > 5:
            print(f"   ... và {len(result.errors) - 5} lỗi khác (xem trong {output_path})")
    
    print("\n" + "="*70)
    print("  ✅ ĐÁNH GIÁ HOÀN TẤT!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

