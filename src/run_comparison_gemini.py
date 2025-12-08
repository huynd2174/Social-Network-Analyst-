"""
Script để chạy comparison với Gemini

Sử dụng API key được cung cấp để so sánh chatbot với Gemini.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import KpopChatbot, ChatbotComparison, EvaluationDatasetGenerator

# ⚠️ BẢO MẬT: KHÔNG hardcode API key. Lấy từ env hoặc tham số.
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def main():
    """Run comparison with Gemini."""
    print("\n" + "="*70)
    print("  🔬 CHATBOT COMPARISON: K-pop Chatbot vs Gemini")
    print("="*70)
    
    # Set API key nếu có
    if not GEMINI_API_KEY:
        print("❌ Thiếu GOOGLE_API_KEY. Đặt biến môi trường GOOGLE_API_KEY hoặc truyền qua tham số.")
        print("   PowerShell: $env:GOOGLE_API_KEY='YOUR_KEY'")
        print("   CMD: set GOOGLE_API_KEY=YOUR_KEY")
        return
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    
    print("\n🔄 Initializing K-pop Chatbot...")
    chatbot = KpopChatbot(verbose=True)
    
    # Check if dataset exists
    dataset_path = "data/evaluation_dataset.json"
    if not os.path.exists(dataset_path):
        print("\n📝 Generating evaluation dataset first...")
        generator = EvaluationDatasetGenerator()
        generator.generate_full_dataset(output_path=dataset_path)
    
    # Initialize comparison
    print("\n🔄 Initializing Comparison Framework...")
    comparison = ChatbotComparison(
        kpop_chatbot=chatbot,
        google_api_key=GEMINI_API_KEY
    )
    
    # Load dataset
    questions = comparison.load_evaluation_dataset(dataset_path)
    print(f"✅ Loaded {len(questions)} questions from dataset")
    
    # Run comparison (limit to 200 questions for faster testing)
    print("\n🔄 Running comparison (200 questions for testing)...")
    print("   ⚠️  Lưu ý: Comparison có thể mất 5-10 phút")
    
    results = comparison.compare_chatbots(
        questions,
        include_chatgpt=False,  # Không dùng ChatGPT
        include_gemini=True,    # Dùng Gemini
        include_baseline=True,  # Include baseline
        max_questions=200       # Limit để test nhanh
    )
    
    print("\n" + "="*70)
    print("  ✅ COMPARISON COMPLETE!")
    print("="*70)
    print(f"\n📄 Results saved to: data/comparison_results.json")
    print("\n📊 Summary:")
    for chatbot_name, summary in results['summary'].items():
        print(f"  {chatbot_name}:")
        print(f"    - Accuracy: {summary['accuracy']:.2%}")
        print(f"    - Avg Response Time: {summary['avg_response_time']:.2f}s")
        if 'accuracy_by_hops' in summary:
            print(f"    - By Hops: {summary['accuracy_by_hops']}")


if __name__ == "__main__":
    main()





