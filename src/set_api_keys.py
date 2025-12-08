"""
Helper script to set API keys for chatbot comparison.

Usage:
    python src/set_api_keys.py --google YOUR_GOOGLE_API_KEY
    python src/set_api_keys.py --openai YOUR_OPENAI_API_KEY
    python src/set_api_keys.py --google YOUR_GOOGLE_API_KEY --openai YOUR_OPENAI_API_KEY
"""

import argparse
import os
import sys

def set_google_api_key(key: str):
    """Set Google API key in current session."""
    os.environ["GOOGLE_API_KEY"] = key
    print("✅ GOOGLE_API_KEY đã được set (chỉ trong session hiện tại)")
    print("   💡 Để set vĩnh viễn, dùng:")
    print("      PowerShell: $env:GOOGLE_API_KEY='YOUR_KEY'")
    print("      CMD: set GOOGLE_API_KEY=YOUR_KEY")

def set_openai_api_key(key: str):
    """Set OpenAI API key in current session."""
    os.environ["OPENAI_API_KEY"] = key
    print("✅ OPENAI_API_KEY đã được set (chỉ trong session hiện tại)")
    print("   💡 Để set vĩnh viễn, dùng:")
    print("      PowerShell: $env:OPENAI_API_KEY='YOUR_KEY'")
    print("      CMD: set OPENAI_API_KEY=YOUR_KEY")

def main():
    parser = argparse.ArgumentParser(description="Set API keys for chatbot comparison")
    parser.add_argument("--google", type=str, help="Google Gemini API key")
    parser.add_argument("--openai", type=str, help="OpenAI API key")
    
    args = parser.parse_args()
    
    if not args.google and not args.openai:
        print("⚠️  Chưa cung cấp API key nào")
        print("\nUsage:")
        print("  python src/set_api_keys.py --google YOUR_GOOGLE_API_KEY")
        print("  python src/set_api_keys.py --openai YOUR_OPENAI_API_KEY")
        print("  python src/set_api_keys.py --google YOUR_KEY --openai YOUR_KEY")
        return
    
    if args.google:
        set_google_api_key(args.google)
    
    if args.openai:
        set_openai_api_key(args.openai)
    
    print("\n✅ API keys đã được set!")
    print("   Bây giờ bạn có thể chạy:")
    print("   python src/demo_chatbot.py")
    print("   hoặc")
    print("   python src/run_chatbot.py --mode compare")

if __name__ == "__main__":
    main()




