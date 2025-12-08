"""
Script để chạy Streamlit trên Google Colab với ngrok tunneling.

Cách sử dụng trên Colab:
1. Cài đặt dependencies:
   !pip install streamlit pyngrok

2. Chạy script này:
   !python src/run_streamlit_colab.py

3. Script sẽ tự động:
   - Khởi động Streamlit trên port 8501
   - Tạo ngrok tunnel
   - Hiển thị URL công khai để truy cập
"""

import os
import sys
import subprocess
import time
from threading import Thread

# Add project paths
CURR_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def check_ngrok():
    """Kiểm tra ngrok đã được cài đặt chưa."""
    try:
        import pyngrok
        return True
    except ImportError:
        return False

def install_ngrok():
    """Cài đặt pyngrok."""
    print("📦 Đang cài đặt pyngrok...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "-q"])
    print("✅ Đã cài đặt pyngrok")

def run_streamlit():
    """Chạy Streamlit app trong background."""
    streamlit_app_path = os.path.join(CURR_DIR, "chatbot", "streamlit_app.py")
    
    if not os.path.exists(streamlit_app_path):
        print(f"❌ Không tìm thấy file: {streamlit_app_path}")
        return
    
    print("🚀 Đang khởi động Streamlit...")
    
    # Chạy streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", streamlit_app_path, "--server.port", "8501", "--server.address", "0.0.0.0"]
    subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Đợi Streamlit khởi động
    time.sleep(5)
    print("✅ Streamlit đã khởi động trên port 8501")

def create_ngrok_tunnel():
    """Tạo ngrok tunnel để expose Streamlit."""
    try:
        from pyngrok import ngrok
        
        # Tạo tunnel
        print("🔗 Đang tạo ngrok tunnel...")
        public_url = ngrok.connect(8501, bind_tls=True)
        
        print("\n" + "="*70)
        print("✅ STREAMLIT ĐÃ SẴN SÀNG!")
        print("="*70)
        print(f"\n🌐 URL công khai: {public_url}")
        print(f"\n💡 Mở URL trên trong trình duyệt để sử dụng chatbot.")
        print("\n⚠️  Lưu ý:")
        print("   - URL này sẽ thay đổi mỗi lần chạy lại")
        print("   - Để dừng: Nhấn Ctrl+C hoặc chạy ngrok.kill()")
        print("="*70 + "\n")
        
        return public_url
    except Exception as e:
        print(f"❌ Lỗi khi tạo ngrok tunnel: {e}")
        print("\n💡 Thử cách khác:")
        print("   1. Cài đặt ngrok: !pip install pyngrok")
        print("   2. Lấy ngrok authtoken từ https://dashboard.ngrok.com/get-started/your-authtoken")
        print("   3. Chạy: ngrok config add-authtoken YOUR_TOKEN")
        return None

def main():
    """Hàm main."""
    print("="*70)
    print("🎤 K-POP CHATBOT - STREAMLIT ON COLAB")
    print("="*70)
    
    # Kiểm tra và cài đặt ngrok
    if not check_ngrok():
        try:
            install_ngrok()
        except Exception as e:
            print(f"❌ Không thể cài đặt pyngrok: {e}")
            print("\n💡 Hãy chạy thủ công:")
            print("   !pip install pyngrok")
            return
    
    # Chạy Streamlit trong background
    streamlit_thread = Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Đợi Streamlit khởi động
    time.sleep(8)
    
    # Tạo ngrok tunnel
    public_url = create_ngrok_tunnel()
    
    if public_url:
        # Giữ script chạy
        try:
            print("\n⏳ Đang chạy... Nhấn Ctrl+C để dừng.\n")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Đang dừng...")
            try:
                from pyngrok import ngrok
                ngrok.kill()
            except:
                pass
            print("✅ Đã dừng.")

if __name__ == "__main__":
    main()


