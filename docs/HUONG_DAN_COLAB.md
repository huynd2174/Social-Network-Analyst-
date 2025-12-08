# Hướng Dẫn Chạy Streamlit trên Google Colab

## Vấn đề

Khi chạy Streamlit trên Google Colab bằng lệnh thông thường:
```python
!streamlit run src/chatbot/streamlit_app.py
```

Giao diện sẽ không hiển thị vì Colab không tự động expose port localhost ra ngoài.

## Giải pháp: Sử dụng ngrok

Ngrok là công cụ tạo tunnel để expose port local ra internet, cho phép truy cập Streamlit từ bên ngoài Colab.

## Cách 1: Sử dụng Script Helper (Khuyến nghị)

### Bước 1: Cài đặt dependencies

```python
!pip install streamlit pyngrok
```

### Bước 2: Chạy script helper

```python
!python src/run_streamlit_colab.py
```

Script sẽ tự động:
- Khởi động Streamlit trên port 8501
- Tạo ngrok tunnel
- Hiển thị URL công khai để truy cập

### Bước 3: Mở URL trong trình duyệt

Script sẽ hiển thị URL dạng:
```
https://xxxx-xxxx-xxxx.ngrok-free.app
```

Copy URL này và mở trong trình duyệt để sử dụng chatbot.

## Cách 2: Chạy thủ công

### Bước 1: Cài đặt ngrok

```python
!pip install pyngrok
```

### Bước 2: Lấy ngrok authtoken (nếu chưa có)

1. Đăng ký tài khoản miễn phí tại: https://dashboard.ngrok.com/signup
2. Lấy authtoken từ: https://dashboard.ngrok.com/get-started/your-authtoken
3. Cấu hình:

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
```

### Bước 3: Khởi động Streamlit

```python
import subprocess
import threading
from pyngrok import ngrok

# Chạy Streamlit trong background
def run_streamlit():
    subprocess.run([
        "streamlit", "run", 
        "src/chatbot/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

# Khởi động Streamlit
streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
streamlit_thread.start()

# Đợi Streamlit khởi động
import time
time.sleep(5)

# Tạo ngrok tunnel
public_url = ngrok.connect(8501, bind_tls=True)
print(f"🌐 URL công khai: {public_url}")
```

### Bước 4: Mở URL trong trình duyệt

Copy URL từ output và mở trong trình duyệt.

## Cách 3: Sử dụng localtunnel (Thay thế ngrok)

Nếu không muốn dùng ngrok, có thể dùng localtunnel:

```python
!npm install -g localtunnel

# Trong một cell riêng, chạy Streamlit:
!streamlit run src/chatbot/streamlit_app.py --server.port 8501 &

# Trong cell khác, tạo tunnel:
!lt --port 8501
```

## Lưu ý quan trọng

1. **URL thay đổi mỗi lần chạy**: Mỗi lần chạy lại script, ngrok sẽ tạo URL mới.

2. **Giới hạn ngrok miễn phí**: 
   - Có thể có giới hạn về số lượng requests
   - URL có thể bị timeout sau một thời gian không sử dụng

3. **Bảo mật**: 
   - URL ngrok là công khai, ai có link đều có thể truy cập
   - Không nên dùng cho dữ liệu nhạy cảm

4. **Dừng server**:
   ```python
   from pyngrok import ngrok
   ngrok.kill()  # Dừng tất cả tunnels
   ```

## Troubleshooting

### Lỗi: "ngrok authtoken not set"

Giải pháp: Cấu hình authtoken như ở Bước 2 của Cách 2.

### Lỗi: "Port already in use"

Giải pháp: Đổi port hoặc kill process đang dùng port:
```python
!lsof -ti:8501 | xargs kill -9
```

### Streamlit không khởi động

Giải pháp: Kiểm tra đường dẫn file:
```python
import os
print(os.path.exists("src/chatbot/streamlit_app.py"))
```

### Không thể truy cập URL

Giải pháp:
- Kiểm tra ngrok tunnel đã được tạo chưa
- Thử tạo lại tunnel
- Kiểm tra firewall/antivirus có chặn không

## Ví dụ hoàn chỉnh cho Colab

```python
# Cell 1: Cài đặt
!pip install streamlit pyngrok

# Cell 2: Cấu hình ngrok (nếu chưa có authtoken)
from pyngrok import ngrok
# ngrok.set_auth_token("YOUR_TOKEN")  # Uncomment và thay YOUR_TOKEN

# Cell 3: Chạy Streamlit với ngrok
import subprocess
import threading
import time

def run_streamlit():
    subprocess.run([
        "streamlit", "run", 
        "src/chatbot/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])

# Khởi động Streamlit
thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()
time.sleep(8)

# Tạo tunnel
public_url = ngrok.connect(8501, bind_tls=True)
print(f"\n✅ Streamlit đã sẵn sàng!")
print(f"🌐 URL: {public_url}\n")
print("💡 Copy URL trên và mở trong trình duyệt để sử dụng chatbot.")
```

## Tài liệu tham khảo

- Streamlit: https://docs.streamlit.io/
- ngrok: https://ngrok.com/docs
- pyngrok: https://pyngrok.readthedocs.io/

