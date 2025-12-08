# Cách thiết lập API Keys cho Chatbot Comparison

Để so sánh chatbot với ChatGPT và Gemini, bạn cần thiết lập API keys.

## 1. Google Gemini API Key

### Cách lấy API Key:
1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập với Google account
3. Tạo API key mới
4. Copy API key

### Cách thiết lập:

#### Windows (PowerShell):
```powershell
$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

#### Windows (Command Prompt):
```cmd
set GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

#### Windows (Permanent - System Environment):
1. Mở "Environment Variables" từ Control Panel
2. Thêm biến mới:
   - Variable name: `GOOGLE_API_KEY`
   - Variable value: `YOUR_API_KEY_HERE`

#### Linux/Mac:
```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

Để lưu vĩnh viễn, thêm vào `~/.bashrc` hoặc `~/.zshrc`:
```bash
echo 'export GOOGLE_API_KEY="YOUR_API_KEY_HERE"' >> ~/.bashrc
source ~/.bashrc
```

## 2. OpenAI API Key (ChatGPT)

### Cách lấy API Key:
1. Truy cập: https://platform.openai.com/api-keys
2. Đăng nhập với OpenAI account
3. Tạo API key mới
4. Copy API key

### Cách thiết lập:

#### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

#### Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=YOUR_API_KEY_HERE
```

#### Linux/Mac:
```bash
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

## 3. Kiểm tra API Keys đã được thiết lập

### Windows (PowerShell):
```powershell
echo $env:GOOGLE_API_KEY
echo $env:OPENAI_API_KEY
```

### Windows (Command Prompt):
```cmd
echo %GOOGLE_API_KEY%
echo %OPENAI_API_KEY%
```

### Linux/Mac:
```bash
echo $GOOGLE_API_KEY
echo $OPENAI_API_KEY
```

## 4. Sử dụng trong Python Script

Bạn cũng có thể set API keys trực tiếp trong Python:

```python
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
```

## 5. Chạy Demo với API Keys

Sau khi set API keys, chạy demo:

```bash
python src/demo_chatbot.py
```

Hoặc chạy comparison trực tiếp:

```bash
python src/run_chatbot.py --mode compare
```

## Lưu ý:

- ⚠️ **KHÔNG** commit API keys vào Git
- ⚠️ **KHÔNG** chia sẻ API keys công khai
- ✅ Sử dụng environment variables
- ✅ Thêm `.env` vào `.gitignore` nếu dùng file .env

## Troubleshooting

### Lỗi: "API key not found"
- Kiểm tra lại đã set environment variable chưa
- Đảm bảo không có khoảng trắng thừa trong API key
- Restart terminal/IDE sau khi set environment variable

### Lỗi: "Invalid API key"
- Kiểm tra lại API key đã copy đúng chưa
- Đảm bảo API key chưa hết hạn
- Kiểm tra account có đủ quota không




