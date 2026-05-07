from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

if key:
    print("API KEY 載入成功")
    print(key[:10])  # 只顯示前面避免外流
else:
    print("抓不到 API KEY")