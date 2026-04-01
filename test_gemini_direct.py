import os
from google import genai
from config import GEMINI_API_KEY, LLM_MODEL

print(f"Model: {LLM_MODEL}")
print(f"Key preview: {GEMINI_API_KEY[:5]}...{GEMINI_API_KEY[-5:]}")

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents="Say exactly 'Test OK'."
    )
    print("SUCCESS!")
    print("Response:", response.text)
except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("ERROR MSG:", str(e))
