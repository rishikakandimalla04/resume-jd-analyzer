import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Models available to your API key:\n")
for m in client.models.list():
    print(m.name, "-", getattr(m, "supported_actions", ""))
