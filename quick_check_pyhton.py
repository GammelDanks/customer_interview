# quick_check.py (optional Test)
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
print("Model list probe:")
print(client.models.list())  # sollte funktionieren; sonst ist der Key wirklich invalid