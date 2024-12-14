import json
import os
import hashlib

def load_json(file_name: str) -> dict:
    if not os.path.exists(file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file_name: str, data: dict):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()
