import os
import json
import hashlib
from typing import List, Dict, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from groq import Groq
from app.utils import load_json, save_json

# Logic xử lý query và các hàm hỗ trợ...
# (Dùng code từ đoạn chatbot của bạn)
