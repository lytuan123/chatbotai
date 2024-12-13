import os
import logging
import json
import re
import hashlib
from typing import Dict, List, Tuple, Optional

import gradio as gr
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from pydantic import Field
from groq import Groq

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEXES = {
    "vondautu": os.path.join(FAISS_INDEX_DIR, "vondautu_index"),
    "xaydung": os.path.join(FAISS_INDEX_DIR, "xaydung_index"),
}

logging.info("Loading embeddings model...")
embedding_model = SentenceTransformerEmbeddings(model_name='intfloat/multilingual-e5-large', model_kwargs={"device": "cpu"})
logging.info("Embeddings model loaded.")

def load_faiss_index(menu_item: str, index_dir: str) -> FAISS:
    index_file = os.path.join(index_dir, "index.faiss")
    metadata_file = os.path.join(index_dir, "index.pkl")
    if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
        raise FileNotFoundError(f"FAISS index for '{menu_item}' not found in {index_dir}.")
    logging.info(f"Loading FAISS index for '{menu_item}' from {index_dir}")
    return FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)

faiss_indices = {}
for menu_item, index_dir in FAISS_INDEXES.items():
    faiss_indices[menu_item] = load_faiss_index(menu_item, index_dir)
logging.info("All FAISS indices loaded successfully.")

def load_json(file_name: str) -> Dict:
    if not os.path.exists(file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file_name: str, data: Dict):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()

# LLM cho Groq
class GroqLLM(LLM):
    api_key: str = Field(...)
    model: str = Field(default="llama-3.3-70b-versatile")
    max_tokens: int = Field(default=200, ge=1)  # Tăng max_tokens để mô hình nói dài hơn

    class Config:
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, 'client', Groq(api_key=self.api_key))

    def _call(self, prompt: str, stop=None):
        logging.info("Sending request to Groq API...")
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.max_tokens,
                stream=False,
            )
            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content.strip()
                logging.info(f"Generated answer: {answer}")
                return answer
            else:
                logging.error("Groq API returned no choices.")
                return "Tôi không tìm thấy câu trả lời."
        except Exception as e:
            logging.error(f"Error: {e}")
            return f"Đã xảy ra lỗi: {e}"

    @property
    def _llm_type(self):
        return "groq_llm"

groq_llm = GroqLLM(api_key=GROQ_API_KEY, max_tokens=1024)


# system_message: hướng dẫn mô hình hành xử
# - Nếu có thông tin trong tài liệu, dùng nó
# - Nếu không, cố gắng trả lời như một AI thông minh, hữu ích
system_message = """Bạn là một trợ lý ảo thông minh, thân thiện và hữu ích. Bạn cũng có thể tương tác với người dùng như một AI thông minh.
Bạn có dữ liệu từ một số tài liệu.
- Nếu người dùng hỏi về nội dung có trong tài liệu, hãy dùng context đó để trả lời.
- Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời với kiến thức chung của bạn, như một AI thông minh.
- Nếu người dùng chỉ chào, hãy chào lại.
- Nếu thấy một từ không quen, cố gắng đoán từ gần nghĩa trong tài liệu hoặc hỏi lại người dùng.
- Hãy trả lời một cách chuyên nghiệp, ngắn gọn, súc tích nhưng đầy đủ thông tin.
- Hãy đảm bảo trả lời trọn vẹn câu, không kết thúc giữa chừng.
- Hãy trả lời hoàn toàn bằng tiếng Việt. Nếu gặp từ nước ngoài, hãy phiên âm hoặc dịch sang tiếng Việt.
- Hãy cung cấp câu trả lời đầy đủ, không cắt ngang."""

# Thiết lập chain
def get_chain_for_topic(selected_menu: str):
    if selected_menu not in faiss_indices:
        logging.error(f"No index found for {selected_menu}.")
        return None
    retriever = faiss_indices[selected_menu].as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # ConversationalRetrievalChain mặc định sẽ cố lấy context.
    # Nếu không có context, chain trả về kết quả chung chung, dựa vào LLM.
    chain = ConversationalRetrievalChain.from_llm(
        llm=groq_llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    return chain

chains = {}
for m in FAISS_INDEXES.keys():
    ch = get_chain_for_topic(m)
    if ch:
        chains[m] = ch

# Cache để lưu câu hỏi-đáp
CACHE_FILE = "cache.json"
cache_data = load_json(CACHE_FILE)

def handle_query(query: str, selected_menu: str, chat_history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    normalized_query = query.strip().lower()
    query_hash = hash_prompt(normalized_query)

    if not query.strip():
        return chat_history, chat_history

    # Check cache
    if query_hash in cache_data:
        answer = cache_data[query_hash]
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        return chat_history, chat_history

    if selected_menu not in chains:
        # Không có chain cho chủ đề này
        chat_history.append({"role": "assistant", "content": "Chưa có chain cho chủ đề này."})
        return chat_history, chat_history

    chain = chains[selected_menu]

    # Thêm user message vào hội thoại
    chat_history.append({"role": "user", "content": query})

    # Gọi chain
    # Ta có thể chèn system_message vào đầu prompt mỗi lần gọi
    # Chain từ langchain không có tham số system_message trực tiếp,
    # nhưng ta có thể kết hợp system_message bằng cách modify prompt chain,
    # hoặc tiền xử lý prompt. Ở đây, ta sẽ prompt chèn system_message vào query:
    # Lưu ý: ConversationalRetrievalChain thường có 2 bước: condense question và combine docs.
    # Ta có thể override prompt chain, nhưng ở đây để đơn giản ta tiền xử lý query:
    final_query = f"{system_message}\n\n{query}"
    result = chain({"question": final_query})
    answer = result["answer"]

    # Lưu vào cache
    cache_data[query_hash] = answer
    save_json(CACHE_FILE, cache_data)

    # Thêm assistant message
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, chat_history
