import os
import logging
import hashlib
from typing import List, Dict, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
from app.utils import load_json, save_json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not found in the .env file.")

FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEXES = {
    "vondautu": os.path.join(FAISS_INDEX_DIR, "vondautu_index"),
    "xaydung": os.path.join(FAISS_INDEX_DIR, "xaydung_index"),
}

# Initialize embeddings
logging.info("Loading embedding model...")
embedding_model = SentenceTransformerEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"}
)
logging.info("Embedding model loaded successfully.")

# Helper function to load FAISS index
def load_faiss_index(menu_item: str, index_dir: str) -> FAISS:
    index_file = os.path.join(index_dir, "index.faiss")
    metadata_file = os.path.join(index_dir, "index.pkl")
    if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
        raise FileNotFoundError(f"FAISS index for '{menu_item}' not found in {index_dir}.")
    logging.info(f"Loading FAISS index for '{menu_item}' from {index_dir}")
    return FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)

# Load all FAISS indices
faiss_indices = {}
for menu_item, index_dir in FAISS_INDEXES.items():
    try:
        faiss_indices[menu_item] = load_faiss_index(menu_item, index_dir)
    except FileNotFoundError as e:
        logging.error(e)
logging.info("All FAISS indices loaded successfully.")

# LLM Wrapper for Groq API
class GroqLLM(LLM):
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 1024

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "client", Groq(api_key=self.api_key))

    def _call(self, prompt: str, stop=None):
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.max_tokens,
                stream=False,
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "Xin lỗi, tôi không tìm thấy câu trả lời phù hợp."
        except Exception as e:
            logging.error(f"Error calling Groq API: {e}")
            return f"Đã xảy ra lỗi: {e}"

    @property
    def _llm_type(self) -> str:
        return "groq_llm"

groq_llm = GroqLLM(api_key=GROQ_API_KEY)

# Load cache for queries
CACHE_FILE = "cache.json"
cache_data = load_json(CACHE_FILE)

# Define system message for context
system_message = """Bạn là một trợ lý ảo thông minh, thân thiện và hữu ích. Bạn cũng có thể tương tác với người dùng như một AI thông minh.
Bạn có dữ liệu từ một số tài liệu.
- Nếu người dùng hỏi về nội dung có trong tài liệu, hãy dùng context đó để trả lời.
- Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời với kiến thức chung của bạn, như một AI thông minh.
- Nếu người dùng chỉ chào, hãy chào lại.
- Nếu thấy một từ không quen, cố gắng đoán từ gần nghĩa trong tài liệu hoặc hỏi lại người dùng.
- Hãy trả lời một cách chuyên nghiệp, ngắn gọn, súc tích nhưng đầy đủ thông tin.
- Hãy đảm bảo trả lời trọn vẹn câu, không kết thúc giữa chừng.
- Hãy cung cấp câu trả lời đầy đủ, không cắt ngang."""

# Create Conversational Chain for each FAISS index
def get_chain_for_topic(selected_menu: str):
    if selected_menu not in faiss_indices:
        logging.error(f"No index found for {selected_menu}.")
        return None
    retriever = faiss_indices[selected_menu].as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=groq_llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

chains = {menu: get_chain_for_topic(menu) for menu in FAISS_INDEXES.keys() if get_chain_for_topic(menu)}

# Handle user query
def handle_query(query: str, selected_menu: str, chat_history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    normalized_query = query.strip().lower()
    query_hash = hashlib.md5(normalized_query.encode("utf-8")).hexdigest()

    if not query.strip():
        return chat_history, chat_history

    # Check cache
    if query_hash in cache_data:
        answer = cache_data[query_hash]
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        return chat_history, chat_history

    if selected_menu not in chains:
        chat_history.append({"role": "assistant", "content": "Chưa có chain cho chủ đề này."})
        return chat_history, chat_history

    chain = chains[selected_menu]
    chat_history.append({"role": "user", "content": query})

    # Add system message
    final_query = f"{system_message}\n\n{query}"
    try:
        result = chain({"question": final_query})
        answer = result.get("answer", "Tôi không thể trả lời câu hỏi này.")
    except Exception as e:
        logging.error(f"Error in chain execution: {e}")
        answer = "Đã xảy ra lỗi khi xử lý yêu cầu của bạn."

    # Save to cache
    cache_data[query_hash] = answer
    save_json(CACHE_FILE, cache_data)

    chat_history.append({"role": "assistant", "content": answer})
    return chat_history, chat_history
