import os
import logging
import json
import hashlib
from typing import Dict, List, Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from groq import Client as GroqClient  # Thay đổi import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

class ChatRequest(BaseModel):
    query: str
    topic: str
    chat_history: List[Dict[str, str]] = []

class ChatbotManager:
    def __init__(self, faiss_index_dir: str = "faiss_index"):
        self.faiss_index_dir = faiss_index_dir
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name='intfloat/multilingual-e5-large'
        )
        self.faiss_indices = {}
        self.chains = {}
        self.cache_file = "cache.json"
        self.cache_data = self._load_cache()
        self._initialize_indices()

    def _load_cache(self) -> Dict:
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({}, f)
        with open(self.cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache_data, f, ensure_ascii=False, indent=4)

    def _initialize_indices(self):
        topics = [d for d in os.listdir(self.faiss_index_dir) 
                  if os.path.isdir(os.path.join(self.faiss_index_dir, d))]
        
        for topic in topics:
            index_dir = os.path.join(self.faiss_index_dir, topic)
            self._load_faiss_index(topic, index_dir)
            self._create_retrieval_chain(topic)

    def _load_faiss_index(self, topic: str, index_dir: str):
        index_file = os.path.join(index_dir, "index.faiss")
        metadata_file = os.path.join(index_dir, "index.pkl")
        
        if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
            logging.warning(f"FAISS index for '{topic}' not found.")
            return
        
        self.faiss_indices[topic] = FAISS.load_local(
            index_dir, 
            self.embedding_model, 
            allow_dangerous_deserialization=True
        )
        logging.info(f"Loaded FAISS index for '{topic}'")

    def _create_retrieval_chain(self, topic: str):
        if topic not in self.faiss_indices:
            logging.error(f"No index found for {topic}.")
            return

        retriever = self.faiss_indices[topic].as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        llm = GroqLLM(
            api_key=os.getenv('GROQ_API_KEY', ''), 
            max_tokens=1024
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=False
        )
        
        self.chains[topic] = chain

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()

    def handle_query(self, request: ChatRequest) -> Dict:
        query = request.query.strip().lower()
        query_hash = self._hash_prompt(query)

        # Check cache
        if query_hash in self.cache_data:
            return {
                "answer": self.cache_data[query_hash],
                "chat_history": request.chat_history + [
                    {"role": "user", "content": request.query},
                    {"role": "assistant", "content": self.cache_data[query_hash]}
                ]
            }

        # Check if topic has a chain
        if request.topic not in self.chains:
            return {
                "answer": "Chủ đề không hợp lệ.",
                "chat_history": request.chat_history
            }

        chain = self.chains[request.topic]

        # System message for context
        system_message = """Bạn là một trợ lý ảo thông minh, thân thiện và hữu ích. Vai trò của bạn bao gồm:
        - Trả lời các câu hỏi dựa trên dữ liệu trong tài liệu mà bạn có.
        - Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời dựa trên kiến thức chung của bạn hoặc cung cấp câu trả lời hợp lý nhất.
        - Ghi nhớ ngữ cảnh hội thoại để trả lời các câu hỏi liên quan một cách tự nhiên và logic.
        - Nếu câu hỏi không rõ ràng hoặc cần thêm thông tin, hãy hỏi lại người dùng để làm rõ.
        - Luôn giao tiếp một cách thân thiện, dễ hiểu, và tích cực. Nếu người dùng chỉ chào, hãy chào lại với thái độ nhiệt tình.
        - Nếu phát hiện lỗi chính tả hoặc từ không quen thuộc trong câu hỏi của người dùng, hãy cố gắng đoán nghĩa hoặc hỏi lại để làm rõ.
        - Nếu không chắc chắn về câu trả lời, hãy nói rõ nhưng vẫn cung cấp hướng dẫn hoặc đề xuất giải pháp.
        - Gợi ý thêm những câu hỏi hoặc thông tin liên quan mà người dùng có thể quan tâm.
        - Đảm bảo câu trả lời trọn vẹn, không cắt ngang hoặc kết thúc giữa chừng."""

        # Prepare final query
        final_query = f"{system_message}\n\n{request.query}"

        # Process query
        result = chain({"question": final_query})
        answer = result["answer"]

        # Cache the result
        self.cache_data[query_hash] = answer
        self._save_cache()

        # Update chat history
        updated_chat_history = request.chat_history + [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": answer}
        ]

        return {
            "answer": answer,
            "chat_history": updated_chat_history
        }

class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", max_tokens: int = 1024):
        self.client = GroqClient(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"LLM Generation Error: {e}")
            return "Có lỗi xảy ra khi tạo câu trả lời."

# Global singleton
chatbot_manager = ChatbotManager()
