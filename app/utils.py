import os
import logging
import json
import hashlib
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from groq import Groq

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
            model_name='intfloat/multilingual-e5-large', 
            model_kwargs={"device": "cpu"}
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
        system_message = """Bạn là một trợ lý ảo thông minh, thân thiện và hữu ích. Bạn cũng có thể tương tác với người dùng như một AI thông minh.
Bạn có dữ liệu từ một số tài liệu.
- Nếu người dùng hỏi về nội dung có trong tài liệu, hãy dùng context đó để trả lời.
- Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời với kiến thức chung của bạn, như một AI thông minh.
- Nếu người dùng chỉ chào, hãy chào lại.
- Nếu thấy một từ không quen, cố gắng đoán từ gần nghĩa trong tài liệu hoặc hỏi lại người dùng.
- Hãy trả lời một cách chuyên nghiệp, ngắn gọn, súc tích nhưng đầy đủ thông tin.
- Hãy đảm bảo trả lời trọn vẹn câu, không kết thúc giữa chừng.
- Hãy cung cấp câu trả lời đầy đủ, không cắt ngang."""

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

class GroqLLM(Groq):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", max_tokens: int = 1024):
        super().__init__(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.chat.completions.create(
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
