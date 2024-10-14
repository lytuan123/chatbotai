from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os
import hashlib
import logging
import sqlite3
import re
import unicodedata
import traceback

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please check your .env file.")

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# Define paths for files
file_paths = {
    "Vốn đầu tư": "data/vondautu.pdf",
    "Xây dựng": "data/xaydung.pdf"
}

# Define paths for FAISS index files
index_paths = {
    "Vốn đầu tư": "faiss_index/vondautu_index",
    "Xây dựng": "faiss_index/xaydung_index"
}

# Initialize FAISS indices
faiss_indices = {}

# Build FAISS indices if they do not exist
def build_faiss_indices():
    for menu_item, file_path in file_paths.items():
        save_path = index_paths[menu_item]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(f"{save_path}/index.faiss"):
            logging.info(f"Building FAISS index for {menu_item}")
            # Read the PDF file
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            # Split the text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_text(text)
            # Create documents
            documents = [{"page_content": t} for t in texts]
            # Create FAISS vector store
            vector_store = FAISS.from_documents(documents, embedding_model)
            # Save vector store
            vector_store.save_local(save_path)
            faiss_indices[menu_item] = vector_store
        else:
            # Load existing vector store
            faiss_indices[menu_item] = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)

build_faiss_indices()

# Initialize SQLite database
def initialize_db():
    conn = sqlite3.connect('cache_and_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            prompt_hash TEXT PRIMARY KEY,
            response TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            prompt TEXT PRIMARY KEY,
            response TEXT
        )
    ''')
    conn.commit()
    conn.close()

initialize_db()

# META_PROMPT template
META_PROMPT = """
Chủ đề: {menu}
Ngữ cảnh trước đó: {previous_queries}
Câu hỏi hiện tại: {current_query}
Hướng dẫn:
- Trả lời theo phong cách chuyên nghiệp.
- Đưa ra câu trả lời ngắn gọn và rõ ràng, không dài dòng.
- Nếu câu hỏi nằm ngoài phạm vi dữ liệu, hãy trả lời: 'Câu hỏi này nằm ngoài phạm vi dữ liệu hiện tại. Xin vui lòng hỏi câu hỏi khác.'
"""

# Helper functions for caching and history with SQLite
def hash_prompt(prompt: str) -> str:
    """Create a hash for a given prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()

def save_to_cache(prompt_hash: str, response: str):
    """Save the response to cache in SQLite."""
    logging.info(f"Saving to cache: {prompt_hash}")
    conn = sqlite3.connect('cache_and_history.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO cache (prompt_hash, response) VALUES (?, ?)', (prompt_hash, response))
    conn.commit()
    conn.close()

def get_from_cache(prompt_hash: str) -> str:
    """Get the cached response from SQLite."""
    logging.info(f"Checking cache for prompt_hash: {prompt_hash}")
    conn = sqlite3.connect('cache_and_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT response FROM cache WHERE prompt_hash=?', (prompt_hash,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_to_history(prompt: str, response: str):
    """Save the response to history in SQLite."""
    logging.info(f"Saving to history: {prompt}")
    conn = sqlite3.connect('cache_and_history.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO history (prompt, response) VALUES (?, ?)', (prompt, response))
    conn.commit()
    conn.close()

def get_from_history(prompt: str) -> str:
    """Get the response from history in SQLite."""
    logging.info(f"Checking history for prompt: {prompt}")
    conn = sqlite3.connect('cache_and_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT response FROM history WHERE prompt=?', (prompt,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def normalize_query(query: str) -> str:
    """Chuẩn hóa câu hỏi bằng cách loại bỏ dấu câu và khoảng trắng thừa."""
    query = unicodedata.normalize('NFD', query)
    query = query.encode('ascii', 'ignore').decode("utf-8")
    query = re.sub(r'[^\w\s]', '', query)
    return query.lower().strip()

# Function to generate answer with META_PROMPT and reranking
def generate_answer_with_meta_prompt(query, vector_store, selected_menu, context, max_tokens=150):
    logging.info("Entering generate_answer_with_meta_prompt")
    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
    docs = retriever.get_relevant_documents(query)

    if not docs:
        logging.info("No relevant documents found")
        return "Câu hỏi này nằm ngoài phạm vi dữ liệu hiện tại. Xin vui lòng hỏi câu hỏi khác."

    # Initialize language model
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0, model_name="gpt-4o", max_tokens=max_tokens)

    # Reranking step
    rerank_prompt = f"""
Bạn là một trợ lý thông minh. Người dùng hỏi: "{query}". Dựa trên các đoạn văn bản sau, hãy chọn ra những đoạn phù hợp nhất để trả lời câu hỏi.

Các đoạn văn bản:
"""
    for i, doc in enumerate(docs):
        rerank_prompt += f"\n[{i+1}] {doc.page_content}"

    rerank_prompt += "\n\nHãy liệt kê số thứ tự của các đoạn văn bản phù hợp nhất, tối đa 3 đoạn."

    rerank_response = llm([{"role": "user", "content": rerank_prompt}])

    # Extract indices from the response
    relevant_indices = re.findall(r'\d+', rerank_response.content)
    relevant_indices = [int(idx)-1 for idx in relevant_indices if int(idx)-1 < len(docs)]

    if not relevant_indices:
        logging.info("No relevant indices found after reranking")
        return "Câu hỏi này nằm ngoài phạm vi dữ liệu hiện tại. Xin vui lòng hỏi câu hỏi khác."

    # Select the top relevant documents
    selected_docs = [docs[idx] for idx in relevant_indices]

    # Prepare final prompt
    final_prompt = f"""
Chủ đề: {selected_menu}
Ngữ cảnh trước đó: {context if context else "Không có ngữ cảnh trước đó."}
Câu hỏi hiện tại: {query}
Dựa trên các thông tin sau, hãy trả lời câu hỏi một cách ngắn gọn và rõ ràng:
"""
    for doc in selected_docs:
        final_prompt += f"\n{doc.page_content}"

    final_prompt += "\n\nCâu trả lời:"

    # Generate the final answer
    answer_response = llm([{"role": "user", "content": final_prompt}])

    return answer_response.content.strip()

# API endpoint to handle queries (with META_PROMPT)
@app.route('/query', methods=['POST'])
def query_with_meta_prompt():
    try:
        data = request.json
        query = data.get("query", "")
        selected_menu = data.get("menu", "")
        context = data.get("context", "")  # Ngữ cảnh từ người dùng (nếu có)

        if not selected_menu:
            return jsonify({"error": "No menu selected."}), 400

        # Kiểm tra nếu menu đã có FAISS index chưa
        if selected_menu not in faiss_indices or not faiss_indices[selected_menu]:
            return jsonify({"error": f"Menu '{selected_menu}' không tồn tại hoặc FAISS index chưa được nạp"}), 400

        vector_store = faiss_indices[selected_menu]

        # Chuẩn hóa câu hỏi
        normalized_query = normalize_query(query)

        # Tạo ngữ cảnh đầy đủ với META_PROMPT
        contextual_prompt = META_PROMPT.format(
            menu=selected_menu,
            previous_queries=context if context else "Không có ngữ cảnh trước đó.",
            current_query=normalized_query
        ).strip()

        # Kiểm tra cache trước
        prompt_hash = hash_prompt(contextual_prompt)
        cached_answer = get_from_cache(prompt_hash)
        if cached_answer:
            logging.info("Answer retrieved from cache")
            return jsonify({"answer": cached_answer, "context": context})

        # Kiểm tra lịch sử
        history_answer = get_from_history(contextual_prompt)
        if history_answer:
            logging.info("Answer retrieved from history")
            # Lưu vào cache để tăng tốc độ cho lần sau
            save_to_cache(prompt_hash, history_answer)
            return jsonify({"answer": history_answer, "context": context})

        # Tạo câu trả lời bằng RAG
        answer = generate_answer_with_meta_prompt(normalized_query, vector_store, selected_menu, context, max_tokens=150)

        # Lưu lại câu trả lời vào cache và lịch sử
        save_to_cache(prompt_hash, answer)
        save_to_history(contextual_prompt, answer)

        # Cập nhật ngữ cảnh hội thoại với câu hỏi và câu trả lời hiện tại
        updated_context = f"{context} {normalized_query} {answer}".strip() if context else f"{normalized_query} {answer}"

        # Giới hạn độ dài của ngữ cảnh để tránh quá dài
        max_context_length = 1000  # Bạn có thể điều chỉnh giá trị này
        if len(updated_context) > max_context_length:
            updated_context = updated_context[-max_context_length:]

        return jsonify({"answer": answer, "context": updated_context})
    except Exception as e:
        logging.error(f"Error in /query endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "An error occurred while processing your request."}), 500

# API endpoint to check server status
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "Server is running"}), 200

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))