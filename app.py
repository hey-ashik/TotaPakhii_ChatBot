import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import uuid
from datetime import datetime
import requests
import json
import re

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Flask app initialization
app = Flask(__name__)
CORS(app, origins=["https://chat.ashikai.xyz", "http://127.0.0.1:7860", "http://localhost:7860"])

# API Keys Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdfrag")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Pinecone, Embeddings, and LLM
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.5, max_tokens=500) if GROQ_API_KEY else None

# Global storage
chat_sessions = {}
STRICT_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks. Use ONLY the following pieces of retrieved context to answer the question. If you don't know the answer from the context, just say "I was unable to find an answer in the provided documents." Do not use any outside knowledge.

Context: {context}

Question: {question}

Helpful Answer:"""
STRICT_PROMPT = PromptTemplate.from_template(STRICT_PROMPT_TEMPLATE)

def load_vector_store():
    if not pc: return None
    try:
        return PineconeVectorStore(index=pc.Index(PINECONE_INDEX_NAME), embedding=embeddings)
    except Exception as e:
        print(f"❌ Error loading Pinecone index: {e}")
        return None

def create_conversational_qa_chain(vector_store):
    if not llm: return None
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
        qa_chain.combine_docs_chain.llm_chain.prompt = STRICT_PROMPT
        return qa_chain
    except Exception as e:
        print(f"❌ Error creating Conversational QA chain: {e}")
        return None

def tavily_web_search(query, max_results=5):
    if not TAVILY_API_KEY: return None, "Tavily API key not configured"
    try:
        response = requests.post("https://api.tavily.com/search", json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "advanced", "include_answer": True, "max_results": max_results}, timeout=30)
        return (response.json(), None) if response.ok else (None, f"Tavily API error: {response.status_code}")
    except Exception as e:
        return None, f"Web search error: {str(e)}"

def format_web_search_response(search_data, original_query):
    if not search_data or not search_data.get('results'):
        return "I couldn't find any relevant results on the web for your query.", []
    results = search_data.get('results', [])
    answer = search_data.get('answer', '')
    if llm and (not answer or len(answer.split()) < 5) and results:
        context = "\n\n".join([f"Source Title: {r.get('title', '')}\nContent Snippet: {r.get('content', '')}" for r in results[:5]])
        prompt = f'Based on web results, answer: "{original_query}"\n\nResults:\n{context}\n\nAnswer:'
        try:
            answer = llm.invoke(prompt).content
        except Exception:
            if not answer: answer = "I found web results but had trouble creating a summary."
    sources = [[f"Web Source {i+1}", f"**{r.get('title', '')}**\n{r.get('content', '')[:250]}...\n🔗 {r.get('url', '#')}"] for i, r in enumerate(results[:3])]
    return answer, sources

def query_pdfs_with_context(query, session_id=None):
    if not llm: return "Service Unavailable", [], "failure"
    vector_store = load_vector_store()
    if not vector_store: return "Knowledge Base Unavailable", [], "failure"
    qa_chain = create_conversational_qa_chain(vector_store)
    if not qa_chain: return "Knowledge Base Unavailable", [], "failure"
    try:
        history = [msg for pair in chat_sessions.get(session_id, {}).get('messages', []) for msg in (HumanMessage(content=pair[0]), AIMessage(content=pair[1]))]
        response = qa_chain.invoke({"question": query, "chat_history": history})
        answer, source_docs = response.get('answer', ''), response.get('source_documents', [])
        if "unable to find an answer" in answer.lower():
            return "No relevant information found in the knowledge base.", [], "failure"
        sources = [[f"Page {doc.metadata.get('page', i+1)}", doc.page_content[:200].strip() + "..."] for i, doc in enumerate(source_docs[:2])]
        return answer, sources, "pdf"
    except Exception as e:
        return "Knowledge Base Error", [], "failure"

def query_web_search(query):
    search_data, error = tavily_web_search(query)
    if error: return f"Web Search Failed: {error}", [], "web"
    return format_web_search_response(search_data, query) + ("web",)

def query_deep_search(query, session_id=None):
    if not llm: return "Service Unavailable", [], "deepsearch"
    try:
        history_str = "\n".join([f"Human: {p[0]}\nAI: {p[1]}" for p in chat_sessions.get(session_id, {}).get('messages', [])])
        prompt = PromptTemplate.from_template("History:\n{chat_history}\n\nUser Question: {question}\n\nAnswer:")
        chain = prompt | llm
        return chain.invoke({"question": query, "chat_history": history_str}).content, [], "deepsearch"
    except Exception:
        return "LLM Error", [], "deepsearch"

def get_demo_response(query):
    return "Demo Mode: API keys not configured.", [], "demo"

def is_current_event_query(query: str) -> bool:
    query_lower = query.lower()
    current_event_keywords = ['today', 'latest', 'current', 'news', 'who won', 'stock price', 'weather', 'recent', 'this week', 'as of today', 'in 2024', 'in 2023']
    return any(keyword in query_lower for keyword in current_event_keywords)

def perform_rag_search(query, session_id=None):
    print(f"🤖 Activating RAG Search Agent for query: '{query}'")
    if is_current_event_query(query):
        print("   ▶️ Intent: Current Event. Starting with Web Search.")
        if TAVILY_API_KEY:
            web_answer, web_sources, web_type = query_web_search(query)
            if web_sources:
                print("   ✅ Success! Found current information via Web Search.")
                return web_answer, web_sources, "web"
        print("   ⚠️ Web Search failed. Falling back to Deep Search.")
        if GROQ_API_KEY: return query_deep_search(query, session_id)
    else:
        print("   ▶️ Intent: General Query. Starting with Knowledge Base.")
        if GROQ_API_KEY and PINECONE_API_KEY:
            kb_answer, kb_sources, kb_type = query_pdfs_with_context(query, session_id)
            if kb_type == "pdf":
                print("   ✅ Success! Found a relevant answer in the Knowledge Base.")
                return kb_answer, kb_sources, "pdf"
        print("   ⚠️ Knowledge Base insufficient. Trying Deep Search.")
        if GROQ_API_KEY:
            deep_answer, deep_sources, deep_type = query_deep_search(query, session_id)
            if deep_answer and "as an ai" not in deep_answer.lower() and "i cannot" not in deep_answer.lower():
                 print("   ✅ Success! Generated a response using Deep Search.")
                 return deep_answer, deep_sources, "deepsearch"
        print("   ⚠️ Deep Search was insufficient. Final attempt with Web Search.")
        if TAVILY_API_KEY: return query_web_search(query)
    print("   ❌ All search methods failed or are not configured.")
    return get_demo_response(query)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        search_type = data.get('type', 'knowledgebase')
        session_id = data.get('session_id')
        
        if not message: return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        if not session_id or session_id not in chat_sessions:
            session_id = str(uuid.uuid4())[:8]
            chat_sessions[session_id] = {'title': 'New Chat', 'messages': [], 'created_at': datetime.now().strftime("%H:%M")}
        
        raw_content, sources, result_type = "", [], "demo"

        if search_type == 'ragsearch':
            raw_content, sources, result_type = perform_rag_search(message, session_id)
        elif search_type == 'web':
            raw_content, sources, result_type = query_web_search(message) if TAVILY_API_KEY else get_demo_response(message)
        elif search_type == 'deepsearch':
            raw_content, sources, result_type = query_deep_search(message, session_id) if GROQ_API_KEY else get_demo_response(message)
        else:
            raw_content, sources, result_type = query_pdfs_with_context(message, session_id) if GROQ_API_KEY and PINECONE_API_KEY else get_demo_response(message)
        
        prefix = {
            "pdf": "📄 From Knowledge Base:",
            "web": "🌐 From Web Search:",
            "deepsearch": "🧠 From Deep Search:",
            "demo": "🤖 Demo Mode:"
        }.get(result_type, "")

        final_answer = f"{prefix}\n\n{raw_content}".replace('**', '')

        session = chat_sessions[session_id]
        session['messages'].append([message, final_answer])
        if len(session['messages']) == 1: session['title'] = message[:30] + "..."
        
        return jsonify({'success': True, 'answer': final_answer, 'sources': sources, 'session_id': session_id, 'search_type': result_type})
        
    except Exception as e:
        print(f"❌ Error in chat API: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=7860)