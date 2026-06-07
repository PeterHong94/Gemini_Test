import streamlit as st
import os
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 파일 경로 정의
HISTORY_FILE = "chat_history.json"
INDEX_DIR = "my_faiss_index"

st.set_page_config(page_title="기억력 만렙 RAG 챗봇", page_icon="🤖")
st.title("🤖 영구 기억 & 대화 제어 RAG 챗봇")

# API 키 설정 (본인의 키로 대체하세요)
GOOGLE_API_KEY = "AIzaSyAnesZNt_tBAe5Qr7VybRyFj5PaiVadaYs" 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ------------------------------------------------------------------
# 2. 사이드바 컨트롤 영역 (기억 스위치 + 데이터 업로드)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 챗봇 설정 컨트롤러")
    
    # 📌 요청하신 대화 기억 선택 스위치
    remember_chat = st.toggle("현재 대화 기록 기억하기", value=True, 
                              help="이 기능을 켜면 대화 내용이 노트북에 영구 저장되어 다음에 앱을 켜도 이어집니다.")
    
    st.markdown("---")
    st.header("📂 데이터 지식베이스")
    uploaded_file = st.file_uploader("챗봇에게 학습시킬 텍스트(.txt) 파일을 올려주세요", type=["txt"])

# ------------------------------------------------------------------
# 3. 데이터 로딩 및 초기화 (앱 시작 시 1회 실행)
# ------------------------------------------------------------------
# [임베딩 모델 정의]
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ① 대화 기록 불러오기 (스위치가 켜져 있고, 기존 파일이 있다면 로드)
if "chat_history" not in st.session_state:
    if remember_chat and os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.chat_history = json.load(f)
        st.sidebar.success("💬 이전 대화 기록을 불러왔습니다!")
    else:
        st.session_state.chat_history = []

# ② 지식베이스(FAISS DB) 파일 자동 불러오기
if "vector_store" not in st.session_state:
    if os.path.exists(INDEX_DIR):
        st.session_state.vector_store = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        st.sidebar.info("💾 기존에 저장된 지식베이스를 불러왔습니다!")
    else:
        st.session_state.vector_store = None

# ------------------------------------------------------------------
# 4. 사이드바 파일 업로드 처리 (지식베이스 하드디스크 저장)
# ------------------------------------------------------------------
if uploaded_file is not None and st.session_state.vector_store is None:
    with st.sidebar:
        with st.spinner("문서를 읽고 지식을 추출하는 중..."):
            raw_text = uploaded_file.read().decode("utf-8")
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_text(raw_text)
            
            # FAISS DB 생성 및 파일로 영구 저장
            st.session_state.vector_store = FAISS.from_texts(texts, embeddings)
            st.session_state.vector_store.save_local(INDEX_DIR)
            st.success("✅ 지식베이스 학습 및 저장 완료!")

# ------------------------------------------------------------------
# 5. 대화 화면 렌더링 및 입력 처리
# ------------------------------------------------------------------
# 기존 대화 기록 화면에 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if user_query := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                # ✨ 구글 실시간 검색 엔진 툴을 Gemini에 장착합니다.
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0.3,
                    tools=[{"google_search": {}}] # <--- 구글 검색 기능 활성화!
                )
                
                if st.session_state.vector_store is not None:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    # ✨ 프롬프트 지침 완화: 내 문서를 베이스로 하되, 최신/부가 정보는 인터넷 검색을 활용하도록 유도
                    system_prompt = (
                        "당신은 스마트하고 친절한 지식 비서입니다.\n"
                        "1. 먼저 아래 제공된 '참고할 문서 내용'을 최우선으로 확인하여 답변을 작성하세요.\n"
                        "2. 만약 문서 내용이 부족하거나 최신 정보, 부가적인 설명이 필요하다면 구글 검색 기능을 활용하여 내용을 보완하세요.\n"
                        "3. 사용자가 언제 작성된 정보인지 묻는다면 실시간 검색 결과를 바탕으로 현재가 2026년임을 인지하고 답변하세요.\n\n"
                        f"참고할 문서 내용:\n{context_text}"
                    )
                    
                    prompt_messages = [("system", system_prompt)]
                    for h in st.session_state.chat_history[:-1]:
                        prompt_messages.append((h["role"], h["content"]))
                    prompt_messages.append(("human", user_query))
                    
                    prompt = ChatPromptTemplate.from_messages(prompt_messages)
                    chain = prompt | llm
                    response = chain.invoke({})
                    answer = response.content
                else:
                    # 지식베이스가 없을 때도 구글 검색을 활용해 답변할 수 있도록 처리
                    system_prompt = "당신은 구글 검색을 활용하여 사용자의 질문에 정확하게 답하는 비서입니다."
                    prompt_messages = [("system", system_prompt)]
                    for h in st.session_state.chat_history[:-1]:
                        prompt_messages.append((h["role"], h["content"]))
                    prompt_messages.append(("human", user_query))
                    
                    prompt = ChatPromptTemplate.from_messages(prompt_messages)
                    chain = prompt | llm
                    response = chain.invoke({})
                    answer = response.content
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                if remember_chat:
                    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=4)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")