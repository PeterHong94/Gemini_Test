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
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                
                if st.session_state.vector_store is not None:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    system_prompt = (
                        "당신은 주어진 참고 문서를 바탕으로 답변하는 비서입니다.\n"
                        "만약 문서에 없는 내용이라면 모른다고 답하세요.\n\n"
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
                    answer = "사이드바에 텍스트 파일을 업로드해 주시면, 해당 데이터를 기반으로 한 RAG 대화가 가능합니다!"
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # 📌 대화 저장 로직: '현재 대화 기록 기억하기' 토글이 켜져 있을 때만 파일로 저장
                if remember_chat:
                    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=4)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")