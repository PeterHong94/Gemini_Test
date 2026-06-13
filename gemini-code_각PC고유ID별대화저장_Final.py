import streamlit as st
import os
import json
import hashlib  # ✨ IP 주소를 깨끗한 문자열 ID로 암호화하기 위해 추가
import pandas as pd
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 고정 경로 정의
INDEX_DIR = "my_faiss_index"

st.set_page_config(page_title="IP 기억형 RAG 챗봇", page_icon="🤖")
st.title("🤖 IP 주소 기반 자동 기억 RAG 챗봇")

# API 키 설정 (본인의 키로 대체하세요)
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY" 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ------------------------------------------------------------------
# ✨ [핵심 변경] 접속자의 IP/네트워크 정보를 추출하여 고유 ID 생성
# ------------------------------------------------------------------
if "user_session_id" not in st.session_state:
    try:
        # Streamlit에 접속한 클라이언트의 네트워크 헤더 정보 추출
        # 대다수의 프록시 및 로컬 환경에서 클라이언트 식별용 주소를 가져옵니다.
        headers = st.context.headers
        client_ip = headers.get("X-Forwarded-For", headers.get("Remote-Addr", "unknown_ip"))
        
        # 만약 로컬(localhost) 테스트 중이라 주소가 안 잡히면 보완책 적용
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()
            
        # IP 주소 그대로 파일명을 만들면 특수문자(.이나 :) 때문에 에러가 날 수 있으므로,
        # hashlib를 사용해 안전한 8자리 문자열(해시값)로 변환합니다.
        ip_hash = hashlib.md5(client_ip.encode("utf-8")).hexdigest()[:8]
        st.session_state.user_session_id = f"user_{ip_hash}"
    except Exception:
        # 혹시 모를 예외 발생 시 시스템 꼬임 방지용 기본값
        st.session_state.user_session_id = "user_default"

# 이 사용자의 IP에 묶이는 고유 대화 파일명 정의
USER_HISTORY_FILE = f"chat_history_{st.session_state.user_session_id}.json"
# ------------------------------------------------------------------

# 2. 사이드바 컨트롤 영역
with st.sidebar:
    st.header("⚙️ 챗봇 설정 컨트롤러")
    
    # 📌 현재 내 네트워크 기반 고유 ID 확인
    st.info(f"💻 내 PC 네트워크 ID: **{st.session_state.user_session_id}**")
    st.caption("※ 같은 네트워크 환경(PC)이라면 새로고침해도 이 ID가 유지됩니다.")
    
    remember_chat = st.toggle("현재 대화 기록 기억하기", value=True, 
                              help="이 기능을 켜면 이 PC에서의 대화 내용이 노트북에 영구 저장되어 다음에 앱을 켜도 이어집니다.")
    
    st.markdown("---")
    st.header("📂 데이터 지식베이스 구축")
    text_file = st.file_uploader("기본 참고 문서 (.txt)", type=["txt"], key="txt_file")
    data_file_1 = st.file_uploader("데이터 파일 1 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="data_1")
    data_file_2 = st.file_uploader("데이터 파일 2 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="data_2")
    train_button = st.button("🚀 모든 파일 결합하여 지식베이스 구축")

# 3. 데이터 로딩 및 초기화 (앱 시작 시 1회 실행)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ① 대화 기록 로드 (오직 내 IP 고유 파일에서만 읽어옴)
if "chat_history" not in st.session_state:
    if remember_chat and os.path.exists(USER_HISTORY_FILE):
        with open(USER_HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.chat_history = json.load(f)
        st.sidebar.success("💬 이전 대화 기록을 자동으로 불러왔습니다!")
    else:
        st.session_state.chat_history = []

# ② 지식베이스(FAISS DB) 로드
if "vector_store" not in st.session_state:
    if os.path.exists(INDEX_DIR):
        st.session_state.vector_store = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        st.sidebar.info("💾 기존에 저장된 공용 지식베이스를 불러왔습니다!")
    else:
        st.session_state.vector_store = None

# 4. 모든 업로드 파일 데이터 통합 가공 및 RAG 학습
if train_button:
    if text_file is not None:
        with st.sidebar:
            with st.spinner("업로드된 모든 파일을 분석하고 통합하는 중..."):
                all_texts = []
                
                raw_text = text_file.read().decode("utf-8")
                all_texts.append(f"--- [기본 참고 문서: {text_file.name}] ---\n" + raw_text)
                
                for i, file in enumerate([data_file_1, data_file_2], start=1):
                    if file is not None:
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_excel(file)
                        
                        for index, row in df.iterrows():
                            row_details = [f"[{col}]: {val}" for col, val in row.items() if pd.notna(val)]
                            row_string = f"데이터 파일 {i}({file.name}) - {index+1}번째 행: " + ", ".join(row_details)
                            all_texts.append(row_string)
                
                combined_text = "\n\n".join(all_texts)
                text_splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=100)
                texts = text_splitter.split_text(combined_text)
                
                st.session_state.vector_store = FAISS.from_texts(texts, embeddings)
                st.session_state.vector_store.save_local(INDEX_DIR)
                st.success("✅ 공용 지식베이스 구축 완료!")
                st.rerun()
    else:
        st.sidebar.error("⚠️ 최소한 '기본 참고 문서(.txt)' 파일은 업로드해야 학습이 가능합니다.")

# 5. 대화 화면 렌더링 및 입력 처리
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0.3,
                    tools=[{"google_search": {}}]
                )
                
                if st.session_state.vector_store is not None:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6})
                    docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    system_prompt = (
                        "당신은 제공된 문서 정보와 표 형식 데이터(Excel/CSV)를 함께 융합하여 분석하는 인공지능 비서입니다.\n"
                        "1. 아래 '통합 참고 내용'을 기반으로 사용자의 질문에 가장 먼저 대답하세요.\n"
                        "2. 일반 서술형 문서 내용과 표 데이터 수치를 상호 교차 분석하여 정확하게 인용해 주세요.\n"
                        "3. 내부 정보만으로 답변이 어렵거나 외부의 실시간 정보가 필요하다면 구글 검색 기능을 활용하여 정교하게 내용을 보완하세요.\n"
                        "4. 현재 연도는 2026년입니다.\n\n"
                        f"통합 참고 내용:\n{context_text}"
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
                    answer = "사이드바에서 파일을 업로드하고 [지식베이스 구축] 버튼을 누르면 파일 기반 분석 대화가 가능합니다!"
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                if remember_chat:
                    with open(USER_HISTORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=4)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")