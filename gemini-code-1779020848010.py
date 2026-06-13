import streamlit as st
import os
import json
import pandas as pd
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 파일 및 경로 정의
HISTORY_FILE = "chat_history.json"
INDEX_DIR = "my_faiss_index"

st.set_page_config(page_title="만능 올인원 RAG 챗봇", page_icon="🤖")
st.title("🤖 텍스트 & 엑셀/CSV 통합 RAG 챗봇")

# API 키 설정 (본인의 키로 대체하세요)
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY" 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ------------------------------------------------------------------
# 2. 사이드바 컨트롤 영역 (대화 기억 토글 + 텍스트 1개 + 엑셀/CSV 2개)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 챗봇 설정 컨트롤러")
    remember_chat = st.toggle("현재 대화 기록 기억하기", value=True, 
                              help="이 기능을 켜면 대화 내용이 노트북에 영구 저장되어 다음에 앱을 켜도 이어집니다.")
    
    st.markdown("---")
    st.header("📂 데이터 지식베이스 구축")
    
    # ① 기존 일반 텍스트 문서 업로드 메뉴 유지
    st.subheader("1. 기본 지식 문서")
    text_file = st.file_uploader("기본 참고 문서 (.txt)", type=["txt"], key="txt_file")
    
    # ② 추가 요청하신 엑셀/CSV 파일 2개 업로드 메뉴
    st.subheader("2. 추가 데이터 파일 (선택)")
    data_file_1 = st.file_uploader("데이터 파일 1 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="data_1")
    data_file_2 = st.file_uploader("데이터 파일 2 (Excel/CSV)", type=["xlsx", "xls", "csv"], key="data_2")
    
    # 통합 학습 실행 버튼
    train_button = st.button("🚀 모든 파일 결합하여 지식베이스 구축")

# ------------------------------------------------------------------
# 3. 기존 데이터 자동 로딩 (앱 시작 시 1회 실행)
# ------------------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ① 대화 기록 로드
if "chat_history" not in st.session_state:
    if remember_chat and os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.chat_history = json.load(f)
        st.sidebar.success("💬 이전 대화 기록을 불러왔습니다!")
    else:
        st.session_state.chat_history = []

# ② 지식베이스(FAISS DB) 로드
if "vector_store" not in st.session_state:
    if os.path.exists(INDEX_DIR):
        st.session_state.vector_store = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        st.sidebar.info("💾 기존에 저장된 데이터 지식베이스를 불러왔습니다!")
    else:
        st.session_state.vector_store = None

# ------------------------------------------------------------------
# 4. 모든 업로드 파일 데이터 통합 가공 및 RAG 학습
# ------------------------------------------------------------------
if train_button:
    # 최소한 기본 텍스트 파일은 등록되어야 학습 시작
    if text_file is not None:
        with st.sidebar:
            with st.spinner("업로드된 모든 파일을 분석하고 통합하는 중..."):
                all_texts = []
                
                # [처리 1] 기본 텍스트 파일(.txt) 읽기 및 처리
                raw_text = text_file.read().decode("utf-8")
                all_texts.append(f"--- [기본 참고 문서: {text_file.name}] ---\n" + raw_text)
                
                # [처리 2] 데이터 파일 1, 2가 업로드되었다면 추가 처리
                for i, file in enumerate([data_file_1, data_file_2], start=1):
                    if file is not None:
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_excel(file)
                        
                        # 행(Row) 데이터를 문장 형태로 가공하여 리스트에 추가
                        for index, row in df.iterrows():
                            row_details = [f"[{col}]: {val}" for col, val in row.items() if pd.notna(val)]
                            row_string = f"데이터 파일 {i}({file.name}) - {index+1}번째 행: " + ", ".join(row_details)
                            all_texts.append(row_string)
                
                # 모아진 모든 종류의 지식 데이터를 하나로 결합 후 청킹(Chunking)
                combined_text = "\n\n".join(all_texts)
                text_splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=100)
                texts = text_splitter.split_text(combined_text)
                
                # FAISS 벡터 스토어 통합 생성 및 로컬 저장
                st.session_state.vector_store = FAISS.from_texts(texts, embeddings)
                st.session_state.vector_store.save_local(INDEX_DIR)
                st.success("✅ 문서 및 엑셀/CSV 통합 지식베이스 구축 완료!")
                st.rerun()
    else:
        st.sidebar.error("⚠️ 최소한 '기본 참고 문서(.txt)' 파일은 업로드해야 학습이 가능합니다.")

# ------------------------------------------------------------------
# 5. 대화 화면 렌더링 및 입력 처리 (구글 실시간 검색 연동 포함)
# ------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("문서와 데이터에 대해 자유롭게 질문하세요..."):
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
                    # 질문과 가장 밀접한 지식 조각 6개 추출
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
                    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=4)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")