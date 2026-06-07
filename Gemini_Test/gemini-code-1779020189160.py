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

st.set_page_config(page_title="데이터 마스터 RAG 챗봇", page_icon="📊")
st.title("📊 엑셀 & CSV 통합 분석 RAG 챗봇")

# API 키 설정 (본인의 키로 대체하세요)
GOOGLE_API_KEY = "AIzaSyAnesZNt_tBAe5Qr7VybRyFj5PaiVadaYs" 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ------------------------------------------------------------------
# 2. 사이드바 컨트롤 영역 (대화 기억 토글 + 엑셀/CSV 2개 업로드 메뉴)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 챗봇 설정 컨트롤러")
    remember_chat = st.toggle("현재 대화 기록 기억하기", value=True, 
                              help="이 기능을 켜면 대화 내용이 노트북에 영구 저장되어 다음에 앱을 켜도 이어집니다.")
    
    st.markdown("---")
    st.header("📂 데이터 지식베이스")
    st.write("분석할 파일 2개를 각각 업로드해 주세요. (Excel 또는 CSV 지원)")
    
    # 📌 확장자 허용 항목에 csv 추가
    file_a = st.file_uploader("첫 번째 파일 업로드", type=["xlsx", "xls", "csv"], key="data_file_a")
    file_b = st.file_uploader("두 번째 파일 업로드", type=["xlsx", "xls", "csv"], key="data_file_b")
    
    train_button = st.button("🚀 두 파일 결합하여 챗봇 학습시키기")

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
# 4. 파일 데이터 가공 및 RAG 학습 (버튼 클릭 시 실행)
# ------------------------------------------------------------------
if train_button:
    if file_a is not None and file_b is not None:
        with st.sidebar:
            with st.spinner("두 파일의 데이터를 분석하고 통합하는 중..."):
                all_texts = []
                
                # 각 파일을 순회하며 포맷에 맞게 텍스트 추출
                for i, file in enumerate([file_a, file_b], start=1):
                    # 파일명 확인하여 csv와 excel 분기 처리
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    
                    # 각 행(Row)을 문장 형태로 변환하여 AI 주입 데이터로 가공
                    for index, row in df.iterrows():
                        row_details = [f"[{col}]: {val}" for col, val in row.items() if pd.notna(val)]
                        row_string = f"파일 {i}({file.name}) - {index+1}번째 행 데이터: " + ", ".join(row_details)
                        all_texts.append(row_string)
                
                # 추출된 데이터를 텍스트 묶음으로 합친 뒤 쪼개기
                combined_text = "\n".join(all_texts)
                text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                texts = text_splitter.split_text(combined_text)
                
                # FAISS 벡터 스토어 생성 및 로컬 저장
                st.session_state.vector_store = FAISS.from_texts(texts, embeddings)
                st.session_state.vector_store.save_local(INDEX_DIR)
                st.success("✅ 엑셀/CSV 데이터 지식베이스 통합 학습 완료!")
                st.rerun()
    else:
        st.sidebar.error("⚠️ 학습을 진행하려면 두 개의 파일을 모두 업로드해야 합니다.")

# ------------------------------------------------------------------
# 5. 대화 화면 렌더링 및 입력 처리 (구글 실시간 검색 연동 포함)
# ------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("데이터에 대해 질문하거나 궁금한 점을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                # 구글 검색 엔진이 탑재된 제미나이 모델 선언
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0.3,
                    tools=[{"google_search": {}}]
                )
                
                if st.session_state.vector_store is not None:
                    # 사용자 질문과 밀접한 행 데이터 5개 검색
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    system_prompt = (
                        "당신은 업로드된 표 형식의 데이터(Excel/CSV)를 정교하게 분석하는 데이터 전문 비서입니다.\n"
                        "1. 제공된 '참고할 데이터 내용'을 면밀히 분석하여 최우선으로 답변을 작성하세요.\n"
                        "2. 데이터에 수치나 통계가 있다면 명확하게 인용하고, 필요시 비교/분석해 주세요.\n"
                        "3. 만약 내부 데이터 내용이 부족하거나 외부 정보(최신 트렌드, 시장 동향, 일반 상식 등)와 결합이 필요하다면 구글 검색 기능을 활용하여 풍부하게 답변을 보완하세요.\n"
                        "4. 현재 연도는 2026년입니다.\n\n"
                        f"참고할 데이터 내용:\n{context_text}"
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
                    answer = "사이드바에서 두 개의 파일(Excel/CSV)을 업로드하고 [학습시키기] 버튼을 누르면 데이터 분석 대화가 가능합니다!"
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                if remember_chat:
                    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=4)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")