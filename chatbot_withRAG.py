import streamlit as st
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 페이지 설정 및 API 키 확인
st.set_page_config(page_title="나만의 RAG 챗봇", page_icon="🤖")
st.title("🤖 최신 표준 RAG 챗봇 (버전 에러 해결)")

# 여기에 본인의 Gemini API 키를 입력하거나 윈도우 환경변수에 등록하세요.
GOOGLE_API_KEY = "AIzaSyAnesZNt_tBAe5Qr7VybRyFj5PaiVadaYs" 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. 세션 상태(Session State) 초기화 - 대화 기록 및 벡터 DB 유지
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# ✨ [수정] 앱이 실행될 때 로컬 하드디스크에 저장된 지식이 있는지 확인하고 자동 로드
if "vector_store" not in st.session_state:
    if os.path.exists("my_faiss_index"):
        # 저장된 폴더가 있다면 파일에서 읽어와 뇌에 장착
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        st.session_state.vector_store = FAISS.load_local(
            "my_faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        st.sidebar.info("💾 기존에 저장된 지식베이스를 성공적으로 불러왔습니다!")
    else:
        st.session_state.vector_store = None

# 3. 사이드바: 학습시킬 데이터(텍스트 파일) 업로드 기능
with st.sidebar:
    st.header("📂 데이터 지식베이스")
    uploaded_file = st.file_uploader("챗봇에게 학습시킬 텍스트(.txt) 파일을 올려주세요", type=["txt"])
    
    if uploaded_file is not None and st.session_state.vector_store is None:
        with st.spinner("문서를 읽고 지식을 추출하는 중..."):
            # 파일 읽기 및 텍스트 쪼개기 (Chunking)
            raw_text = uploaded_file.read().decode("utf-8")
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_text(raw_text)
            
            # Gemini 임베딩 모델을 사용하여 벡터 저장소(FAISS DB) 구축
            #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            # 최신 구글 권장 임베딩 모델명으로 변경
            #embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            # 기존: model="models/text-embedding-004"
            #embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

            # [기존 코드] FAISS DB 구축
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            st.session_state.vector_store = FAISS.from_texts(texts, embeddings)

            # ✨ [추가] 생성된 지식베이스를 내 노트북의 'my_faiss_index'라는 폴더에 파일로 저장!
            st.session_state.vector_store.save_local("my_faiss_index")
            st.success("✅ 지식베이스 학습 및 로컬 저장 완료!")

# 4. 저장된 이전 대화 기록이 있다면 화면에 뿌려주기
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 사용자 채팅 입력 처리
if user_query := st.chat_input("질문을 입력하세요..."):
    # 사용자가 입력한 메시지 화면 표시 및 기록
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                # 두뇌 역할의 최신 Gemini 모델 선언
                #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                # 최신 표준 모델명인 gemini-2.5-flash로 교체
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                
                # 지식이 업로드 되었을 때와 아닐 때를 나누어 처리
                if st.session_state.vector_store is not None:
                    # ① 사용자의 질문과 관련된 문서 찾아오기 (유사도 검색)
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke(user_query)
                    
                    # ② 찾아온 문서들을 하나의 텍스트 맥락으로 합치기
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    # ③ 프롬프트 시스템 메시지에 데이터 주입
                    system_prompt = (
                        "당신은 주어진 참고 문서를 바탕으로 답변하는 비서입니다.\n"
                        "만약 문서에 없는 내용이라면 모른다고 답하세요.\n\n"
                        f"참고할 문서 내용:\n{context_text}"
                    )
                    
                    # ④ 과거 대화 이력과 현재 질문을 정렬
                    prompt_messages = [("system", system_prompt)]
                    for h in st.session_state.chat_history[:-1]: # 마지막 질문 제외한 과거 이력
                        prompt_messages.append((h["role"], h["content"]))
                    prompt_messages.append(("human", user_query))
                    
                    prompt = ChatPromptTemplate.from_messages(prompt_messages)
                    
                    # ⑤ 최신 표준 파이프라인 구조 (| 기호로 프롬프트와 모델 직접 연결)
                    chain = prompt | llm
                    response = chain.invoke({})
                    answer = response.content
                else:
                    # 데이터가 아직 업로드되지 않았다면 안내 메시지 출력
                    answer = "사이드바에 텍스트 파일을 업로드해 주시면, 해당 데이터를 기반으로 한 RAG 대화가 가능합니다!"
                
                # 결과 출력 및 기록
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")