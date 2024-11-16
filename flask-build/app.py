from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import JSONLoader, CSVLoader  # 수정된 경로
from langchain_huggingface import HuggingFaceEmbeddings      # 수정된 경로
from sentence_transformers import SentenceTransformer

import numpy as np
import faiss  # Facebook의 FAISS 라이브러리 임포트
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

ANTHROPIC_API_KEY = "{YOUR_API_KEY}"

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key=ANTHROPIC_API_KEY,
    # base_url="...",
    # other params...
)

# JSON 파일 로드
FILE_PATH = "sample_data_for_vectorDB/v2.json"
print(FILE_PATH)
# # JSONLoader로 데이터 로드 (json 내 전체 데이터를 metadata로 저장)
loader = JSONLoader(FILE_PATH, jq_schema=".[]", text_content=False)
documents = loader.load()

# 각 행을 `{query: answer}` 딕셔너리로 구성하여 documents 리스트 생성
# documents = []
# for row in csv_data:
#     query = row.metadata['query']  # CSV의 Query 컬럼
#     answer = row.page_content       # CSV의 Answer 컬럼
#     documents.append({"page_content": f"{query}: {answer}"})

# 임베딩 모델 로드
# embedding_model = HuggingFaceEmbeddings(
#     model_name="jhgan/ko-sroberta-multitasks")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 각 문서의 page_content를 텍스트로 추출하고 임베딩 생성
texts = [doc.page_content for doc in documents]
# embeddings = embedding_model.embed_documents(texts)  # 각 문서에 대한 임베딩 생성
embeddings = embedding_model.encode(texts)  # 각 문서에 대한 임베딩 생성

# FAISS 인덱스 생성 및 데이터 추가
dimension = len(embeddings[0])  # 벡터의 차원 수
print(f'벡터의 차원수 {dimension}')
faiss_index = faiss.IndexFlatL2(dimension)  # L2 거리 기준의 인덱스 생성
faiss_index.add(np.array(embeddings).astype('float32'))  # FAISS 인덱스에 임베딩 추가

# LangChain FAISS로 래핑해서 사용하기 위함
# vectorstore = FAISS(embeddings=embedding_model, index=faiss_index)

# 각 문서의 ID를 0, 1, 2, ... 형태로 지정
index_to_docstore_id = {i: str(i) for i in range(len(documents))}
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

vectorstore = FAISS(
    embedding_function=embedding_model,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)
print(f"Added {faiss_index.ntotal} documents to FAISS index.")


def get_answer(question):
    # Step 3: Retriever 설정
    # (search_kwargs={"k": 10}) 디폴트 k=5
    retriever = vectorstore.as_retriever() # 여기서 prompt가 임베딩 되어 문서와 비교됨

    # Step 4: 프롬프트 템플릿 설정
    template = """
    {query}
    """

    prompt = PromptTemplate.from_template(template)

    # LLMChain을 통해 프롬프트가 포함된 체인 생성
    # llm_chain_with_prompt = LLMChain(prompt=prompt, llm=llm)
    # llm_chain_with_prompt = LLMChain(prompt|llm)

    # Step 5: RetrievalQA 체인 설정
    llm_chain = RetrievalQA.from_chain_type(
      llm=llm,  # LLMChain 자체가 아니라 LLM을 직접 전달
      chain_type="stuff",  # 검색된 문서를 간단히 LLM에 전달
      retriever=retriever,
      return_source_documents=True,
    )

    # Step 6: 질문을 통해 RAG 실행
    response = llm_chain.invoke(question)  # 딕셔너리 형식으로 호출
    print(f"Raw response: {response}")  # 반환값 확인

    return response['result']


################################FOR FLASK####################################

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question")
    if question:
        try:
            answer = get_answer(question)  # AI 모델 응답 함수 호출
            print(answer)
            return jsonify({"answer": answer})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No question provided"}), 400

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
