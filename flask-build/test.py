from langchain_anthropic import ChatAnthropic
import pandas as pd
from langchain.schema import Document
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

ANTHROPIC_API_KEY="{YOUR_API_KEY}"

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

# CSV 파일 경로
FILE_PATH = "sample_data_for_vectorDB/qaset_컴퓨터구성_장태무.csv"

# CSV 파일 로드
data = pd.read_csv(FILE_PATH)

# 데이터 확인
print(f"Columns: {data.columns}")
print(f"데이터 개수: {len(data)}")

# 각 행을 LangChain의 Document 형식으로 변환
# 'q'와 'a' 컬럼을 사용하여 문서를 생성 (필요에 따라 조정 가능)
documents = [
    Document(
        page_content=f"Q: {row['q']} A: {row['a']}",
        metadata={"question": row['q'], "answer": row['a']}
    )
    for _, row in data.iterrows()
]

# 문서 길이 확인
print(f"문서 개수: {len(documents)}")

# 첫 3개 문서 내용 출력
for i, doc in enumerate(documents[:3]):
    print(f"\n문서 {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")


# 임베딩 모델 로드
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 각 문서의 page_content를 텍스트로 추출하고 임베딩 생성
texts = [doc.page_content for doc in documents]
embeddings = embedding_model.embed_documents(texts)  # 각 문서에 대한 임베딩 생성

# FAISS 인덱스 생성 및 데이터 추가
dimension = len(embeddings[0])  # 벡터의 차원 수
print(f'벡터의 차원수 {dimension}')
faiss_index = faiss.IndexFlatL2(dimension)  # L2 거리 기준의 인덱스 생성
faiss_index.add(np.array(embeddings).astype('float32'))  # FAISS 인덱스에 임베딩 추가

#LangChain FAISS로 래핑해서 사용하기 위함
#vectorstore = FAISS(embeddings=embedding_model, index=faiss_index)

# 각 문서의 ID를 0, 1, 2, ... 형태로 지정
index_to_docstore_id = {i: str(i) for i in range(len(documents))}
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

vectorstore = FAISS(
    embedding_function=embedding_model,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# 인덱스 확인
print(f"Added {faiss_index.ntotal} documents to FAISS index.")

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

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
  llm_chain_with_prompt = LLMChain(prompt=prompt, llm=llm)

  # Step 5: RetrievalQA 체인 설정
  llm_chain = RetrievalQA.from_chain_type(
      llm=llm,  # LLMChain 자체가 아니라 LLM을 직접 전달
      chain_type="stuff",  # 검색된 문서를 간단히 LLM에 전달
      retriever=retriever,
      return_source_documents=True,
  )

  # Step 6: 질문을 통해 RAG 실행
  #question = "화장품을 잘 판매할 인플루언서"  # 예시 질문
  response = llm_chain({"query": question})  # 딕셔너리 형식으로 호출

  return response['result']

print(get_answer("설명을 자세히 해주시는 강좌 추천좀해줘"))

  #print("\n[LLM의 답변]")
  #print(response['result'])  # 생성된 응답

  #print("\n\n[참조된 문서]\n")
  #print(response['source_documents'])  # 참조된 문서

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question")
    if question:
        try:
            answer = get_answer(question)  # AI 모델 응답 함수 호출
            return jsonify({"result": answer})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No question provided"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)