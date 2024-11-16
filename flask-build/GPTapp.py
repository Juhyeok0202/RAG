from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

import numpy as np
import faiss  # Facebook의 FAISS 라이브러리 임포트
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# OpenAI API 키 설정
OPENAI_API_KEY = "{YOUR_API_KEY}"

# OpenAI GPT 모델 설정
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=1024,
    openai_api_key=OPENAI_API_KEY
)

FILE_NAME = "query_answer_dataset_utf8.csv"

# JSON 파일 로드
FILE_PATH = "sample_data_for_vectorDB/v2.json"
print(FILE_PATH)

# JSONLoader로 데이터 로드 (json 내 전체 데이터를 metadata로 저장)
loader = JSONLoader(FILE_PATH, jq_schema=".[]", text_content=False)
documents = loader.load()

# 임베딩 모델 로드
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 각 문서의 page_content를 텍스트로 추출하고 임베딩 생성
texts = [doc.page_content for doc in documents]
embeddings = embedding_model.encode(texts)  # 각 문서에 대한 임베딩 생성

# FAISS 인덱스 생성 및 데이터 추가
dimension = len(embeddings[0])  # 벡터의 차원 수
print(f'벡터의 차원수 {dimension}')
faiss_index = faiss.IndexFlatL2(dimension)  # L2 거리 기준의 인덱스 생성
faiss_index.add(np.array(embeddings).astype('float32'))  # FAISS 인덱스에 임베딩 추가

# LangChain FAISS로 래핑해서 사용하기 위함
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
    try:
        retriever = vectorstore.as_retriever()  # 여기서 prompt가 임베딩 되어 문서와 비교됨

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
        print("llm_chain 생성 완료")
        # LLM 체인 호출
        response = llm_chain.invoke(question)  # invoke() 사용
        print(f"Raw response: {response}")  # 디버깅용 반환값 확인

        # 반환값이 문자열일 경우 그대로 반환
        if isinstance(response, str):
            return response

        # 반환값이 사전일 경우 'result' 키 확인
        elif isinstance(response, dict) and 'result' in response:
            return response['result']

        # 예기치 않은 형식의 반환값 처리
        else:
            raise ValueError(f"Unexpected response format: {response}")
    except Exception as e:
        raise ValueError(f"Error in get_answer: {str(e)}")


################################FOR FLASK####################################

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        print(f"Request data: {data}")  # 클라이언트 요청 데이터 디버깅
        question = data.get("question")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        answer = get_answer(question)
        print(f"Generated answer: {answer}")  # 생성된 응답 디버깅
        return jsonify({"answer": answer})
    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({"error": f"ValueError: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
