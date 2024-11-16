# 사용법
※RAG.PY를 사용합니다. 나머지는 버리는 코드이며, 혹시 몰라 백업목적으로 같이 업로드합니다.

모델 키를 등록합니다.
모델을 선택합니다.
임베딩 모델을 선택합니다.
Flask로 배포하여 localhost:5000 에 postman으로 프롬프트 api를 테스트합니다.
![image](https://github.com/user-attachments/assets/750fa5c9-2976-4f3a-8503-96bc2f27d3b7)

## VectorDB에 데이터를 구성해야합니다.
현 코드는 Q A 컬럼을 가진 CSV를 임베딩하여 VectorDB에 넣어 FAISS vectorDB를 구성합니다.

## 임베딩 모델
한국어 임베딩 모델에서 가장 우수한 성능을 가진 HuggingSpace의 jhgan/ko-sroberta-multitask 을 사용합니다.

## 언어모델
언어 모델은 GPT4는 비싸므로, 한국어 모델 중에 가장 우수한 야놀자의 EEVE모델을 사용합니다.
(현재는 로컬 CPU환경에서 테스트 코드를 돌리기 위해 GPT API보다 값싼 Clude모델을 사용합니다.)

