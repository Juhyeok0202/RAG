# 베이스 이미지로 Python 3.8 사용
FROM python:3.8-slim

# 작업 디렉토리 생성
WORKDIR /app

# 필요한 파일 복사
COPY . /app

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 서버의 포트를 열기
EXPOSE 5000

# Flask 서버 실행
CMD ["python", "flask-build/app.py"]
