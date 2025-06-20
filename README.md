# AWS RAG 시스템

PDF 문서 기반 질의응답 시스템

## 🚀 빠른 시작

### 1. 설정 파일 수정
`config.py`에서 본인 환경에 맞게 수정:
```python
# AWS 프로필
AWS_PROFILE = 'your-profile'

# OpenSearch 정보
OPENSEARCH_ENDPOINT = 'your-opensearch-endpoint'
OPENSEARCH_USERNAME = 'your-username'
OPENSEARCH_PASSWORD = 'your-password'
OPENSEARCH_INDEX_NAME = 'your-index-name'
```

### 2. 백엔드 실행
main.py의 191번 라인에서 사용할 파일 선택
pdf_path = './santafe.pdf' 또는 pdf_path = './crob_santafe.pdf'

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 3. 프론트엔드 실행
```bash
chmod +x start_service.sh
./start_service.sh
```

## 📝 주요 기능
- PDF 문서 처리 및 벡터 인덱싱
- AWS Bedrock 기반 질의응답
- Streamlit 웹 인터페이스
- OpenSearch 벡터 검색

## 🛠 주요 파일
- `config.py`: 환경 설정
- `main.py`: 백엔드 시스템 실행
- `streamlit_app.py`: 웹 인터페이스
- `start_service.sh`: 프론트엔드 실행 스크립트 