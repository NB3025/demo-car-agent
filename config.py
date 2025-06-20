import os
from dotenv import load_dotenv

load_dotenv()

# AWS 설정
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')
AWS_PROFILE = os.getenv('AWS_PROFILE', 'profile')

# 환경 변수로 설정하지 않으면 기본값 사용
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Bedrock 설정
BEDROCK_REGION = os.getenv('BEDROCK_REGION', 'us-west-2')
EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'
LLM_MODEL_ID = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'

# OpenSearch 설정
# OpenSearch Endpoint 입력 예시 : https://jblo4yloneq.us-west-2.es.amazonaws.com
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT', 'OpenSearch Endpoint')
# OpenSearch Username 입력 예시 : user-01
OPENSEARCH_USERNAME = os.getenv('OPENSEARCH_USERNAME', 'OpenSearch Username')
# OpenSearch Password 입력 예시 : password!
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', 'OpenSearch Password')
# OpenSearch Index Name 입력 예시 : santafe-manual-index
OPENSEARCH_INDEX_NAME = os.getenv('OPENSEARCH_INDEX_NAME', 'OpenSearch Index Name')

# 문서 처리 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS_PER_CHUNK = 512

# 파일 경로
TEST_PDF_PATH = './crob_santafe.pdf'
FULL_PDF_PATH = './santafe.pdf' 