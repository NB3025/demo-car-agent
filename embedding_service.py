import boto3
import json
import logging
from typing import List, Dict, Any
import numpy as np
from botocore.exceptions import ClientError

from config import (
    AWS_REGION, 
    BEDROCK_REGION, 
    EMBEDDING_MODEL_ID,
    AWS_PROFILE,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockEmbeddingService:
    """AWS Bedrock을 사용한 임베딩 서비스"""
    
    def __init__(self):
        """Bedrock 클라이언트 초기화"""
        try:
            # AWS 세션 설정 (프로필 또는 환경변수 사용)
            if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
                # 환경변수가 설정되어 있으면 사용
                session = boto3.Session(
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=BEDROCK_REGION
                )
            else:
                # AWS 프로필 사용
                session = boto3.Session(
                    profile_name=AWS_PROFILE,
                    region_name=BEDROCK_REGION
                )
            
            self.bedrock_runtime = session.client(
                service_name='bedrock-runtime',
                region_name=BEDROCK_REGION
            )
            
            logger.info(f"Bedrock 클라이언트 초기화 완료 (Region: {BEDROCK_REGION})")
            
        except Exception as e:
            logger.error(f"Bedrock 클라이언트 초기화 실패: {str(e)}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """단일 텍스트에 대한 임베딩을 생성합니다."""
        try:
            # 텍스트 전처리
            cleaned_text = self._preprocess_text(text)
            
            # Bedrock 요청 본문 구성
            body = json.dumps({
                "inputText": cleaned_text,
                "dimensions": 1024,  # Titan Embed Text v2의 기본 차원
                "normalize": True
            })
            
            # Bedrock API 호출
            response = self.bedrock_runtime.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # 응답 파싱
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            
            if not embedding:
                raise ValueError("임베딩 생성 실패: 빈 응답")
            
            return embedding
            
        except ClientError as e:
            logger.error(f"Bedrock API 호출 실패: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류: {str(e)}")
            raise
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """여러 텍스트에 대한 임베딩을 배치로 생성합니다."""
        logger.info(f"{len(texts)}개 텍스트에 대한 임베딩 생성 시작")
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    embedding = self.create_embedding(text)
                    batch_embeddings.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"텍스트 임베딩 실패 (인덱스 {i}): {str(e)}")
                    # 실패한 경우 영벡터로 대체
                    batch_embeddings.append([0.0] * 1024)
            
            embeddings.extend(batch_embeddings)
            logger.info(f"배치 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} 완료")
        
        logger.info(f"총 {len(embeddings)}개 임베딩 생성 완료")
        return embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text or not text.strip():
            return ""
        
        # 텍스트 길이 제한 (Titan Embed Text v2는 최대 8192 토큰)
        max_chars = 30000  # 대략적인 문자 수 제한
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.warning(f"텍스트가 너무 길어 {max_chars}자로 잘림")
        
        # 불필요한 공백 제거
        text = ' '.join(text.split())
        
        return text
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """두 임베딩 간의 코사인 유사도를 계산합니다."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 코사인 유사도 계산
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"유사도 계산 실패: {str(e)}")
            return 0.0
    
    def test_connection(self) -> bool:
        """Bedrock 연결 테스트"""
        try:
            test_text = "연결 테스트"
            embedding = self.create_embedding(test_text)
            
            if embedding and len(embedding) == 1024:
                logger.info("Bedrock 연결 테스트 성공")
                return True
            else:
                logger.error("Bedrock 연결 테스트 실패: 잘못된 임베딩 형식")
                return False
                
        except Exception as e:
            logger.error(f"Bedrock 연결 테스트 실패: {str(e)}")
            return False 