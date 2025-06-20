import logging
from typing import List, Dict, Any, Optional
import json
import boto3
from datetime import datetime

from document_processor import DocumentProcessor, DocumentChunk
from embedding_service import BedrockEmbeddingService
from vector_store import OpenSearchVectorStore
from config import (
    TEST_PDF_PATH,
    FULL_PDF_PATH,
    LLM_MODEL_ID,
    BEDROCK_REGION,
    AWS_PROFILE,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """통합 RAG 시스템"""
    
    def __init__(self):
        """RAG 시스템 초기화"""
        try:
            # 각 컴포넌트 초기화
            self.document_processor = DocumentProcessor()
            self.embedding_service = BedrockEmbeddingService()
            self.vector_store = OpenSearchVectorStore()
            
            # Bedrock 클라이언트 (LLM용)
            if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
                session = boto3.Session(
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=BEDROCK_REGION
                )
            else:
                session = boto3.Session(
                    profile_name=AWS_PROFILE,
                    region_name=BEDROCK_REGION
                )
            
            self.bedrock_runtime = session.client(
                service_name='bedrock-runtime',
                region_name=BEDROCK_REGION
            )
            
            logger.info("RAG 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 실패: {str(e)}")
            raise
    
    def setup_system(self) -> bool:
        """시스템 초기 설정 (인덱스 생성 등)"""
        try:
            logger.info("RAG 시스템 설정 시작")
            
            # 연결 테스트
            if not self.embedding_service.test_connection():
                logger.error("Bedrock 임베딩 서비스 연결 실패")
                return False
            
            # 벡터 인덱스 생성
            if not self.vector_store.create_index():
                logger.error("OpenSearch 인덱스 생성 실패")
                return False
            
            logger.info("RAG 시스템 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 설정 중 오류: {str(e)}")
            return False
    
    def process_and_index_document(self, pdf_path: str) -> bool:
        """문서를 처리하고 인덱싱합니다."""
        try:
            logger.info(f"문서 처리 및 인덱싱 시작: {pdf_path}")
            
            # 1. 문서 처리 (파싱 및 청킹)
            chunks = self.document_processor.process_document(pdf_path)
            
            if not chunks:
                logger.error("문서 청킹 실패")
                return False
            
            logger.info(f"{len(chunks)}개 청크 생성 완료")
            
            # 2. 임베딩 생성
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.create_embeddings_batch(texts)
            
            if len(embeddings) != len(chunks):
                logger.error("임베딩 생성 실패")
                return False
            
            logger.info(f"{len(embeddings)}개 임베딩 생성 완료")
            
            # 3. 벡터 저장소에 인덱싱
            success = self.vector_store.add_documents(chunks, embeddings)
            
            if success:
                logger.info("문서 인덱싱 완료")
                return True
            else:
                logger.error("문서 인덱싱 실패")
                return False
                
        except Exception as e:
            logger.error(f"문서 처리 및 인덱싱 중 오류: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5, search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """질의에 대한 검색을 수행합니다."""
        try:
            logger.info(f"검색 시작: '{query}' (타입: {search_type})")
            
            # 질의 임베딩 생성
            query_embedding = self.embedding_service.create_embedding(query)
            
            if not query_embedding:
                logger.error("질의 임베딩 생성 실패")
                return []
            
            # 검색 수행
            if search_type == "vector":
                results = self.vector_store.search(query_embedding, k)
            elif search_type == "hybrid":
                results = self.vector_store.hybrid_search(query, query_embedding, k)
            else:
                logger.error(f"지원하지 않는 검색 타입: {search_type}")
                return []
            
            logger.info(f"검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"검색 중 오류: {str(e)}")
            return []
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]], 
                       max_context_length: int = 4000) -> str:
        """검색 결과를 바탕으로 답변을 생성합니다."""
        try:
            if not search_results:
                return "관련 정보를 찾을 수 없습니다."
            
            # 컨텍스트 구성
            context_parts = []
            current_length = 0
            
            for result in search_results:
                content = result['content']
                page_info = f"(페이지 {result['page_number']})"
                
                part = f"{content} {page_info}"
                
                if current_length + len(part) > max_context_length:
                    break
                
                context_parts.append(part)
                current_length += len(part)
            
            context = "\n\n".join(context_parts)
            
            # 프롬프트 구성
            prompt = f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.

문서 내용:
{context}

질문: {query}

답변 시 다음 사항을 지켜주세요:
1. 문서 내용에 기반하여 정확하게 답변하세요
2. 답변 근거가 되는 페이지 번호를 포함하세요
3. 문서에 없는 내용은 추측하지 마세요
4. 한국어로 자연스럽게 답변하세요

답변:"""
            
            # Claude 모델 호출
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=LLM_MODEL_ID,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            answer = response_body['content'][0]['text']
            
            logger.info("답변 생성 완료")
            return answer
            
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {str(e)}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def query(self, question: str, k: int = 5, search_type: str = "hybrid") -> Dict[str, Any]:
        """질의응답을 수행합니다."""
        try:
            start_time = datetime.now()
            
            # 검색 수행
            search_results = self.search(question, k, search_type)
            
            if not search_results:
                return {
                    "question": question,
                    "answer": "관련 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "search_time": (datetime.now() - start_time).total_seconds()
                }
            
            # 답변 생성
            answer = self.generate_answer(question, search_results)
            
            # 소스 정보 구성
            sources = []
            for result in search_results:
                source = {
                    "content": result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                    "page_number": result['page_number'],
                    "score": result['score'],
                    "section_type": result['section_type'],
                    "has_images": result['has_images']
                }
                sources.append(source)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "search_time": total_time,
                "search_type": search_type,
                "results_count": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"질의응답 중 오류: {str(e)}")
            return {
                "question": question,
                "answer": f"오류가 발생했습니다: {str(e)}",
                "sources": [],
                "search_time": 0,
                "error": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태를 반환합니다."""
        try:
            # Bedrock 연결 테스트
            bedrock_status = self.embedding_service.test_connection()
            
            # OpenSearch 인덱스 통계
            index_stats = self.vector_store.get_index_stats()
            
            return {
                "bedrock_status": "connected" if bedrock_status else "disconnected",
                "opensearch_status": index_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 확인 중 오류: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def reset_system(self) -> bool:
        """시스템을 초기화합니다 (인덱스 삭제 후 재생성)."""
        try:
            logger.info("시스템 초기화 시작")
            
            # 인덱스 삭제
            if not self.vector_store.delete_index():
                logger.error("인덱스 삭제 실패")
                return False
            
            # 인덱스 재생성
            if not self.vector_store.create_index():
                logger.error("인덱스 재생성 실패")
                return False
            
            logger.info("시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 중 오류: {str(e)}")
            return False 