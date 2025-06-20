import boto3
import json
import logging
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import RequestError
import time

from config import (
    AWS_REGION,
    OPENSEARCH_ENDPOINT,
    OPENSEARCH_USERNAME,
    OPENSEARCH_PASSWORD,
    OPENSEARCH_INDEX_NAME,
    AWS_PROFILE,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY
)
from document_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearchVectorStore:
    """AWS OpenSearch를 사용한 벡터 저장소"""
    
    def __init__(self):
        """OpenSearch 클라이언트 초기화"""
        try:
            # OpenSearch 클라이언트 설정
            self.client = OpenSearch(
                hosts=[{'host': OPENSEARCH_ENDPOINT.replace('https://', ''), 'port': 443}],
                http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=60
            )
            
            self.index_name = OPENSEARCH_INDEX_NAME
            
            # 연결 테스트
            if self.client.ping():
                logger.info("OpenSearch 연결 성공")
            else:
                raise ConnectionError("OpenSearch 연결 실패")
                
        except Exception as e:
            logger.error(f"OpenSearch 클라이언트 초기화 실패: {str(e)}")
            raise
    
    def create_index(self, dimension: int = 1024) -> bool:
        """벡터 검색을 위한 인덱스를 생성합니다."""
        try:
            # 인덱스가 이미 존재하는지 확인
            if self.client.indices.exists(index=self.index_name):
                logger.info(f"인덱스 '{self.index_name}'가 이미 존재합니다.")
                return True
            
            # 인덱스 매핑 정의
            index_mapping = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100,
                        "number_of_shards": 1,
                        "number_of_replicas": 2
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "text"},
                                "author": {"type": "text"},
                                "subject": {"type": "text"},
                                "total_pages": {"type": "integer"},
                                "source_file": {"type": "keyword"},
                                "processing_timestamp": {"type": "date"}
                            }
                        },
                        "page_number": {"type": "integer"},
                        "chunk_id": {"type": "keyword"},
                        "section_type": {"type": "keyword"},
                        "has_images": {"type": "boolean"},
                        "image_descriptions": {"type": "text"}
                    }
                }
            }
            
            # 인덱스 생성
            response = self.client.indices.create(
                index=self.index_name,
                body=index_mapping
            )
            
            logger.info(f"인덱스 '{self.index_name}' 생성 완료")
            return True
            
        except RequestError as e:
            if e.error == 'resource_already_exists_exception':
                logger.info(f"인덱스 '{self.index_name}'가 이미 존재합니다.")
                return True
            else:
                logger.error(f"인덱스 생성 실패: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"인덱스 생성 중 오류: {str(e)}")
            return False
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        """문서 청크와 임베딩을 인덱스에 추가합니다."""
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("청크 수와 임베딩 수가 일치하지 않습니다.")
            
            logger.info(f"{len(chunks)}개 문서를 인덱스에 추가 시작")
            
            # 배치로 문서 추가
            batch_size = 100
            success_count = 0
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                # 배치 요청 구성
                bulk_body = []
                
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    # 인덱스 액션
                    bulk_body.append({
                        "index": {
                            "_index": self.index_name,
                            "_id": chunk.chunk_id
                        }
                    })
                    
                    # 문서 데이터
                    doc = {
                        "content": chunk.content,
                        "embedding": embedding,
                        "metadata": chunk.metadata,
                        "page_number": chunk.page_number,
                        "chunk_id": chunk.chunk_id,
                        "section_type": chunk.section_type,
                        "has_images": chunk.has_images,
                        "image_descriptions": chunk.image_descriptions
                    }
                    bulk_body.append(doc)
                
                # 배치 실행
                response = self.client.bulk(body=bulk_body)
                
                # 결과 확인
                if response.get('errors'):
                    for item in response['items']:
                        if 'index' in item and item['index'].get('status') not in [200, 201]:
                            logger.warning(f"문서 인덱싱 실패: {item['index']}")
                        else:
                            success_count += 1
                else:
                    success_count += len(batch_chunks)
                
                logger.info(f"배치 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} 완료")
            
            # 인덱스 새로고침
            self.client.indices.refresh(index=self.index_name)
            
            logger.info(f"총 {success_count}개 문서 인덱싱 완료")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"문서 추가 중 오류: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], k: int = 5, 
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """벡터 유사도 검색을 수행합니다."""
        try:
            # 기본 KNN 쿼리
            knn_query = {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            }
            
            # 필터 조건 추가
            if filters:
                bool_query = {
                    "bool": {
                        "must": [knn_query],
                        "filter": []
                    }
                }
                
                for field, value in filters.items():
                    if isinstance(value, list):
                        bool_query["bool"]["filter"].append({
                            "terms": {field: value}
                        })
                    else:
                        bool_query["bool"]["filter"].append({
                            "term": {field: value}
                        })
                
                search_query = bool_query
            else:
                search_query = knn_query
            
            # 검색 실행
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": search_query,
                    "size": k,
                    "_source": {
                        "excludes": ["embedding"]  # 임베딩은 제외하고 반환
                    }
                }
            )
            
            # 결과 파싱
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit['_score'],
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata'],
                    'page_number': hit['_source']['page_number'],
                    'chunk_id': hit['_source']['chunk_id'],
                    'section_type': hit['_source']['section_type'],
                    'has_images': hit['_source']['has_images'],
                    'image_descriptions': hit['_source']['image_descriptions']
                }
                results.append(result)
            
            logger.info(f"검색 완료: {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            logger.error(f"검색 중 오류: {str(e)}")
            return []
    
    def hybrid_search(self, query_text: str, query_embedding: List[float], 
                     k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """하이브리드 검색 (벡터 + 키워드)을 수행합니다."""
        try:
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": k,
                                        "boost": alpha
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["content^2", "image_descriptions"],
                                    "type": "best_fields",
                                    "boost": 1 - alpha
                                }
                            }
                        ]
                    }
                },
                "size": k,
                "_source": {
                    "excludes": ["embedding"]
                }
            }
            
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # 결과 파싱
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit['_score'],
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata'],
                    'page_number': hit['_source']['page_number'],
                    'chunk_id': hit['_source']['chunk_id'],
                    'section_type': hit['_source']['section_type'],
                    'has_images': hit['_source']['has_images'],
                    'image_descriptions': hit['_source']['image_descriptions']
                }
                results.append(result)
            
            logger.info(f"하이브리드 검색 완료: {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류: {str(e)}")
            return []
    
    def delete_index(self) -> bool:
        """인덱스를 삭제합니다."""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"인덱스 '{self.index_name}' 삭제 완료")
                return True
            else:
                logger.info(f"인덱스 '{self.index_name}'가 존재하지 않습니다.")
                return True
                
        except Exception as e:
            logger.error(f"인덱스 삭제 중 오류: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보를 반환합니다."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"error": "인덱스가 존재하지 않습니다."}
            
            stats = self.client.indices.stats(index=self.index_name)
            count_response = self.client.count(index=self.index_name)
            
            return {
                "document_count": count_response['count'],
                "index_size": stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                "status": "healthy" if self.client.cluster.health()['status'] in ['green', 'yellow'] else "unhealthy"
            }
            
        except Exception as e:
            logger.error(f"인덱스 통계 조회 중 오류: {str(e)}")
            return {"error": str(e)} 