#!/usr/bin/env python3
"""
AWS 기반 RAG 시스템 메인 실행 스크립트
"""

import os
import json
import logging
from typing import Dict, Any
import argparse

from rag_system import RAGSystem
from config import TEST_PDF_PATH, FULL_PDF_PATH, OPENSEARCH_ENDPOINT, OPENSEARCH_PASSWORD


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """환경 설정 확인"""
    # OpenSearch 설정 확인 (config.py의 기본값 사용)
    opensearch_endpoint = os.getenv('OPENSEARCH_ENDPOINT', OPENSEARCH_ENDPOINT)
    opensearch_password = os.getenv('OPENSEARCH_PASSWORD', OPENSEARCH_PASSWORD)
    
    if not opensearch_endpoint or not opensearch_password:
        logger.error("OpenSearch 연결 정보가 설정되지 않았습니다.")
        logger.info("config.py 파일의 기본값이 사용됩니다.")
        return False
    
    logger.info(f"OpenSearch 엔드포인트: {opensearch_endpoint}")
    
    # AWS 프로필 확인
    aws_profile = os.getenv('AWS_PROFILE', 'workagent')
    logger.info(f"AWS 프로필 사용: {aws_profile}")
    
    # AWS 자격 증명 확인 (환경변수가 설정되어 있으면 프로필보다 우선)
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
        logger.info("AWS 환경변수 자격 증명 사용")
    else:
        logger.info(f"AWS 프로필 '{aws_profile}' 사용")
    
    return True

def test_system_setup(rag_system: RAGSystem) -> bool:
    """시스템 설정 테스트"""
    logger.info("=== 시스템 설정 테스트 ===")
    
    # 시스템 설정
    if not rag_system.setup_system():
        logger.error("시스템 설정 실패")
        return False
    
    # 시스템 상태 확인
    status = rag_system.get_system_status()
    logger.info(f"시스템 상태: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    if status.get('bedrock_status') != 'connected':
        logger.error("Bedrock 연결 실패")
        return False
    
    if 'error' in status.get('opensearch_status', {}):
        logger.error(f"OpenSearch 연결 실패: {status['opensearch_status']['error']}")
        return False
    
    logger.info("시스템 설정 테스트 완료")
    return True

def test_document_processing(rag_system: RAGSystem, pdf_path: str) -> bool:
    """문서 처리 테스트"""
    logger.info(f"=== 문서 처리 테스트: {pdf_path} ===")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF 파일이 존재하지 않습니다: {pdf_path}")
        return False
    
    # 문서 처리 및 인덱싱
    success = rag_system.process_and_index_document(pdf_path)
    
    if success:
        logger.info("문서 처리 및 인덱싱 완료")
        
        # 인덱스 통계 확인
        status = rag_system.get_system_status()
        opensearch_stats = status.get('opensearch_status', {})
        
        if 'document_count' in opensearch_stats:
            logger.info(f"인덱싱된 문서 수: {opensearch_stats['document_count']}")
            logger.info(f"인덱스 크기: {opensearch_stats['index_size']} bytes")
        
        return True
    else:
        logger.error("문서 처리 및 인덱싱 실패")
        return False

def test_search_and_qa(rag_system: RAGSystem) -> bool:
    """검색 및 질의응답 테스트"""
    logger.info("=== 검색 및 질의응답 테스트 ===")
    
    # 테스트 질문들
    test_questions = [
        "글로브 박스는 어떻게 열어요?",
        "주의사항이 무엇인가요?",
        "동승석 멀티 트레이는 어떻게 사용하나요?",
        "안전벨트 착용 방법을 알려주세요",
        "차량 점검은 어떻게 하나요?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n--- 테스트 질문 {i}: {question} ---")
        
        # 질의응답 수행
        result = rag_system.query(question, k=3, search_type="hybrid")
        
        # 결과 출력
        logger.info(f"답변: {result['answer']}")
        logger.info(f"검색 시간: {result['search_time']:.2f}초")
        logger.info(f"검색된 소스 수: {result['results_count']}")
        
        # 소스 정보 출력
        if result['sources']:
            logger.info("관련 소스:")
            for j, source in enumerate(result['sources'][:2], 1):
                logger.info(f"  {j}. 페이지 {source['page_number']} (점수: {source['score']:.3f})")
                logger.info(f"     내용: {source['content'][:100]}...")
        
        print("-" * 80)
    
    return True

def interactive_mode(rag_system: RAGSystem):
    """대화형 모드"""
    logger.info("=== 대화형 질의응답 모드 ===")
    logger.info("질문을 입력하세요 (종료: 'quit' 또는 'exit')")
    
    while True:
        try:
            question = input("\n질문: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                logger.info("대화형 모드를 종료합니다.")
                break
            
            if not question:
                continue
            
            # 질의응답 수행
            result = rag_system.query(question, k=5, search_type="hybrid")
            
            # 결과 출력
            print(f"\n답변: {result['answer']}")
            print(f"검색 시간: {result['search_time']:.2f}초")
            
            if result['sources']:
                print("\n관련 소스:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. 페이지 {source['page_number']} (점수: {source['score']:.3f})")
                    print(f"     {source['content'][:150]}...")
            
        except KeyboardInterrupt:
            logger.info("\n대화형 모드를 종료합니다.")
            break
        except Exception as e:
            logger.error(f"오류 발생: {str(e)}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='AWS 기반 RAG 시스템')
    parser.add_argument('--mode', choices=['setup', 'index', 'test', 'interactive', 'reset'], 
                       default='test', help='실행 모드')
    parser.add_argument('--pdf', choices=['test', 'full'], default='test', 
                       help='사용할 PDF 파일')
    parser.add_argument('--reset', action='store_true', help='시스템 초기화')
    
    args = parser.parse_args()
    
    # 환경 설정 확인
    if not setup_environment():
        return 1
    
    try:
        # RAG 시스템 초기화
        logger.info("RAG 시스템 초기화 중...")
        rag_system = RAGSystem()
        
        # PDF 파일 경로 설정
        # pdf_path = TEST_PDF_PATH if args.pdf == 'test' else FULL_PDF_PATH
        pdf_path = './santafe.pdf'
        
        if args.mode == 'setup':
            # 시스템 설정만 수행
            success = test_system_setup(rag_system)
            return 0 if success else 1
            
        elif args.mode == 'index':
            # 문서 인덱싱만 수행
            if not test_system_setup(rag_system):
                return 1
            success = test_document_processing(rag_system, pdf_path)
            return 0 if success else 1
            
        elif args.mode == 'test':
            # 전체 테스트 수행
            if not test_system_setup(rag_system):
                return 1
            if not test_document_processing(rag_system, pdf_path):
                return 1
            test_search_and_qa(rag_system)
            return 0
            
        elif args.mode == 'interactive':
            # 대화형 모드
            if not test_system_setup(rag_system):
                return 1
            interactive_mode(rag_system)
            return 0
            
        elif args.mode == 'reset':
            # 시스템 초기화
            success = rag_system.reset_system()
            logger.info("시스템 초기화 완료" if success else "시스템 초기화 실패")
            return 0 if success else 1
            
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 