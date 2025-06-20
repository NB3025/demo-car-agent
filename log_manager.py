#!/usr/bin/env python3
"""
RAG 시스템의 질문/답변 로그 관리
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QALogManager:
    """질문/답변 로그 관리 클래스"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Args:
            log_dir: 로그 파일들을 저장할 디렉토리
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로그 파일 경로들
        self.json_log_file = self.log_dir / "qa_sessions.jsonl"
        self.csv_log_file = self.log_dir / "qa_sessions.csv"
        self.daily_log_dir = self.log_dir / "daily"
        self.daily_log_dir.mkdir(exist_ok=True)
        
        # CSV 헤더 초기화
        self._initialize_csv()
    
    def _initialize_csv(self):
        """CSV 파일 헤더 초기화"""
        if not self.csv_log_file.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'session_id', 'user_question', 'system_answer',
                'search_results_count', 'confidence_score', 'response_time',
                'user_feedback', 'error_message'
            ])
            df.to_csv(self.csv_log_file, index=False, encoding='utf-8')
    
    def log_qa_session(self, 
                      session_id: str,
                      user_question: str, 
                      system_answer: str,
                      search_results: List[Dict] = None,
                      confidence_score: float = None,
                      response_time: float = None,
                      user_feedback: str = None,
                      error_message: str = None) -> str:
        """
        질문/답변 세션을 로그에 저장
        
        Args:
            session_id: 세션 고유 ID
            user_question: 사용자 질문
            system_answer: 시스템 답변
            search_results: 검색 결과 리스트
            confidence_score: 답변 신뢰도 점수
            response_time: 응답 시간 (초)
            user_feedback: 사용자 피드백
            error_message: 오류 메시지 (있는 경우)
            
        Returns:
            저장된 로그의 고유 ID
        """
        timestamp = datetime.now()
        log_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{session_id}"
        
        # JSON 로그 데이터 구성
        log_data = {
            'log_id': log_id,
            'timestamp': timestamp.isoformat(),
            'session_id': session_id,
            'user_question': user_question,
            'system_answer': system_answer,
            'search_results': search_results or [],
            'search_results_count': len(search_results) if search_results else 0,
            'confidence_score': confidence_score,
            'response_time': response_time,
            'user_feedback': user_feedback,
            'error_message': error_message
        }
        
        try:
            # JSONL 파일에 저장 (각 라인이 하나의 JSON)
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
            
            # CSV 파일에도 저장
            self._append_to_csv(log_data)
            
            # 일별 로그 파일에도 저장
            self._save_daily_log(timestamp, log_data)
            
            logger.info(f"QA 세션 로그 저장 완료: {log_id}")
            return log_id
            
        except Exception as e:
            logger.error(f"로그 저장 실패: {str(e)}")
            raise
    
    def _append_to_csv(self, log_data: Dict):
        """CSV 파일에 로그 데이터 추가"""
        try:
            # 기존 데이터 읽기
            df = pd.read_csv(self.csv_log_file, encoding='utf-8')
            
            # 새 행 추가
            new_row = {
                'timestamp': log_data['timestamp'],
                'session_id': log_data['session_id'],
                'user_question': log_data['user_question'],
                'system_answer': log_data['system_answer'],
                'search_results_count': log_data['search_results_count'],
                'confidence_score': log_data['confidence_score'],
                'response_time': log_data['response_time'],
                'user_feedback': log_data['user_feedback'],
                'error_message': log_data['error_message']
            }
            
            # DataFrame에 추가
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # CSV 파일 저장
            df.to_csv(self.csv_log_file, index=False, encoding='utf-8')
            
        except Exception as e:
            logger.warning(f"CSV 저장 실패: {str(e)}")
    
    def _save_daily_log(self, timestamp: datetime, log_data: Dict):
        """일별 로그 파일에 저장"""
        try:
            daily_file = self.daily_log_dir / f"qa_{timestamp.strftime('%Y%m%d')}.jsonl"
            
            with open(daily_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.warning(f"일별 로그 저장 실패: {str(e)}")
    
    def get_recent_sessions(self, limit: int = 100) -> List[Dict]:
        """최근 QA 세션들을 조회"""
        try:
            sessions = []
            
            if self.json_log_file.exists():
                with open(self.json_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sessions.append(json.loads(line))
            
            # 최신 순으로 정렬하여 반환
            sessions.sort(key=lambda x: x['timestamp'], reverse=True)
            return sessions[:limit]
            
        except Exception as e:
            logger.error(f"세션 조회 실패: {str(e)}")
            return []
    
    def get_sessions_by_date(self, date: str) -> List[Dict]:
        """특정 날짜의 세션들을 조회 (YYYY-MM-DD 형식)"""
        try:
            daily_file = self.daily_log_dir / f"qa_{date.replace('-', '')}.jsonl"
            
            if not daily_file.exists():
                return []
            
            sessions = []
            with open(daily_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sessions.append(json.loads(line))
            
            return sessions
            
        except Exception as e:
            logger.error(f"날짜별 세션 조회 실패: {str(e)}")
            return []
    
    def update_user_feedback(self, log_id: str, feedback: str):
        """사용자 피드백 업데이트"""
        try:
            # CSV 파일에서 해당 로그 찾아서 업데이트
            df = pd.read_csv(self.csv_log_file, encoding='utf-8')
            
            # log_id는 timestamp_sessionid 형식이므로 timestamp 부분 추출
            target_timestamp = log_id.split('_')[0] + '_' + log_id.split('_')[1]
            
            # 해당하는 행 찾기 (timestamp 기준)
            mask = df['timestamp'].str.contains(target_timestamp, na=False)
            
            if mask.any():
                df.loc[mask, 'user_feedback'] = feedback
                df.to_csv(self.csv_log_file, index=False, encoding='utf-8')
                logger.info(f"피드백 업데이트 완료: {log_id}")
            else:
                logger.warning(f"해당 로그를 찾을 수 없음: {log_id}")
                
        except Exception as e:
            logger.error(f"피드백 업데이트 실패: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """로그 통계 정보 반환"""
        try:
            if not self.csv_log_file.exists():
                return {"total_sessions": 0}
            
            df = pd.read_csv(self.csv_log_file, encoding='utf-8')
            
            stats = {
                "total_sessions": len(df),
                "average_response_time": df['response_time'].mean() if 'response_time' in df.columns else None,
                "average_confidence_score": df['confidence_score'].mean() if 'confidence_score' in df.columns else None,
                "sessions_with_feedback": df['user_feedback'].notna().sum() if 'user_feedback' in df.columns else 0,
                "sessions_with_errors": df['error_message'].notna().sum() if 'error_message' in df.columns else 0,
                "most_recent_session": df['timestamp'].max() if len(df) > 0 else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 조회 실패: {str(e)}")
            return {"error": str(e)}

def main():
    """테스트용 메인 함수"""
    log_manager = QALogManager()
    
    # 테스트 로그 저장
    log_id = log_manager.log_qa_session(
        session_id="test_session_001",
        user_question="글로브 박스는 어떻게 여나요?",
        system_answer="레버(1)을 당기면 글로브 박스가 열립니다.",
        search_results=[{"content": "레버 관련 내용", "score": 0.95}],
        confidence_score=0.95,
        response_time=1.2
    )
    
    print(f"로그 저장 완료: {log_id}")
    
    # 통계 조회
    stats = log_manager.get_statistics()
    print(f"통계: {stats}")

if __name__ == "__main__":
    main() 