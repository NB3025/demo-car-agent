#!/usr/bin/env python3
"""
Streamlit 기반 RAG 질의응답 웹 애플리케이션
"""

import streamlit as st
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# 로컬 모듈 임포트
from rag_system import RAGSystem
from log_manager import QALogManager
from pdf_page_viewer import PDFPageViewer, render_page_image_in_streamlit, create_pdf_viewer_link
from config import FULL_PDF_PATH
# ßßßfrom config import Config

# 로깅 설정
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="🚗 자동차 매뉴얼 AI 어시스턴트",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stChat {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    .stats-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRAGApp:
    """Streamlit RAG 앱 클래스"""
    
    def __init__(self):
        """초기화"""
        self.init_session_state()
        self.init_components()
    
    def init_session_state(self):
        """세션 상태 초기화"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        
        if 'log_manager' not in st.session_state:
            st.session_state.log_manager = QALogManager()
        
        if 'pdf_page_viewer' not in st.session_state:
            st.session_state.pdf_page_viewer = PDFPageViewer()
        
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
    
    def init_components(self):
        """컴포넌트 초기화"""
        try:
            if not st.session_state.system_initialized:
                with st.spinner("RAG 시스템을 초기화하고 있습니다..."):
                    # RAG 시스템 초기화
                    st.session_state.rag_system = RAGSystem()
                    st.session_state.system_initialized = True
                    st.success("✅ RAG 시스템 초기화 완료!")
        except Exception as e:
            st.error(f"❌ RAG 시스템 초기화 실패: {str(e)}")
            st.stop()
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.header("🛠️ 설정")
            
            # 세션 정보
            st.subheader("📱 세션 정보")
            st.write(f"**세션 ID:** `{st.session_state.session_id}`")
            st.write(f"**시작 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            # 새 세션 시작
            if st.button("🔄 새 세션 시작"):
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.session_state.messages = []
                st.rerun()
            
            st.divider()
            
            # 검색 설정
            st.subheader("🔍 검색 설정")
            
            # 검색 설정을 세션 상태에 저장
            if 'search_config' not in st.session_state:
                st.session_state.search_config = {
                    'max_results': 5,
                    'search_type': 'hybrid'
                }
            
            st.session_state.search_config['max_results'] = st.slider(
                "최대 검색 결과 수", 
                1, 10, 
                st.session_state.search_config['max_results'],
                help="더 많은 결과를 검색하면 더 다양한 정보를 얻을 수 있지만 응답 시간이 길어질 수 있습니다."
            )
            
            st.session_state.search_config['search_type'] = st.selectbox(
                "검색 방식",
                ["hybrid", "vector", "keyword"],
                index=["hybrid", "vector", "keyword"].index(st.session_state.search_config['search_type']),
                help="hybrid: 벡터+키워드 (권장), vector: 의미 검색, keyword: 키워드 검색"
            )
            
            # 현재 설정 표시
            st.info(f"🎯 현재 설정: {st.session_state.search_config['search_type']} 검색, 최대 {st.session_state.search_config['max_results']}개 결과")
            
            # 페이지 이미지 표시 설정
            st.session_state.show_page_images = st.checkbox(
                "📖 매뉴얼 페이지 이미지 표시",
                value=st.session_state.get('show_page_images', True),
                help="답변과 함께 관련 매뉴얼 페이지를 이미지로 표시합니다 (처리 시간이 다소 걸릴 수 있습니다)"
            )
            
            st.divider()
            
            # 통계 정보
            self.render_statistics()
            
            st.divider()
            
            # 최근 대화 이력
            self.render_recent_conversations()
            
            st.divider()
            
            # 페이지 뷰어 캐시 관리
            self.render_cache_management()
    
    def render_statistics(self):
        """통계 정보 렌더링"""
        st.subheader("📊 사용 통계")
        
        try:
            stats = st.session_state.log_manager.get_statistics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("총 대화", stats.get('total_sessions', 0))
                st.metric("피드백 수", stats.get('sessions_with_feedback', 0))
            
            with col2:
                avg_time = stats.get('average_response_time')
                if avg_time:
                    st.metric("평균 응답시간", f"{avg_time:.1f}초")
                
                avg_score = stats.get('average_confidence_score')
                if avg_score:
                    st.metric("평균 신뢰도", f"{avg_score:.2f}")
                    
        except Exception as e:
            st.error(f"통계 로딩 실패: {str(e)}")
    
    def render_recent_conversations(self):
        """최근 대화 이력 렌더링"""
        st.subheader("🕒 최근 대화")
        
        try:
            recent_sessions = st.session_state.log_manager.get_recent_sessions(limit=5)
            
            if recent_sessions:
                for session in recent_sessions:
                    with st.expander(f"💬 {session['user_question'][:30]}..."):
                        st.write(f"**시간:** {session['timestamp'][:19]}")
                        st.write(f"**질문:** {session['user_question']}")
                        st.write(f"**답변:** {session['system_answer'][:100]}...")
                        if session.get('confidence_score'):
                            st.write(f"**신뢰도:** {session['confidence_score']:.2f}")
            else:
                st.write("아직 대화 이력이 없습니다.")
                
        except Exception as e:
            st.error(f"대화 이력 로딩 실패: {str(e)}")
    
    def render_chat_interface(self):
        """채팅 인터페이스 렌더링"""
        st.header("🤖 자동차 매뉴얼 AI 어시스턴트")
        st.write("자동차 사용법, 주의사항, 문제해결 등에 대해 질문해보세요!")
        
        st.success("✨ **새로운 기능**: 답변과 함께 실제 매뉴얼 페이지를 이미지로 표시하고 PDF에서 직접 열 수 있습니다!")
        
        # 기능 소개
        with st.expander("🔍 사용 가능한 기능들", expanded=False):
            st.markdown("""
            - **🤖 AI 답변**: 질문에 대한 상세한 답변 생성
            - **📖 페이지 참조**: 관련 매뉴얼 페이지 번호 표시
            - **🖼️ 페이지 이미지**: 실제 매뉴얼 페이지를 이미지로 표시
            - **📂 PDF 뷰어**: 브라우저에서 해당 페이지 직접 열기
            - **💾 이미지 다운로드**: 매뉴얼 페이지 이미지 저장
            - **⚙️ 검색 설정**: 검색 방식과 결과 수 조정 가능
            - **📊 상세 소스**: 관련 텍스트 내용과 신뢰도 점수 표시
            """)
        
        # 이전 메시지들 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 빠른 페이지 참조, 페이지 이미지 및 소스 정보 표시 (어시스턴트 메시지인 경우)
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        self.render_quick_page_reference(message["sources"])
                        
                        # 페이지 이미지 표시 (설정이 활성화된 경우)
                        if st.session_state.get('show_page_images', True):
                            self.render_page_images(message["sources"])
                    
                    self.render_sources(message["sources"])
                
                # 피드백 버튼 (어시스턴트 메시지인 경우)
                if message["role"] == "assistant" and "log_id" in message:
                    self.render_feedback_buttons(message["log_id"])
        
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요..."):
            self.handle_user_input(prompt)
    
    def handle_user_input(self, user_input: str):
        """사용자 입력 처리"""
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성중입니다..."):
                response_data = self.get_ai_response(user_input)
                
                # 응답 표시
                st.markdown(response_data["answer"])
                
                # 빠른 페이지 참조 정보 표시
                if response_data.get("sources"):
                    self.render_quick_page_reference(response_data["sources"])
                
                # 페이지 이미지 표시 (설정이 활성화된 경우)
                if response_data.get("sources") and st.session_state.get('show_page_images', True):
                    self.render_page_images(response_data["sources"])
                
                # 소스 정보 표시
                if response_data.get("sources"):
                    self.render_sources(response_data["sources"])
                
                # 응답을 세션 상태에 저장
                assistant_message = {
                    "role": "assistant",
                    "content": response_data["answer"],
                    "sources": response_data.get("sources", []),
                    "log_id": response_data.get("log_id"),
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.messages.append(assistant_message)
                
                # 피드백 버튼 표시
                if response_data.get("log_id"):
                    self.render_feedback_buttons(response_data["log_id"])
    
    def get_ai_response(self, user_question: str) -> Dict[str, Any]:
        """AI 응답 생성"""
        start_time = time.time()
        
        try:
            # RAG 시스템으로 답변 생성
            response = st.session_state.rag_system.query(
                question=user_question,
                k=st.session_state.search_config['max_results'],
                search_type=st.session_state.search_config['search_type']
            )
            
            response_time = time.time() - start_time
            
            # 소스 정보 가져오기 (이미 sources에 포함됨)
            sources = []
            if response.get("sources"):
                for source in response["sources"]:
                    sources.append({
                        "content": source.get("content", ""),
                        "page": source.get("page_number", "N/A"),
                        "score": source.get("score", 0.0),
                        "section_type": source.get("section_type", "general"),
                        "has_images": source.get("has_images", False)
                    })
            
            # 로그 저장
            log_id = st.session_state.log_manager.log_qa_session(
                session_id=st.session_state.session_id,
                user_question=user_question,
                system_answer=response.get("answer", ""),
                search_results=response.get("sources", []),
                confidence_score=response.get("results_count", 0) / 5.0 if response.get("results_count") else None,
                response_time=response_time
            )
            
            return {
                "answer": response.get("answer", "죄송합니다. 답변을 생성할 수 없습니다."),
                "sources": sources,
                "log_id": log_id,
                "response_time": response_time
            }
            
        except Exception as e:
            error_message = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            
            # 오류 로그 저장
            log_id = st.session_state.log_manager.log_qa_session(
                session_id=st.session_state.session_id,
                user_question=user_question,
                system_answer="",
                error_message=str(e),
                response_time=time.time() - start_time
            )
            
            return {
                "answer": error_message,
                "sources": [],
                "log_id": log_id,
                "response_time": time.time() - start_time
            }
    
    def render_sources(self, sources: List[Dict]):
        """소스 정보 렌더링 - 차량용 매뉴얼 페이지 정보 강화"""
        if sources:
            with st.expander(f"📚 참고된 매뉴얼 페이지 ({len(sources)}개)", expanded=False):
                # 페이지별로 그룹화
                page_groups = {}
                for source in sources:
                    page = source.get('page', 'N/A')
                    if page not in page_groups:
                        page_groups[page] = []
                    page_groups[page].append(source)
                
                # 페이지별로 표시
                for page_num in sorted(page_groups.keys(), key=lambda x: int(x) if x != 'N/A' else 999):
                    page_sources = page_groups[page_num]
                    
                    # 페이지 헤더
                    st.markdown(f"### 📄 페이지 {page_num}")
                    
                    # 페이지 내 각 소스 표시
                    for i, source in enumerate(page_sources):
                        score = source.get('score', 0)
                        section_type = source.get('section_type', 'general')
                        has_images = source.get('has_images', False)
                        content = source.get('content', 'N/A')
                        
                        # 점수에 따른 색상 결정
                        if score >= 0.8:
                            border_color = "#4CAF50"  # 녹색 (높은 관련성)
                            relevance_text = "높은 관련성"
                        elif score >= 0.6:
                            border_color = "#FF9800"  # 주황색 (중간 관련성)
                            relevance_text = "중간 관련성"
                        else:
                            border_color = "#9E9E9E"  # 회색 (낮은 관련성)
                            relevance_text = "낮은 관련성"
                        
                        # 섹션 타입 한국어 변환
                        section_type_kr = {
                            'general': '일반',
                            'warning': '경고',
                            'caution': '주의',
                            'note': '참고',
                            'procedure': '절차',
                            'specification': '사양'
                        }.get(section_type, section_type)
                        
                        # 이미지 포함 아이콘
                        image_icon = "🖼️" if has_images else "📝"
                        
                        st.markdown(f"""
                        <div style="
                            border-left: 4px solid {border_color};
                            background-color: #f8f9fa;
                            padding: 15px;
                            margin: 10px 0;
                            border-radius: 5px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            <div style="
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                                margin-bottom: 10px;
                                font-size: 0.9em;
                                color: #666;
                            ">
                                <div>
                                    <span style="
                                        background-color: {border_color};
                                        color: white;
                                        padding: 2px 8px;
                                        border-radius: 12px;
                                        font-size: 0.8em;
                                        margin-right: 8px;
                                    ">{relevance_text}</span>
                                    <span style="
                                        background-color: #e3f2fd;
                                        color: #1976d2;
                                        padding: 2px 8px;
                                        border-radius: 12px;
                                        font-size: 0.8em;
                                        margin-right: 8px;
                                    ">{section_type_kr}</span>
                                    <span>{image_icon} {"이미지 포함" if has_images else "텍스트만"}</span>
                                </div>
                                <div style="font-weight: bold;">
                                    유사도: {score:.3f}
                                </div>
                            </div>
                            <div style="
                                background-color: white;
                                padding: 12px;
                                border-radius: 4px;
                                line-height: 1.5;
                                font-size: 0.95em;
                            ">
                                {content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 페이지 요약 정보
                total_pages = len(page_groups)
                unique_sections = set(source.get('section_type', 'general') for source in sources)
                images_count = sum(1 for source in sources if source.get('has_images', False))
                
                st.markdown("---")
                st.markdown("### 📊 매뉴얼 참조 요약")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("참조 페이지 수", total_pages)
                with col2:
                    st.metric("관련 섹션 수", len(unique_sections))
                with col3:
                    st.metric("이미지 포함 항목", images_count)
                with col4:
                    avg_score = sum(s.get('score', 0) for s in sources) / len(sources)
                    st.metric("평균 관련성", f"{avg_score:.3f}")
                
                if unique_sections:
                    st.markdown("**포함된 섹션 유형:** " + ", ".join(unique_sections))
    
    def render_feedback_buttons(self, log_id: str):
        """피드백 버튼 렌더링"""
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("👍 좋아요", key=f"like_{log_id}"):
                st.session_state.log_manager.update_user_feedback(log_id, "👍 좋아요")
                st.success("피드백이 저장되었습니다!")
        
        with col2:
            if st.button("👎 별로예요", key=f"dislike_{log_id}"):
                st.session_state.log_manager.update_user_feedback(log_id, "👎 별로예요")
                st.success("피드백이 저장되었습니다!")
    
    def render_quick_page_reference(self, sources: List[Dict]):
        """빠른 페이지 참조 정보 표시"""
        if sources:
            # 페이지별로 그룹화
            page_info = {}
            for source in sources:
                page = source.get('page', 'N/A')
                if page != 'N/A':
                    if page not in page_info:
                        page_info[page] = {
                            'has_images': False,
                            'sections': set(),
                            'max_score': 0
                        }
                    page_info[page]['has_images'] = page_info[page]['has_images'] or source.get('has_images', False)
                    page_info[page]['sections'].add(source.get('section_type', 'general'))
                    page_info[page]['max_score'] = max(page_info[page]['max_score'], source.get('score', 0))
            
            if page_info:
                st.markdown("### 📖 매뉴얼 페이지 참조")
                
                # 페이지를 점수 순으로 정렬
                sorted_pages = sorted(page_info.items(), key=lambda x: x[1]['max_score'], reverse=True)
                
                cols = st.columns(min(len(sorted_pages), 4))
                
                for i, (page, info) in enumerate(sorted_pages[:4]):  # 최대 4개 페이지만 표시
                    with cols[i % 4]:
                        # 관련성에 따른 색상
                        if info['max_score'] >= 0.8:
                            bg_color = "#e8f5e8"
                            border_color = "#4CAF50"
                        elif info['max_score'] >= 0.6:
                            bg_color = "#fff3e0"
                            border_color = "#FF9800"
                        else:
                            bg_color = "#f5f5f5"
                            border_color = "#9E9E9E"
                        
                        # 섹션 정보
                        sections_kr = []
                        for section in info['sections']:
                            section_kr = {
                                'general': '일반',
                                'warning': '경고',
                                'caution': '주의',
                                'note': '참고',
                                'procedure': '절차',
                                'specification': '사양'
                            }.get(section, section)
                            sections_kr.append(section_kr)
                        
                        image_indicator = "🖼️" if info['has_images'] else "📝"
                        
                        st.markdown(f"""
                        <div style="
                            background-color: {bg_color};
                            border: 2px solid {border_color};
                            border-radius: 8px;
                            padding: 12px;
                            margin: 5px 0;
                            text-align: center;
                        ">
                            <div style="font-size: 1.2em; font-weight: bold; color: #333;">
                                📄 페이지 {page}
                            </div>
                            <div style="font-size: 0.9em; color: #666; margin: 5px 0;">
                                {image_indicator} {", ".join(sections_kr)}
                            </div>
                            <div style="font-size: 0.8em; color: #888;">
                                관련성: {info['max_score']:.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
    
    def render_page_images(self, sources: List[Dict]):
        """관련 페이지들을 이미지로 표시"""
        if not sources:
            return
        
        # 페이지별로 그룹화
        page_groups = {}
        for source in sources:
            page = source.get('page', 'N/A')
            if page != 'N/A':
                if page not in page_groups:
                    page_groups[page] = []
                page_groups[page].append(source)
        
        if not page_groups:
            return
        
        # 사이드바에서 페이지 이미지 표시 설정
        show_images = True
        if 'show_page_images' not in st.session_state:
            st.session_state.show_page_images = True
        
        if not st.session_state.show_page_images:
            return
        
        st.markdown("### 📖 매뉴얼 페이지 이미지")
        
        # 페이지를 관련성 순으로 정렬
        sorted_pages = sorted(
            page_groups.items(), 
            key=lambda x: max(s.get('score', 0) for s in x[1]), 
            reverse=True
        )
        
        # 최대 3개 페이지까지만 표시 (성능 고려)
        max_pages_to_show = min(3, len(sorted_pages))
        
        if max_pages_to_show > 1:
            # 여러 페이지인 경우 탭으로 구분
            tab_names = [f"📄 페이지 {page}" for page, _ in sorted_pages[:max_pages_to_show]]
            tabs = st.tabs(tab_names)
            
            for i, (page_num, page_sources) in enumerate(sorted_pages[:max_pages_to_show]):
                with tabs[i]:
                    self._render_single_page_image(page_num, page_sources)
        else:
            # 단일 페이지인 경우 바로 표시
            page_num, page_sources = sorted_pages[0]
            self._render_single_page_image(page_num, page_sources)
    
    def _render_single_page_image(self, page_num: int, page_sources: List[Dict]):
        """단일 페이지 이미지와 정보를 렌더링"""
        try:
            # 페이지 정보 헤더
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"#### 📄 매뉴얼 페이지 {page_num}")
                
                # 페이지 소스 정보 요약
                max_score = max(s.get('score', 0) for s in page_sources)
                sections = set(s.get('section_type', 'general') for s in page_sources)
                has_images = any(s.get('has_images', False) for s in page_sources)
                
                section_names = []
                for section in sections:
                    section_kr = {
                        'general': '일반',
                        'warning': '경고',
                        'caution': '주의',
                        'note': '참고',
                        'procedure': '절차',
                        'specification': '사양'
                    }.get(section, section)
                    section_names.append(section_kr)
                
                st.markdown(f"""
                **관련성**: {max_score:.3f} | 
                **섹션**: {', '.join(section_names)} | 
                **이미지**: {'포함됨' if has_images else '없음'}
                """)
            
            with col2:
                # PDF 뷰어에서 열기 버튼
                pdf_link = create_pdf_viewer_link(FULL_PDF_PATH, int(page_num))
                st.markdown(f"""
                <a href="{pdf_link}" target="_blank">
                    <button style="
                        background-color: #1f77b4;
                        color: white;
                        padding: 8px 16px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                    ">📂 PDF에서 열기</button>
                </a>
                """, unsafe_allow_html=True)
            
            # 페이지 이미지 생성 및 표시
            with st.spinner(f"페이지 {page_num} 이미지를 생성하고 있습니다..."):
                image_path = st.session_state.pdf_page_viewer.extract_page_image(
                    FULL_PDF_PATH, 
                    int(page_num),
                    dpi=120  # 웹 표시용으로 적당한 해상도
                )
                
                if image_path:
                    # 이미지 표시
                    render_page_image_in_streamlit(
                        image_path, 
                        int(page_num),
                        width=600  # 적당한 크기로 제한
                    )
                    
                    # 이미지 다운로드 링크
                    with open(image_path, "rb") as file:
                        st.download_button(
                            label="💾 이미지 다운로드",
                            data=file.read(),
                            file_name=f"manual_page_{page_num}.png",
                            mime="image/png"
                        )
                else:
                    st.error(f"페이지 {page_num} 이미지를 생성할 수 없습니다.")
            
            # 관련 텍스트 내용 표시
            with st.expander(f"📝 페이지 {page_num} 텍스트 내용", expanded=False):
                for i, source in enumerate(page_sources, 1):
                    st.markdown(f"**구간 {i}** (점수: {source.get('score', 0):.3f})")
                    st.markdown(source.get('content', 'N/A'))
                    if i < len(page_sources):
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"페이지 {page_num} 표시 중 오류: {e}")
            logger.error(f"페이지 이미지 표시 실패 (페이지 {page_num}): {e}")
    
    def render_cache_management(self):
        """페이지 뷰어 캐시 관리"""
        st.subheader("🗂️ 페이지 캐시")
        
        try:
            # 캐시 통계 가져오기
            cache_stats = st.session_state.pdf_page_viewer.get_cache_stats()
            
            if cache_stats:
                st.markdown(f"""
                **캐시된 페이지**: {cache_stats.get('cached_pages', 0)}개  
                **이미지 파일**: {cache_stats.get('image_files', 0)}개  
                **사용 용량**: {cache_stats.get('total_size_mb', 0):.1f} MB
                """)
                
                # 캐시 정리 버튼
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ 캐시 정리", help="모든 페이지 이미지 캐시를 삭제합니다"):
                        with st.spinner("캐시를 정리하고 있습니다..."):
                            st.session_state.pdf_page_viewer.clear_cache()
                        st.success("캐시가 정리되었습니다!")
                        st.rerun()
                
                with col2:
                    if st.button("📊 캐시 새로고침", help="캐시 통계를 새로고침합니다"):
                        st.rerun()
            else:
                st.write("캐시 정보를 불러올 수 없습니다.")
                
        except Exception as e:
            st.error(f"캐시 관리 오류: {e}")
    
    def run(self):
        """앱 실행"""
        self.render_sidebar()
        self.render_chat_interface()

def main():
    """메인 함수"""
    try:
        app = StreamlitRAGApp()
        app.run()
        
    except Exception as e:
        st.error(f"❌ 애플리케이션 오류: {str(e)}")
        st.write("페이지를 새로고침해주세요.")

if __name__ == "__main__":
    main() 