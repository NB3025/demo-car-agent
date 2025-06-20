#!/usr/bin/env python3
"""
PDF 페이지 이미지 변환 및 뷰어 모듈
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

import fitz  # PyMuPDF
from PIL import Image
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFPageViewer:
    """PDF 페이지를 이미지로 변환하고 관리하는 클래스"""
    
    def __init__(self, images_dir: str = "images/pages", cache_info_file: str = "images/page_cache.json"):
        """
        초기화
        
        Args:
            images_dir: 페이지 이미지를 저장할 디렉토리
            cache_info_file: 캐시 정보를 저장할 JSON 파일
        """
        self.images_dir = Path(images_dir)
        self.cache_info_file = Path(cache_info_file)
        
        # 디렉토리 생성
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.cache_info_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 캐시 정보 로드
        self.cache_info = self._load_cache_info()
        
        logger.info(f"PDF 페이지 뷰어 초기화 완료: {self.images_dir}")
    
    def _load_cache_info(self) -> Dict:
        """캐시 정보 로드"""
        try:
            if self.cache_info_file.exists():
                with open(self.cache_info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"캐시 정보 로드 실패: {e}")
            return {}
    
    def _save_cache_info(self):
        """캐시 정보 저장"""
        try:
            with open(self.cache_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"캐시 정보 저장 실패: {e}")
    
    def _get_pdf_hash(self, pdf_path: str) -> str:
        """PDF 파일의 해시값 계산"""
        try:
            with open(pdf_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()[:8]
        except Exception as e:
            logger.error(f"PDF 해시 계산 실패: {e}")
            return "unknown"
    
    def _get_page_image_path(self, pdf_hash: str, page_num: int) -> Path:
        """페이지 이미지 파일 경로 생성"""
        filename = f"{pdf_hash}_page_{page_num:03d}.png"
        return self.images_dir / filename
    
    def extract_page_image(self, pdf_path: str, page_num: int, dpi: int = 150) -> Optional[str]:
        """
        PDF의 특정 페이지를 이미지로 변환하여 저장
        
        Args:
            pdf_path: PDF 파일 경로
            page_num: 페이지 번호 (1부터 시작)
            dpi: 이미지 해상도
            
        Returns:
            저장된 이미지 파일 경로 (성공시) 또는 None (실패시)
        """
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF 파일이 존재하지 않습니다: {pdf_path}")
                return None
            
            # PDF 해시 계산
            pdf_hash = self._get_pdf_hash(pdf_path)
            
            # 캐시 확인
            cache_key = f"{pdf_hash}_{page_num}"
            if cache_key in self.cache_info:
                image_path = Path(self.cache_info[cache_key]["image_path"])
                if image_path.exists():
                    logger.debug(f"캐시된 이미지 사용: {image_path}")
                    return str(image_path)
            
            # 이미지 파일 경로
            image_path = self._get_page_image_path(pdf_hash, page_num)
            
            # PDF 열기
            doc = fitz.open(pdf_path)
            
            if page_num < 1 or page_num > len(doc):
                logger.error(f"잘못된 페이지 번호: {page_num} (총 {len(doc)}페이지)")
                doc.close()
                return None
            
            # 페이지 선택 (0부터 시작하므로 -1)
            page = doc[page_num - 1]
            
            # 페이지를 이미지로 변환
            mat = fitz.Matrix(dpi/72, dpi/72)  # 해상도 설정
            pix = page.get_pixmap(matrix=mat)
            
            # PNG로 저장
            pix.save(str(image_path))
            
            doc.close()
            
            # 캐시 정보 업데이트
            self.cache_info[cache_key] = {
                "pdf_path": pdf_path,
                "pdf_hash": pdf_hash,
                "page_num": page_num,
                "image_path": str(image_path),
                "dpi": dpi,
                "file_size": os.path.getsize(image_path)
            }
            self._save_cache_info()
            
            logger.info(f"페이지 이미지 생성 완료: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"페이지 이미지 추출 실패 (페이지 {page_num}): {e}")
            return None
    
    def extract_multiple_pages(self, pdf_path: str, page_numbers: List[int], dpi: int = 150) -> Dict[int, str]:
        """
        여러 페이지를 한 번에 이미지로 변환
        
        Args:
            pdf_path: PDF 파일 경로
            page_numbers: 페이지 번호 리스트
            dpi: 이미지 해상도
            
        Returns:
            {페이지번호: 이미지경로} 딕셔너리
        """
        results = {}
        
        try:
            doc = fitz.open(pdf_path)
            pdf_hash = self._get_pdf_hash(pdf_path)
            mat = fitz.Matrix(dpi/72, dpi/72)
            
            for page_num in page_numbers:
                try:
                    # 캐시 확인
                    cache_key = f"{pdf_hash}_{page_num}"
                    if cache_key in self.cache_info:
                        image_path = Path(self.cache_info[cache_key]["image_path"])
                        if image_path.exists():
                            results[page_num] = str(image_path)
                            continue
                    
                    # 페이지 유효성 확인
                    if page_num < 1 or page_num > len(doc):
                        logger.warning(f"잘못된 페이지 번호: {page_num}")
                        continue
                    
                    # 이미지 생성
                    page = doc[page_num - 1]
                    pix = page.get_pixmap(matrix=mat)
                    image_path = self._get_page_image_path(pdf_hash, page_num)
                    pix.save(str(image_path))
                    
                    # 결과 저장
                    results[page_num] = str(image_path)
                    
                    # 캐시 업데이트
                    self.cache_info[cache_key] = {
                        "pdf_path": pdf_path,
                        "pdf_hash": pdf_hash,
                        "page_num": page_num,
                        "image_path": str(image_path),
                        "dpi": dpi,
                        "file_size": os.path.getsize(image_path)
                    }
                    
                except Exception as e:
                    logger.error(f"페이지 {page_num} 처리 실패: {e}")
                    continue
            
            doc.close()
            self._save_cache_info()
            
            logger.info(f"{len(results)}개 페이지 이미지 생성 완료")
            return results
            
        except Exception as e:
            logger.error(f"다중 페이지 추출 실패: {e}")
            return {}
    
    def get_page_info(self, pdf_path: str) -> Dict:
        """PDF 페이지 정보 가져오기"""
        try:
            doc = fitz.open(pdf_path)
            info = {
                "total_pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "file_size": os.path.getsize(pdf_path)
            }
            doc.close()
            return info
        except Exception as e:
            logger.error(f"PDF 정보 가져오기 실패: {e}")
            return {}
    
    def clear_cache(self, pdf_path: Optional[str] = None):
        """캐시 정리"""
        try:
            if pdf_path:
                # 특정 PDF의 캐시만 정리
                pdf_hash = self._get_pdf_hash(pdf_path)
                keys_to_remove = [k for k in self.cache_info.keys() if k.startswith(pdf_hash)]
                
                for key in keys_to_remove:
                    image_path = Path(self.cache_info[key]["image_path"])
                    if image_path.exists():
                        image_path.unlink()
                    del self.cache_info[key]
                    
                logger.info(f"{pdf_path}의 캐시 {len(keys_to_remove)}개 정리")
                
            else:
                # 전체 캐시 정리
                for image_file in self.images_dir.glob("*.png"):
                    image_file.unlink()
                self.cache_info.clear()
                logger.info("전체 캐시 정리 완료")
            
            self._save_cache_info()
            
        except Exception as e:
            logger.error(f"캐시 정리 실패: {e}")
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 정보"""
        try:
            total_files = len(list(self.images_dir.glob("*.png")))
            total_size = sum(f.stat().st_size for f in self.images_dir.glob("*.png"))
            
            return {
                "cached_pages": len(self.cache_info),
                "image_files": total_files,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_directory": str(self.images_dir)
            }
        except Exception as e:
            logger.error(f"캐시 통계 계산 실패: {e}")
            return {}


def render_page_image_in_streamlit(image_path: str, page_num: int, caption: str = "", width: Optional[int] = None):
    """
    Streamlit에서 페이지 이미지 렌더링
    
    Args:
        image_path: 이미지 파일 경로
        page_num: 페이지 번호
        caption: 이미지 캡션
        width: 이미지 너비 (픽셀)
    """
    try:
        if os.path.exists(image_path):
            st.image(
                image_path,
                caption=caption or f"📄 매뉴얼 페이지 {page_num}",
                width=width,
                use_container_width=width is None
            )
        else:
            st.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    except Exception as e:
        st.error(f"이미지 표시 실패: {e}")


def create_pdf_viewer_link(pdf_path: str, page_num: int) -> str:
    """
    PDF 뷰어 링크 생성 (브라우저에서 PDF 열기)
    
    Args:
        pdf_path: PDF 파일 경로
        page_num: 페이지 번호
        
    Returns:
        PDF 뷰어 링크
    """
    # 웹 브라우저에서 PDF를 열 수 있는 링크 생성
    # 로컬 파일의 경우 file:// 프로토콜 사용
    if os.path.isabs(pdf_path):
        file_url = f"file://{pdf_path}#page={page_num}"
    else:
        # 상대 경로를 절대 경로로 변환
        abs_path = os.path.abspath(pdf_path)
        file_url = f"file://{abs_path}#page={page_num}"
    
    return file_url 