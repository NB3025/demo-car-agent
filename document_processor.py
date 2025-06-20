import pymupdf4llm
import pymupdf as fitz
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import io
import base64
import tiktoken
from datetime import datetime
from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_TOKENS_PER_CHUNK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """문서 청크를 나타내는 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    page_number: int
    chunk_id: str
    section_type: Optional[str] = None
    has_images: bool = False
    image_descriptions: List[str] = None
    
    def __post_init__(self):
        if self.image_descriptions is None:
            self.image_descriptions = []

class DocumentProcessor:
    """PDF 문서를 처리하고 청킹하는 클래스"""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """PDF에서 텍스트와 이미지를 추출합니다."""
        logger.info(f"PDF 파일 처리 시작: {pdf_path}")
        
        try:
            # PyMuPDF4LLM을 사용하여 마크다운으로 변환
            md_text = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=True,
                write_images=True,
                image_path="./images/",
                image_format="png",
                dpi=150
            )
            
            # 원본 PDF에서 메타데이터 추출
            doc = fitz.open(pdf_path)
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'total_pages': doc.page_count
            }
            
            # 이미지 정보 추출
            images_info = self._extract_images_info(doc)
            
            doc.close()
            
            return {
                'markdown_content': md_text,
                'metadata': metadata,
                'images_info': images_info
            }
            
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {str(e)}")
            raise
    
    def _extract_images_info(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """PDF에서 이미지 정보를 추출합니다."""
        images_info = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        images_info.append({
                            'page_number': page_num + 1,
                            'image_index': img_index,
                            'xref': xref,
                            'width': pix.width,
                            'height': pix.height,
                            'data': base64.b64encode(img_data).decode()
                        })
                    
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"이미지 추출 실패 (페이지 {page_num + 1}, 이미지 {img_index}): {str(e)}")
                    continue
        
        return images_info
    
    def identify_section_type(self, text: str) -> str:
        """텍스트 내용을 기반으로 섹션 타입을 식별합니다."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['주의', '경고', '위험', '조심']):
            return 'warning'
        elif any(keyword in text_lower for keyword in ['사용법', '조작', '작동', '방법']):
            return 'instruction'
        elif any(keyword in text_lower for keyword in ['사양', '규격', '제원']):
            return 'specification'
        elif any(keyword in text_lower for keyword in ['문제해결', '고장', '점검', '진단']):
            return 'troubleshooting'
        else:
            return 'general'
    
    def create_chunks(self, content: Dict[str, Any]) -> List[DocumentChunk]:
        """문서 내용을 청크로 분할합니다."""
        logger.info("문서 청킹 시작")
        
        markdown_content = content['markdown_content']
        metadata = content['metadata']
        images_info = content['images_info']
        
        # 페이지별로 분할
        if isinstance(markdown_content, list):
            # page_chunks=True인 경우 리스트로 반환됨
            page_contents = markdown_content
        else:
            # 단일 문자열인 경우 페이지 구분자로 분할
            page_contents = markdown_content.split('\n---\n')
        
        chunks = []
        chunk_id = 0
        
        for page_num, page_content in enumerate(page_contents):
            # PyMuPDF4LLM이 dict 형태로 반환할 수 있음
            if isinstance(page_content, dict):
                # dict인 경우 'text' 키에서 텍스트 추출
                text_content = page_content.get('text', '') or page_content.get('content', '') or str(page_content)
            else:
                text_content = str(page_content)
            
            if not text_content.strip():
                continue
                
            # 페이지 내 이미지 정보 찾기
            page_images = [img for img in images_info if img['page_number'] == page_num + 1]
            
            # 섹션별로 분할 (헤딩 기준)
            sections = self._split_by_sections(text_content)
            
            for section in sections:
                # 토큰 수 확인
                token_count = len(self.tokenizer.encode(section))
                
                if token_count <= MAX_TOKENS_PER_CHUNK:
                    # 단일 청크로 처리
                    chunk = DocumentChunk(
                        content=section,
                        metadata={
                            **metadata,
                            'source_file': content.get('source_file', ''),
                            'processing_timestamp': content.get('timestamp', '')
                        },
                        page_number=page_num + 1,
                        chunk_id=f"chunk_{chunk_id}",
                        section_type=self.identify_section_type(section),
                        has_images=len(page_images) > 0,
                        image_descriptions=[f"페이지 {page_num + 1}에 {len(page_images)}개의 이미지 포함"] if page_images else []
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                else:
                    # 긴 섹션을 여러 청크로 분할
                    sub_chunks = self._split_long_section(section, page_num + 1, page_images)
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_id = f"chunk_{chunk_id}"
                        sub_chunk.metadata = {
                            **metadata,
                            'source_file': content.get('source_file', ''),
                            'processing_timestamp': content.get('timestamp', '')
                        }
                        chunks.append(sub_chunk)
                        chunk_id += 1
        
        logger.info(f"총 {len(chunks)}개의 청크 생성 완료")
        return chunks
    
    def _split_by_sections(self, content: str) -> List[str]:
        """마크다운 헤딩을 기준으로 섹션을 분할합니다."""
        # 헤딩 패턴 (# ## ### 등)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        sections = []
        current_section = []
        
        for line in lines:
            if re.match(heading_pattern, line):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = [line]
                else:
                    current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return [section for section in sections if section.strip()]
    
    def _split_long_section(self, section: str, page_number: int, page_images: List[Dict]) -> List[DocumentChunk]:
        """긴 섹션을 여러 청크로 분할합니다."""
        sentences = re.split(r'[.!?]\s+', section)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + sentence + ". "
            
            if len(self.tokenizer.encode(test_chunk)) <= MAX_TOKENS_PER_CHUNK:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        metadata={},
                        page_number=page_number,
                        chunk_id="",  # 나중에 설정
                        section_type=self.identify_section_type(current_chunk),
                        has_images=len(page_images) > 0,
                        image_descriptions=[f"페이지 {page_number}에 {len(page_images)}개의 이미지 포함"] if page_images else []
                    )
                    chunks.append(chunk)
                
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata={},
                page_number=page_number,
                chunk_id="",  # 나중에 설정
                section_type=self.identify_section_type(current_chunk),
                has_images=len(page_images) > 0,
                image_descriptions=[f"페이지 {page_number}에 {len(page_images)}개의 이미지 포함"] if page_images else []
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_document(self, pdf_path: str) -> List[DocumentChunk]:
        """전체 문서 처리 파이프라인을 실행합니다."""
        logger.info(f"문서 처리 시작: {pdf_path}")
        
        # PDF 내용 추출
        content = self.extract_pdf_content(pdf_path)
        content['source_file'] = pdf_path
        content['timestamp'] = datetime.now().isoformat()
        
        # 청킹
        chunks = self.create_chunks(content)
        
        logger.info(f"문서 처리 완료: {len(chunks)}개 청크 생성")
        return chunks 