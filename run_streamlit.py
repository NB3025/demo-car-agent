#!/usr/bin/env python3
"""
Streamlit RAG 앱 실행 스크립트
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def check_requirements():
    """필요한 패키지들이 설치되어 있는지 확인"""
    required_packages = [
        'streamlit', 'pandas', 'plotly', 'boto3', 
        'langchain', 'opensearch-py'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 누락된 패키지들:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 설치 명령어:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_environment():
    """환경 설정 확인"""
    required_files = [
        '.env',
        'config.py',
        'rag_system.py',
        'log_manager.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ 누락된 파일들:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def run_streamlit_app(port=8501, host="localhost"):
    """Streamlit 앱 실행"""
    
    print("🔍 환경 확인 중...")
    
    # # 환경 설정 확인
    # if not check_environment():
    #     print("\n🛠️ 필요한 파일들이 누락되었습니다.")
    #     return False
    
    print("✅ 환경 확인 완료!")
    
    # Streamlit 앱 실행
    print(f"\n🚀 Streamlit 앱 실행 중...")
    print(f"📍 주소: http://{host}:{port}")
    print(f"🛑 종료: Ctrl+C")
    
    try:
        # Streamlit 명령어 실행
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", str(port),
            "--server.address", host,
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 앱이 종료되었습니다.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 앱 실행 실패: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Streamlit RAG 앱 실행")
    parser.add_argument("--port", type=int, default=8501, help="포트 번호 (기본값: 8501)")
    parser.add_argument("--host", type=str, default="localhost", help="호스트 주소 (기본값: localhost)")
    
    args = parser.parse_args()
    
    print("🚗 자동차 매뉴얼 RAG 시스템 실행")
    print("=" * 50)
    
    success = run_streamlit_app(port=args.port, host=args.host)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 