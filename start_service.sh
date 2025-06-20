#!/bin/bash

# Streamlit RAG 서비스 시작 스크립트

SERVICE_NAME="streamlit_rag"
LOG_FILE="streamlit.log"
PID_FILE="streamlit.pid"

echo "🚀 Streamlit RAG 서비스 시작 중..."

# 기존 프로세스 확인
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat $PID_FILE)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "❌ 서비스가 이미 실행 중입니다 (PID: $OLD_PID)"
        echo "   먼저 ./stop_service.sh 로 중지하세요."
        exit 1
    fi
fi

# nohup으로 백그라운드 실행
nohup python run_streamlit.py --host 0.0.0.0 > $LOG_FILE 2>&1 &
PID=$!

# PID 저장
echo $PID > $PID_FILE

echo "✅ 서비스가 시작되었습니다!"
echo "📍 PID: $PID"
echo "📋 로그 파일: $LOG_FILE"
echo "🌐 접속 주소: http://0.0.0.0:8501"
echo ""
echo "📊 실시간 로그 보기: tail -f $LOG_FILE"
echo "🛑 서비스 중지: ./stop_service.sh" 