#!/bin/bash

# Streamlit RAG 서비스 중지 스크립트

PID_FILE="streamlit.pid"

echo "🛑 Streamlit RAG 서비스 중지 중..."

if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo "✅ 서비스가 중지되었습니다 (PID: $PID)"
        
        # PID 파일 삭제
        rm $PID_FILE
        
        # 프로세스가 완전히 종료될 때까지 대기
        sleep 2
        
        if ps -p $PID > /dev/null 2>&1; then
            echo "⚠️  프로세스가 여전히 실행 중입니다. 강제 종료합니다..."
            kill -9 $PID
        fi
    else
        echo "❌ 서비스가 실행 중이 아닙니다"
        rm $PID_FILE
    fi
else
    echo "❌ PID 파일을 찾을 수 없습니다"
    echo "   수동으로 프로세스를 확인하세요: ps aux | grep streamlit"
fi

# 모든 관련 프로세스 확인
REMAINING=$(ps aux | grep "run_streamlit.py" | grep -v grep | wc -l)
if [ $REMAINING -gt 0 ]; then
    echo "⚠️  아직 실행 중인 관련 프로세스가 있습니다:"
    ps aux | grep "run_streamlit.py" | grep -v grep
fi 