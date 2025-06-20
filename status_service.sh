#!/bin/bash

# Streamlit RAG 서비스 상태 확인 스크립트

PID_FILE="streamlit.pid"
LOG_FILE="streamlit.log"

echo "📊 Streamlit RAG 서비스 상태 확인"
echo "=" * 40

# PID 파일 확인
if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 서비스 실행 중 (PID: $PID)"
        
        # 프로세스 정보
        echo ""
        echo "📋 프로세스 정보:"
        ps -p $PID -o pid,ppid,cmd,etime,pcpu,pmem
        
        # 네트워크 포트 확인
        echo ""
        echo "🌐 포트 사용 상황:"
        netstat -tlnp 2>/dev/null | grep :8501 || echo "   포트 8501이 열려있지 않습니다"
        
    else
        echo "❌ 서비스가 실행 중이 아닙니다 (PID 파일 존재하지만 프로세스 없음)"
        rm $PID_FILE
    fi
else
    echo "❌ 서비스가 실행 중이 아닙니다 (PID 파일 없음)"
fi

# 모든 관련 프로세스 확인
echo ""
echo "🔍 모든 관련 프로세스:"
PROCESSES=$(ps aux | grep "run_streamlit.py\|streamlit" | grep -v grep)
if [ -n "$PROCESSES" ]; then
    echo "$PROCESSES"
else
    echo "   관련 프로세스가 없습니다"
fi

# 로그 파일 확인
echo ""
echo "📋 로그 파일 정보:"
if [ -f "$LOG_FILE" ]; then
    echo "   파일 크기: $(du -h $LOG_FILE | cut -f1)"
    echo "   마지막 수정: $(stat -c %y $LOG_FILE)"
    echo ""
    echo "📄 최근 로그 (마지막 10줄):"
    tail -10 $LOG_FILE
else
    echo "   로그 파일이 없습니다"
fi 