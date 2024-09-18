#!/bin/bash

# 실행 중인 스크립트의 PID를 저장할 변수 초기화
PIDS=()

# Python 스크립트 실행 및 PID 저장
python3 ./app.py &
PIDS+=($!)

python3 ./main.py &
PIDS+=($!)

echo "Both scripts have been executed."

# Ctrl+C 시그널 처리 설정
trap 'kill_scripts' INT

# Ctrl+C 시 호출될 함수 정의
kill_scripts() {
    echo "Caught CTRL+C. Terminating scripts..."
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null
    done
    exit 1
}

# 스크립트가 종료되지 않도록 대기
wait
