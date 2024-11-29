#!/bin/bash

# FILE_PATH
LOG_FILE="top_log.txt"
SLEEP_TIME_S=5
KEY_WORD="mainboard"

if [ "$#" -ne 3 ]; then
    echo "用法: $0 <log_file> <sleep_time_s> <key_word>"
    exit 1
fi

LOG_FILE="$1"
SLEEP_TIME_S="$2"
KEY_WORD="$3"

LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

stop_monitor() {
    echo "Stopping the monitor..."
    exit 0
}

trap stop_monitor SIGINT

# monitor
while true; do
    # add timestamp
    echo "Timestamp: $(date)" >> "$LOG_FILE"
    # Log top command output
    top -b -c -n 1 | head -n 5 >> "$LOG_FILE"
    # Log specific process information
    top -b -c -n 1 | grep "$KEY_WORD" >> "$LOG_FILE"

    sleep $SLEEP_TIME_S
done