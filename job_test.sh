#!/bin/bash

# Docker
CONTAINER_NAME=("OmniVidi_VL" "redis-sc" "nginx-web-sc" "mysql-sc" "city-admin")
docker_staus="Normal"
dockers="[]"
DOCKER_REPLY="{\"status\": \"$docker_staus\", \"containers\": $dockers}"

# SSD
MOUNT_POINT="/mnt/ssd"
ssd_status="Normal"
dirs="[]"
SSD_REPLY="{\"status\": \"$ssd_status\", \"dirs\": $dirs}"

PORT=8080
NETIP="172.16.115.46"
URL="/sys/api"

resopnse=""

# 获取Docker容器状态
get_container_status() {
    docker_staus="Normal"
    dockers="["
    for container in "${CONTAINER_NAME[@]}"; do
        # 检查容器状态
        status=$(docker ps -a --filter "name=$container" --format "{{.Status}}" 2>/dev/null)

        if [ -z "$status" ]; then
            # echo "Container '$container' not found."
            dockers+="{\"name\":\"$container\", \"status\":\"not found\"},"
            docker_staus="ERROR"
        elif [[ "$status" == *"Exited"* ]]; then
            # echo "Container '$container' is stopped."
            dockers+="{\"name\":\"$container\", \"status\":\"stopped\"},"
            
            if docker start "$container" >/dev/null 2>&1; then
                if [[ "$docker_staus" == "Normal" ]]; then
                    docker_staus="WARNING"
                fi
            else
                docker_staus="ERROR"
            fi
        else
            # echo "Container '$container' is running."
            dockers+="{\"name\":\"$container\", \"status\":\"running\"},"
        fi
    done
    dockers="${dockers%,}"
    dockers+="]"
    DOCKER_REPLY="{\"status\": \"$docker_staus\", \"containers\": $dockers}"
}

# 检查挂载状态的函数
check_mount() {
    disk_usage=$(df -h)
    ssd_status="Normal"
    dirs="["
    while read -r line; do
        if [[ "$line" == *"Filesystem"* ]] || [[ "$line" == *"文件系统"* ]]; then
            continue
        fi

        # 提取各个字段
        filesystem=$(echo "$line" | awk '{print $1}')
        total_space=$(echo "$line" | awk '{print $2}')
        used_space=$(echo "$line" | awk '{print $3}')
        available_space=$(echo "$line" | awk '{print $4}')
        use_percentage=$(echo "$line" | awk '{print $5}')

        total_kb=$(echo "$total_space" | sed 's/[A-Za-z]//g')
        available_kb=$(echo "$available_space" | sed 's/[A-Za-z]//g')

        total_kb=$(echo "$total_space" | awk '{if(substr($0, length($0), 1)=="G") print $0*1024*1024; else if (substr($0, length($0), 1)=="M") print $0*1024; else print $0}')
        available_kb=$(echo "$available_space" | awk '{if(substr($0, length($0), 1)=="G") print $0*1024*1024; else if (substr($0, length($0), 1)=="M") print $0*1024; else print $0}')
        
        available_percentage=$(awk "BEGIN {printf \"%.2f\", ($available_kb/$total_kb)*100}")
        dirs+="{\"name\":\"$filesystem\", \"total_space\":\"$total_space\", \"used_space\":\"$used_space\", \"available_space\":\"$available_space\", \"use_percentage\":\"$use_percentage\", \"available_percentage\":\"$available_percentage\"},"
    done <<< "$disk_usage"
    dirs="${dirs%,}"
    dirs+="]"
    SSD_REPLY="{\"status\": \"$ssd_status\", \"dirs\": $dirs}"
}

start_http_server() {
    {
        # 读取HTTP请求并返回状态         
        read request
        if [[ -z "$request" ]]; then
            echo -e "HTTP/1.1 400 Bad Request\r"
            echo -e "Content-Type: text/plain\r"
            echo -e "Connection: close\r"
            echo -e "\r"
            echo "message: 400 Invalid URL"
        elif echo "$request" | grep -q "GET "; then
            path=$(echo "$request" | awk '{print $2}')

            if [[ "$path" == "/dockers" ]]; then
                get_container_status
                echo -e "HTTP/1.1 200 OK\r"
                echo -e "Content-Type: application/json\r"
                echo -e "Connection: close\r"
                echo -e "\r"
                echo "message: $DOCKER_REPLY"
            elif [[ "$path" == "/ssd" ]]; then
                check_mount
                echo -e "HTTP/1.1 200 OK\r"
                echo -e "Content-Type: application/json\r"
                echo -e "Connection: close\r"
                echo -e "\r"
                echo "message: $SSD_REPLY"
            else
                echo -e "HTTP/1.1 404 Not Found\r"
                echo -e "Content-Type: text/plain\r"
                echo -e "Connection: close\r"
                echo -e "\r"
                echo "message: 404 Not Found"
            fi
        else
            echo -e "HTTP/1.1 400 Bad Request\r"
            echo -e "Content-Type: text/plain\r"
            echo -e "Connection: close\r"
            echo -e "\r"
            echo "message: 400 Invalid Request"
        fi
    } | nc -l -p "$PORT" -s "$NETIP"
}


while true; do
    start_http_server
done
