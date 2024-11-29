#!/bin/bash
 
# 启动HTTP服务器，监听8080端口
while true; do
nc -l -p 8080 -s "172.16.115.46" | {
    echo -e "HTTP/1.1 200 OK\r"
    echo -e "Content-Type: text/plain\r"
    echo "\r"
    echo "a: Hello, World!"
    }
done
# {} | nc -l -p 8080 -s "172.16.115.46" << RESPONSE
# HTTP/1.1 200 OK
# Content-Type: text/plain
# Connection: close
 
# a: Hello, World!
# RESPONSE