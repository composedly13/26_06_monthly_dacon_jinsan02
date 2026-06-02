#!/bin/bash
echo "=== DNS resolve ==="
timeout 15 getent hosts pypi.org || echo "DNS FAILED"
echo "=== throughput 10MB ==="
timeout 60 curl -s -o /dev/null \
  -w "speed=%{speed_download}B/s dns=%{time_namelookup}s connect=%{time_connect}s total=%{time_total}s bytes=%{size_download}\n" \
  "https://speed.cloudflare.com/__down?bytes=10000000" || echo "DOWNLOAD FAILED/timeout"
