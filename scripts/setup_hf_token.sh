#!/bin/bash
# HF 토큰 등록
TOKEN="$1"
mkdir -p ~/.cache/huggingface
echo -n "$TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
echo "Token registered."
python3 -c "from huggingface_hub import HfApi; api=HfApi(); u=api.whoami(token='$TOKEN'); print('Logged in as:', u['name'])"
