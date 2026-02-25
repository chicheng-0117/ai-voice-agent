#!/usr/bin/env python3
"""
调用自建 Qwen3-ASR 做一次转录测试。

用法:
  python test_asr_call.py                          # 用环境变量，默认 test.wav
  python test_asr_call.py --file /path/to.wav      # 指定音频文件
  python test_asr_call.py --url http://IP:80/v1    # 指定 ASR 地址

环境变量（可选）:
  QWEN_ASR_BASE_URL  如 http://localhost:80/v1
  QWEN_ASR_API_KEY   自建一般填 EMPTY
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(".env.local")

try:
    from openai import OpenAI
except ImportError:
    print("请先安装: pip install openai")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="调用 Qwen3-ASR 测试转录")
    parser.add_argument("--url", type=str, default=os.getenv("QWEN_ASR_BASE_URL", "").strip().rstrip("/"), help="ASR base URL")
    parser.add_argument("--file", type=str, default="english.mp3", help="音频文件路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-0.6B", help="模型名")
    args = parser.parse_args()

    if not args.url:
        print("未指定 ASR 地址，请设置 QWEN_ASR_BASE_URL 或使用 --url http://IP:端口/v1")
        sys.exit(1)
    if not os.path.isfile(args.file):
        print(f"音频文件不存在: {args.file}")
        sys.exit(1)

    api_key = os.getenv("QWEN_ASR_API_KEY", "EMPTY").strip()
    client = OpenAI(base_url=args.url, api_key=api_key)

    print(f"请求: {args.url}  model={args.model}  file={args.file}")
    with open(args.file, "rb") as f:
        resp = client.audio.transcriptions.create(model=args.model, file=f, language=None)
    text = getattr(resp, "text", None) or str(resp)
    print("识别结果:", text)
    return 0


if __name__ == "__main__":
    sys.exit(main())