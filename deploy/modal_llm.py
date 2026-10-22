"""Serve Dolphin's model on Modal: vLLM, OpenAI-compatible API, bearer-key protected, scale-to-zero.

Deploy (see deploy/deploy_modal.sh, which also creates the secret without printing the key):
    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... bash deploy/deploy_modal.sh

Endpoints (each model is its own scale-to-zero container on one L4 GPU):
    https://<workspace>--sic-llm-chat.modal.run/v1/chat/completions   model "dolphin3-8b"
    https://<workspace>--sic-llm-coder.modal.run/v1/...                model "qwen2.5-coder-14b" (only if SIC_DEPLOY_CODER=1)

Choices:
* GPU L4 (24 GB, $0.80/h on Modal): cheapest GPU that runs an 8B model fast. Dolphin 3 8B is served
  with on-the-fly FP8 weights (Ada/L4 supports FP8): ~9 GB, ~2x faster decoding than bf16.
* Qwen2.5-Coder-14B-Instruct-AWQ (4-bit, ~10 GB) also fits an L4; optional second endpoint.
* scaledown_window=300: container stays warm 5 min after the last request, then scales to zero ($0).
* Weights live in a Modal Volume so cold starts don't re-download (~16 GB first time only).
* Auth: vLLM checks `Authorization: Bearer $VLLM_API_KEY`; the key comes from the Modal Secret
  "sic-llm-key" (created from the gitignored .modal_api_key). /health is unauthenticated.
"""
import os
import subprocess

import modal

APP_NAME = "sic-llm"
GPU = os.environ.get("SIC_MODAL_GPU", "L4")
SCALEDOWN = int(os.environ.get("SIC_MODAL_SCALEDOWN", "300"))
DEPLOY_CODER = os.environ.get("SIC_DEPLOY_CODER", "0") == "1"

CHAT_MODEL = {"repo": "dphn/Dolphin3.0-Llama3.1-8B", "name": "dolphin3-8b",
              "args": ["--quantization", "fp8", "--max-model-len", "16384"]}
CODER_MODEL = {"repo": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ", "name": "qwen2.5-coder-14b",
               "args": ["--quantization", "awq_marlin", "--max-model-len", "16384"]}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.32.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf", "VLLM_CACHE_ROOT": "/cache/vllm"})
)
cache = modal.Volume.from_name("sic-llm-cache", create_if_missing=True)
key = modal.Secret.from_name("sic-llm-key")  # VLLM_API_KEY
app = modal.App(APP_NAME, image=image)


def _serve(m: dict):
    cmd = ["vllm", "serve", m["repo"], "--served-model-name", m["name"], "--host", "0.0.0.0",
           "--port", "8000", "--gpu-memory-utilization", "0.92", "--enable-prefix-caching",
           *m["args"]]
    subprocess.Popen(cmd)  # vLLM reads VLLM_API_KEY from the env and enforces it on /v1/*


FN = dict(gpu=GPU, volumes={"/cache": cache}, secrets=[key], scaledown_window=SCALEDOWN,
          timeout=60 * 60, max_containers=1)  # max 1 container: caps spend for a single user


@app.function(**FN)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def chat():
    _serve(CHAT_MODEL)


if DEPLOY_CODER:
    @app.function(**FN)
    @modal.concurrent(max_inputs=8)
    @modal.web_server(port=8000, startup_timeout=15 * 60)
    def coder():
        _serve(CODER_MODEL)
