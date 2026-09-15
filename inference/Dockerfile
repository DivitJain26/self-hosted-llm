# The model is deliberately NOT copied in. It is mounted at /models at runtime.
ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

ENV HF_HOME=/models/hf
VOLUME ["/models"]
EXPOSE 8000

# Base image ENTRYPOINT is the vLLM OpenAI API server;
# compose supplies the flags via `command:`.