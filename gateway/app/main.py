import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


INFERENCE_URL = os.getenv(
    "INFERENCE_URL",
    "http://inference:8000"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-1.5B-Instruct"
)

TIMEOUT = float(
    os.getenv("REQUEST_TIMEOUT", "300")
)


client = httpx.AsyncClient(
    base_url=INFERENCE_URL,
    timeout=TIMEOUT
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await client.aclose()


app = FastAPI(
    title="LLM Gateway",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(
        256,
        ge=1,
        le=4096
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0
    )
    top_p: float = Field(
        1.0,
        gt=0.0,
        le=1.0
    )
    stop: list[str] | None = None


@app.get("/health")
async def health():
    try:
        r = await client.get(
            "/health",
            timeout=5.0
        )

        upstream = (
            "up"
            if r.status_code == 200
            else "down"
        )

    except httpx.HTTPError:
        upstream = "down"

    return {
        "status": "ok",
        "inference": upstream,
        "model": MODEL_NAME
    }


@app.post("/generate")
async def generate(req: GenerateRequest):

    payload = {
        "model": MODEL_NAME,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }

    if req.stop:
        payload["stop"] = req.stop

    # Start measuring request processing time
    start_time = time.perf_counter()

    try:
        r = await client.post(
            "/v1/completions",
            json=payload
        )

    except httpx.HTTPError as exc:

        processing_time = (
            time.perf_counter() - start_time
        )

        raise HTTPException(
            status_code=502,
            detail={
                "error": f"inference unreachable: {exc}",
                "processing_time_seconds": round(
                    processing_time,
                    4
                )
            }
        )

    # Stop measuring request processing time
    processing_time = (
        time.perf_counter() - start_time
    )

    if r.status_code != 200:
        raise HTTPException(
            status_code=r.status_code,
            detail=r.text
        )

    data = r.json()

    return {
        "text": data["choices"][0]["text"],
        "finish_reason": data["choices"][0].get(
            "finish_reason"
        ),
        "usage": data.get("usage"),

        # Total time spent waiting for inference
        "processing_time_seconds": round(
            processing_time,
            4
        )
    }