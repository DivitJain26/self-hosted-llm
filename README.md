# llm-stack

Client → FastAPI gateway → vLLM → mounted model → back.

- `gateway`   FastAPI + httpx. Port 8000 published to host.
- `inference` vLLM OpenAI-compatible server. Internal network only.
- `models/`   Bind-mounted. Weights and caches live here, never in an image.

## Local (WSL / Linux)

```bash
docker compose build
docker compose up -d
docker compose logs -f inference        # wait for "Application startup complete"
curl -s localhost:8000/health
curl -s -X POST localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain vector norms in one sentence.","max_tokens":64}'
```

WSL needs ≥12 GB for the 1.5B in float32. In `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=8GB
```

then `wsl --shutdown` from PowerShell.

## Apply any config change

`.env` and `command:` changes only reach a **recreated** container.

```bash
docker compose up -d --force-recreate
```

`docker compose restart` and `up -d` alone reuse the old container.

## Swap the model (no rebuild)

```bash
sed -i 's|^MODEL_NAME=.*|MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct|' .env
docker compose up -d --force-recreate inference gateway
```

Any HF id or a local path under `/models` works.

## Publish images

```bash
docker login
docker tag llm-gateway:latest   divit26j/llm-gateway:0.1
docker tag llm-inference:latest divit26j/llm-inference:0.1
docker push divit26j/llm-gateway:0.1
docker push divit26j/llm-inference:0.1
```

Weights are not in either image. Bump the tag for every rebuild.

## EC2 (x86, ≥16 GB RAM, 30 GB disk)

Security group: 22 and 8000 from your IP only.

```bash
# Docker
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

# Project
mkdir -p ~/llm-stack/models && cd ~/llm-stack
# scp docker-compose.prod.yml and .env.ec2 here
mv .env.ec2 .env
sed -i "s|^VLLM_CPU_OMP_THREADS_BIND=.*|VLLM_CPU_OMP_THREADS_BIND=0-$(( $(nproc) - 1 ))|" .env

docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml logs -f inference
```

Weights download on the instance into `./models/hf`. Nothing is copied from your laptop.

Verify isolation:

```bash
curl -s http://<ec2-ip>:8000/health         # works
curl -m 3 http://<ec2-ip>:8001/v1/models    # refused — vLLM not exposed
```

## Deploy a new version

Local:

```bash
docker compose build
docker tag llm-gateway:latest divit26j/llm-gateway:0.2 && docker push divit26j/llm-gateway:0.2
```

EC2:

```bash
sed -i 's|^TAG=.*|TAG=0.2|' .env
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d --force-recreate
```

## Day to day

```bash
docker compose ps
docker compose logs -f gateway
docker compose down                  # models/ survives
docker compose config                # show resolved env — check before debugging
docker stats --no-stream             # CPU% ≈ nproc×100 means all cores engaged
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `inference: down`, 502 from gateway | vLLM not up yet or crashed | `docker compose logs inference` |
| `Available memory ... less than requested memory for kv` | not enough RAM | lower `VLLM_CPU_KVCACHE_SPACE`, smaller model, or more RAM |
| Config edit has no effect | stale container | `up -d --force-recreate` |
| `core ids=[N]` shows one core | auto binding on small VM | set `VLLM_CPU_OMP_THREADS_BIND=0-<n-1>`, reserved CPU `0` |
| Stuck at "Warming up model" for minutes | one core + big warmup | fix binding; keep `MAX_NUM_BATCHED_TOKENS=512` |
| `Failed to create oneDNN linear` | bf16 needs AVX512-BF16/AMX | `DTYPE=float32`, or use a c7i/m7i instance |
| `Failed to infer device type` | CUDA image on a box with no GPU | use the CPU image (`VLLM_IMAGE`) |
| `invalid literal for int(): 'all'` | wrong thread-bind syntax | use a range like `0-3`, or `auto` |

## Performance

```bash
time curl -s -X POST localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain vector norms in one sentence.","max_tokens":64}'
```

tokens/sec ≈ 64 ÷ elapsed. On a 4-vCPU instance without AMX expect low single digits
to low teens. For a real speedup inside vLLM, use `c7i.2xlarge`+ with `DTYPE=bfloat16`.# self-hosted-llm
