# Multi-stage build for csm-ws with CUDA + cuDNN support.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    gcc-10 \
    g++-10 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-10 \
    CXX=g++-10 \
    NVCC_CCBIN=/usr/bin/gcc-10

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

COPY Cargo.toml ./
COPY csm-core ./csm-core
COPY csm-ws ./csm-ws

# Override per GPU: 80=A100, 86=RTX 30xx, 89=RTX 40xx, 90=H100.
ARG CUDA_COMPUTE_CAP=89
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

RUN cargo build --release -p csm-ws --features cudnn

# ----- runtime stage -----
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 csm && \
    mkdir -p /app /models && \
    chown -R csm:csm /app /models

WORKDIR /app
COPY --from=builder /build/target/release/csm-ws /app/csm-ws

USER csm

EXPOSE 8080
ENV RUST_LOG=info \
    HOST=0.0.0.0 \
    PORT=8080 \
    POOL_SIZE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["/app/csm-ws"]
CMD ["--model-id", "sesame/csm-1b"]
