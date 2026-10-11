FROM rust:1-bookworm@sha256:114c7a4425406451c2866b6aafe69fe29b1b298832db1277d411ac73c82d04d6 AS native
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY rust ./rust
RUN cargo build --release --locked

FROM python:3.12-slim-bookworm@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
WORKDIR /app
COPY pyproject.toml ./
COPY python ./python
RUN pip install --no-cache-dir .
COPY examples ./examples
COPY --from=native /build/target/release/libnlbridge_core.so /opt/nlbridge/libnlbridge_core.so
ENV NLBRIDGE_NATIVE=/opt/nlbridge/libnlbridge_core.so
RUN useradd --uid 10001 --create-home bridge && mkdir -p /app/state && chown bridge /app/state
USER bridge
EXPOSE 8080
CMD ["nlbridge", "--config", "examples/offline.toml", "serve", "--host", "0.0.0.0"]
