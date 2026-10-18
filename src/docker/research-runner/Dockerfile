ARG RESEARCH_CPU_IMAGE=unified-ai-scientist/research-cpu:2026.10
FROM ${RESEARCH_CPU_IMAGE}

USER root
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# Debian's docker-cli package contains the client, not the Docker daemon.
# The base image already configures the Aliyun Debian mirror.
RUN apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends docker-cli \
    && rm -rf /var/lib/apt/lists/*

# Exported from uv.lock by the build script. The project itself is loaded from
# the mounted checkout. Never import a macOS .venv into the Linux runtime.
COPY requirements.lock.txt /opt/research-runner/requirements.lock.txt
RUN python -m venv /opt/research-runner/venv \
    && /opt/research-runner/venv/bin/python -m pip install --no-cache-dir \
      --require-hashes -r /opt/research-runner/requirements.lock.txt
ENV PATH="/opt/research-runner/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Legacy standalone review uses PEP 723 script environments. Keep the tool
# pinned; its cache and downloaded environments stay in the private runner HOME.
ARG UV_VERSION=0.11.12
RUN /usr/local/bin/python -m pip install --no-cache-dir "uv==${UV_VERSION}"
ENV UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
COPY entrypoint.sh preflight.py native_state.py /opt/research-runner/
RUN chmod 755 /opt/research-runner/entrypoint.sh
ENTRYPOINT ["/opt/research-runner/entrypoint.sh"]
CMD ["--help"]
