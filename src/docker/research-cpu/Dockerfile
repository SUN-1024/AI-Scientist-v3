FROM node:22-bookworm-slim AS node-runtime
FROM python:3.12-slim

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

ARG DEBIAN_MIRROR=https://mirrors.aliyun.com/debian
ARG DEBIAN_SECURITY_MIRROR=https://mirrors.aliyun.com/debian-security
ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ENV DEBIAN_FRONTEND=noninteractive
RUN set -eu; . /etc/os-release; \
    rm -f /etc/apt/sources.list.d/debian.sources; \
    printf 'deb %s %s main\ndeb %s %s-updates main\ndeb %s %s-security main\n' \
      "$DEBIAN_MIRROR" "$VERSION_CODENAME" "$DEBIAN_MIRROR" "$VERSION_CODENAME" \
      "$DEBIAN_SECURITY_MIRROR" "$VERSION_CODENAME" > /etc/apt/sources.list; \
    apt-get update -o Acquire::Retries=3; \
    apt-get install -y --no-install-recommends bash bc ca-certificates curl git jq \
      procps ripgrep time poppler-utils librsvg2-bin \
      texlive-bibtex-extra texlive-fonts-recommended texlive-latex-base \
      texlive-latex-extra texlive-latex-recommended; \
    rm -rf /var/lib/apt/lists/*

ARG CODEX_VERSION=0.153.4
RUN npm install -g "@openai/codex@${CODEX_VERSION}"
# build-research-cpu-image.sh supplies an archive of the verified pinned revision.
COPY headless /opt/headless
RUN npm ci --ignore-scripts --prefix /opt/headless \
    && npm run build --prefix /opt/headless \
    && npm prune --ignore-scripts --omit=dev --prefix /opt/headless \
    && ln -s /opt/headless/dist/cli.js /usr/local/bin/headless \
    && headless --version
ENV HEADLESS_BIN=/usr/local/bin/headless
RUN pip install --no-cache-dir "httpx==0.27.2" "scholarly==1.7.11" \
      "numpy==2.4.2" "pandas==3.0.1" "matplotlib==3.11.1" "pypdf==5.9.0"
ARG TORCH_CPU_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
# Contracted CPU tasks pin torch's +cpu build. Keep normal dependencies on PyPI.
ENV PIP_EXTRA_INDEX_URL=${TORCH_CPU_INDEX_URL}
RUN useradd -m -u 1000 agent && chmod 755 /home/agent
USER agent
ENV PATH="/home/agent/.local/bin:${PATH}"
WORKDIR /workspace
CMD ["bash"]
