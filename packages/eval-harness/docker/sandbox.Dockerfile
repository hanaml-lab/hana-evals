# A fresh, unprivileged sandbox for each run: the agent's tools execute here, not on the host.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 agent
WORKDIR /home/agent/work

COPY packages/eval-harness /opt/eval-harness
RUN pip install --no-cache-dir /opt/eval-harness

USER agent
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["eval-harness"]
CMD ["--help"]
