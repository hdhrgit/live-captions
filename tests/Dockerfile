# Test image: the relay's runtime plus pytest (and git, for version.sh).
#   docker build -t captions-dvr-tests -f tests/Dockerfile .
#   docker run --rm captions-dvr-tests            # unit + source tests
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libfribidi0 fonts-dejavu-core git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY relay/requirements.txt relay/requirements.txt
COPY tests/requirements.txt tests/requirements.txt
RUN pip install --no-cache-dir -r tests/requirements.txt
COPY . .
CMD ["python", "-m", "pytest", "tests/unit", "tests/sources"]
