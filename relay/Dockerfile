FROM python:3.12-slim
WORKDIR /app
# Burned-in subtitles: a font with Hebrew/Arabic glyphs, and FriBiDi so Pillow
# lays out right-to-left text.
RUN apt-get update && apt-get install -y --no-install-recommends libfribidi0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py *.json ui.html ./
ARG VERSION=dev
RUN echo "$VERSION" > VERSION   # shown in /status and the web page
ENV PYTHONUNBUFFERED=1
EXPOSE 8095
CMD ["python", "relay.py"]
