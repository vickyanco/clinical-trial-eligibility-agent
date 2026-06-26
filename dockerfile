# Clinical Trial Eligibility Agent — containerised FastAPI service
#
# Build:
#   docker build -t eligibility-agent .
#
# Run:
#   docker run -p 8000:8000 eligibility-agent
#
# Run with a real Gemini API key instead of the mock LLM:
#   docker run -p 8000:8000 -e GEMINI_API_KEY=your-key-here eligibility-agent

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]