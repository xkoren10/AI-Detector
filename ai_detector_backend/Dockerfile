FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY ai_detector_backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY ai_detector_backend/ .

# Expose FastAPI on port 8000
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
