FROM python:3.10-slim

# Set working directory
WORKDIR /work

# Install dependencies
COPY ai_detector_backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code including the folder structure
COPY ai_detector_backend/ ./ai_detector_backend/

# Set PYTHONPATH to ensure modules can be found
ENV PYTHONPATH=/work

# Expose FastAPI on port 8000
EXPOSE 8000

# Run app
CMD ["uvicorn", "ai_detector_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
