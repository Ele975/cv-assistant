# 1. Use lightweight python base
FROM python:3.12-slim

# 2. Set working directory inside /app
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-dev \
    libopenblas-dev \ 
    && rm -rf /var/lib/apt/lists/*


# 4. Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project files
COPY . . 

# 6. Expose Gradio default port
EXPOSE 7860

# 7. Run app
# CMD ["python", "src/app.py"]
