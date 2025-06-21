FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Instalacja zależności systemowych (w tym libgomp1 dla LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file for GUI
COPY requirements.txt ./

# Install GUI dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the GUI application code and necessary files
COPY src/ ./src/
COPY model ./model

# Expose port for Streamlit
EXPOSE 8501

# Start Streamlit server
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]