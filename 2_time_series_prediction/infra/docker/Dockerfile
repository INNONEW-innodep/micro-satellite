# Base image with Python 3.10 and necessary scientific stack
FROM python:3.10-slim

# Avoid prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install \
    numpy \
    matplotlib \
    imageio \
    ipywidgets \
    jupyter \
    keras==3.02 \
    tensorflow \
    ipython \
    pillow \
    geopandas \
    rasterio \
    shapely

# Enable Jupyter widget extensions
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build || true
RUN jupyter lab build || true

# Expose Jupyter port
EXPOSE 8888

# Launch Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]