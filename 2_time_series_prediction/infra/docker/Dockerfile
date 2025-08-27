# Base image with Python 3.11 and Tensorflow(GPU supported)
FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

# Avoid prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# (Optional) mirror server setting
RUN sed -i 's|archive.ubuntu.com|mirror.kakao.com|g' /etc/apt/sources.list

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

WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip

# Install Jupyter dependencies
RUN pip install --upgrade ipykernel jupyter jupyterlab ipywidgets

# Install Python dependencies
RUN pip install \
    numpy \
    matplotlib \
    imageio \
    keras==3.0.2 \
    ipython \
    pillow \
    geopandas \
    rasterio \
    shapely

# Install Jupyter widget
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build || true
RUN jupyter lab build || true

# Register kernal (Python 3.11 path)
RUN python -m ipykernel install --sys-prefix --name python311 --display-name "Python 3.11"

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
