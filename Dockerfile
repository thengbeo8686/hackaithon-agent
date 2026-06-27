FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# SYSTEM DEPENDENCIES
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Link python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# PROJECT SETUP
WORKDIR /code

# Copy toàn bộ repository vào thư mục /code trong container
COPY . /code

# INSTALL LIBRARIES
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD AND CACHE EMBEDDING MODEL
# Tải trước và lưu trữ model bge-m3 tại đường dẫn cục bộ để chạy offline trên máy chấm BTC
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-m3'); model.save('/app/models/bge-m3')"

# Cấp quyền chạy cho file entrypoint bash script
RUN chmod +x /code/inference.sh

# EXECUTION
CMD ["bash", "inference.sh"]
