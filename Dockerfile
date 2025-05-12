# Use updated Python base
FROM python:3.10-slim

# Use bash as the default shell
SHELL ["/bin/bash", "-c"]

# System-level dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git wget build-essential \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libglvnd-dev libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install compatible pip tools
RUN python3 -m pip install --no-cache-dir pip==21.3.1 setuptools==59.5.0 wheel

# Install compatible numpy
RUN pip install --no-cache-dir numpy==1.26.4

# Install DreamerV3 and its configs.yaml
RUN pip install --no-cache-dir "dreamerv3 @ git+https://github.com/AndrejOrsula/dreamerv3.git@d4f47fcb18f52777314f2735389cd1f449513c9a" && \
    wget -q https://raw.githubusercontent.com/AndrejOrsula/dreamerv3/main/dreamerv3/configs.yaml \
    -O "$(pip show dreamerv3 | grep Location: | cut -d' ' -f2)/dreamerv3/configs.yaml"

# Install YAML library needed for patched config saving
RUN pip install --no-cache-dir ruamel.yaml

# Patch deprecated safe_dump usage in DreamerV3 config
RUN sed -i '/yaml.safe_dump/d' /usr/local/lib/python3.10/site-packages/dreamerv3/embodied/core/config.py && \
    sed -i '/with io.StringIO() as stream:/a\        from ruamel.yaml import YAML\n        yaml = YAML(typ="safe", pure=True)\n        yaml.dump(dict(self), stream)' \
    /usr/local/lib/python3.10/site-packages/dreamerv3/embodied/core/config.py

# Clone air_hockey_challenge including XMLs and install it
RUN git clone https://github.com/AndrejOrsula/air_hockey_challenge.git /air_hockey_challenge && \
    pip install --no-cache-dir -e /air_hockey_challenge

# Copy your project into the image and install it
COPY . /src
RUN pip install --no-cache-dir -e /src

# Set PYTHONPATH for both cloned projects
ENV PYTHONPATH="/air_hockey_challenge:/src/2023-challenge:${PYTHONPATH}"

# Set working directory
WORKDIR /src

# Default run command
CMD ["python3", "/src/scripts/train_dreamerv3.py"]
