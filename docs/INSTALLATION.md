# Installation Guide

This guide will help you set up the Vostok Deep Learning project on your local machine.

## Prerequisites

- Python 3.8 or higher
- Git
- At least 4GB of RAM (recommended 8GB+)
- CUDA-compatible GPU (optional, for faster training)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vostok-dl.git
cd vostok-dl
```

### 2. Create Virtual Environment

#### Using conda (recommended)
```bash
conda create -n vostok-dl python=3.9
conda activate vostok-dl
```

#### Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

Start Jupyter Lab and open the main notebook:

```bash
jupyter lab
```

Navigate to `Vostok_Icecore_Deeplearning_Project.ipynb` and run the first few cells to verify everything works.

## GPU Support (Optional)

For faster training, install TensorFlow with GPU support:

```bash
pip install tensorflow[and-cuda]
```

Note: Ensure you have compatible NVIDIA drivers and CUDA installed.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed in the correct environment
2. **Memory errors**: Reduce batch size in the notebook if you encounter memory issues
3. **CUDA errors**: Fall back to CPU-only execution if GPU setup fails

### Environment Verification

Run this Python script to verify your setup:

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print("Setup successful!")
```

## Development Setup

If you plan to contribute to the project:

```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

## Docker Setup (Alternative)

A Docker setup is available for reproducible environments:

```bash
docker build -t vostok-dl .
docker run -p 8888:8888 -v $(pwd):/workspace vostok-dl
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with your error message and environment details
