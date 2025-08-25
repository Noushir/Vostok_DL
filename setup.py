"""
Setup script for the Vostok Deep Learning project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vostok-deep-learning",
    version="1.0.0",
    author="Vostok Deep Learning Project",
    author_email="your.email@domain.com",
    description="Deep Learning Reconstruction of Atmospheric CO₂ and Climate State from Vostok Ice Core",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vostok-dl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ]
    },
    keywords="paleoclimatology, deep learning, ice core, climate reconstruction, CO2, vostok",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/vostok-dl/issues",
        "Source": "https://github.com/yourusername/vostok-dl",
        "Documentation": "https://github.com/yourusername/vostok-dl/blob/main/README.md",
    },
)
