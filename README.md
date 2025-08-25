# Deep Learning Reconstruction of Atmospheric CO₂ and Climate State from Vostok Ice Core

A comprehensive deep learning approach to reconstruct atmospheric CO₂ concentrations and classify paleoclimate states using proxy data from the iconic Vostok Ice Core dataset.

## Overview

This project explores the application of deep learning techniques to paleoclimatology, specifically using sequence modeling to understand Earth's ancient climate patterns. We implement two complementary neural network architectures to analyze over 400,000 years of Antarctic ice core data, providing insights into natural climate variability and glacial-interglacial cycles.

## Dataset: Vostok Ice Core

The Vostok ice core, extracted from Antarctica's Vostok Station, represents one of the most significant paleoclimatological datasets ever collected. This ice core contains a continuous record spanning approximately 400,000 years, providing unprecedented insight into Earth's climate history through multiple glacial and interglacial periods.

### Scientific Significance

The Vostok ice core data reveals:
- **Atmospheric Composition**: Direct measurements of ancient atmospheric CO₂ and CH₄ concentrations trapped in air bubbles
- **Temperature Records**: Deuterium isotope ratios (δD) that serve as proxies for past temperatures
- **Dust Levels**: Indicators of atmospheric circulation patterns and aridity
- **Climate Transitions**: Evidence of rapid climate shifts and terminations of ice ages

### Data Sources

The raw data files used in this project were published by NOAA's Paleoclimatology Division:
- `co2nat.txt`: Atmospheric CO₂ concentrations (ppm)
- `ch4nat.txt`: Atmospheric methane concentrations (ppb)
- `deutnat.txt`: Deuterium isotope ratios and derived temperatures
- `dustnat.txt`: Dust concentration levels
- `gt4nat.txt`: Gas age chronology for data alignment

**Repository**: [Vostok Ice Core Data - USAP DC](https://www.usap-dc.org/view/dataset/609242)

### Data Processing and Preprocessing Pipeline

The Vostok dataset represents a unique challenge in paleoclimatology - it's a "vintage" dataset from the early days of ice core research, stored in outdated text formats that require extensive preprocessing to become suitable for modern machine learning applications. Despite being downloaded only **13 times since 2017** due to its complexity, this dataset contains invaluable 400,000-year climate records.

#### Challenges with the Raw Dataset
- **Inconsistent Formats**: Multiple text files with varying column structures and delimiters
- **Mixed Chronologies**: Ice ages vs. gas ages requiring careful temporal alignment
- **Irregular Sampling**: Non-uniform time intervals across different proxy measurements
- **Missing Data**: Gaps and inconsistencies typical of early paleoclimate datasets
- **Legacy Encoding**: Files stored with Latin-1 encoding from older systems

#### Preprocessing Pipeline

The transformation from raw vintage data to ML-ready format involves several critical stages:

1. **File Parsing and Cleaning**
   - Custom parsing functions to handle inconsistent delimiters and column structures
   - Removal of header comments, separator lines, and invalid entries
   - Handling of non-standard numeric formats and special characters
   - Conversion from Latin-1 encoding to modern UTF-8

2. **Chronological Alignment**
   - **Gas-age conversion**: Converting ice ages to gas ages using the `gt4nat.txt` chronology
   - **Temporal interpolation**: Creating a master gas-age to ice-age mapping function
   - **Synchronization**: Aligning all proxy records to a common temporal framework

3. **Data Harmonization**
   - **Uniform resampling**: Interpolating all proxies to a consistent 100-year grid
   - **Gap filling**: Linear interpolation for small gaps, careful handling of larger gaps
   - **Outlier detection**: Statistical identification and treatment of anomalous values
   - **Data validation**: Cross-checking between different proxy measurements

4. **Feature Engineering**
   - **Temperature derivation**: Converting δD isotope ratios to temperature estimates (δD × 0.015)
   - **Climate state labeling**: Data-driven classification based on temperature quantiles
   - **Derivative features**: Computing temperature gradients (dT/dt) for trend analysis
   - **Quality indicators**: Creating flags for data reliability and interpolation status

5. **Machine Learning Preparation**
   - **Normalization**: StandardScaler fitting on training data only
   - **Sequence windowing**: Creating 64-timestep sliding windows (6,400 years each)
   - **Train/validation/test splitting**: Group-based splitting to prevent temporal leakage
   - **Target variable preparation**: Separate preparation for regression (CO₂) and classification (warm/cold) tasks

#### Technical Implementation

```python
# Example of the vintage data parsing challenge
def read_pair(path: pl.Path, names, usecols=(0, 1)):
    rows = []
    with path.open(encoding="latin1") as fh:  # Legacy encoding
        for ln in fh:
            ln = ln.strip()
            # Skip headers, comments, and separator lines
            if (not ln or ln.startswith("#") or "---" in ln
                    or not re.match(r"^\s*-?\d", ln)):
                continue
            # Handle irregular delimiters
            cells = re.split(r"[\t ]+", ln)
            if len(cells) < max(usecols) + 1:
                continue
            try:
                rows.append([float(cells[i]) for i in usecols])
            except ValueError:
                continue  # Skip malformed rows
    return pd.DataFrame(rows, columns=names)
```

#### Data Quality Assessment

The preprocessing pipeline transforms:
- **Raw files**: 5 separate text files with inconsistent formats
- **Temporal coverage**: ~400,000 years with irregular sampling
- **Final dataset**: 1,170 samples at uniform 100-year resolution
- **Completeness**: >99% data coverage after interpolation
- **Quality**: Validated against published paleoclimate records

This extensive preprocessing demonstrates how modern data science techniques can rescue and repurpose valuable historical scientific datasets, making them accessible for contemporary machine learning applications.

## Research Reports

This repository includes two comprehensive research reports documenting the methodology, results, and implications:

### 1. Deep Learning Reconstruction Report
**File**: `Deep_Learning_Reconstruction_of_Atmospheric_CO2_and_Climate_state_from_Vostok (3).pdf`

This report covers:
- Detailed methodology for the dilated CNN regression model
- Bidirectional LSTM classification approach
- Model architecture design and hyperparameter optimization
- Performance evaluation and validation strategies
- Discussion of results in paleoclimatological context

### 2. Final Project Report
**File**: `Vostok_Icecore_Deeplearning_Final_Project (2).pdf`

This comprehensive document includes:
- Literature review of paleoclimate reconstruction methods
- Complete data preprocessing pipeline
- Comparative analysis of deep learning architectures
- Statistical validation and uncertainty quantification
- Implications for understanding natural climate variability

## Methodology

### Problem Formulation

We treat paleoclimate reconstruction as two complementary sequence modeling problems:

1. **Regression Task**: Predicting atmospheric CO₂ concentrations from proxy signals (δD, CH₄, dust)
2. **Classification Task**: Determining climate state (warm vs. cold periods) from the same proxies

### Model Architectures

#### Dilated Convolutional Neural Network (CO₂ Regression)
- **Architecture**: 1D dilated convolutions with increasing dilation rates (1, 2, 4, 8, 16)
- **Receptive Field**: Large temporal context without excessive parameters
- **Output**: Continuous CO₂ concentration predictions
- **Performance**: MAE = 2.7 ppm, R² = 0.98

#### Bidirectional LSTM (Climate Classification)
- **Architecture**: Stacked bidirectional LSTM layers with dense classification head
- **Temporal Modeling**: Captures dependencies from both past and future contexts
- **Output**: Binary classification (warm/cold climate state)
- **Performance**: AUC = 0.98, F1-score = 0.92

### Key Features

- **Sliding Window Approach**: 64-timestep windows (6,400 years) for sequence learning
- **Group-based Splitting**: Prevents data leakage between train/validation/test sets
- **Standardization**: Feature scaling based on training set statistics only
- **Temporal Validation**: Future hold-out testing for generalization assessment

## Repository Structure

```
vostok-dl/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
├── Vostok_Icecore_Deeplearning_Project.ipynb  # Main analysis notebook
├── reports/
│   ├── Deep_Learning_Reconstruction_of_Atmospheric_CO2_and_Climate_state_from_Vostok (3).pdf
│   └── Vostok_Icecore_Deeplearning_Final_Project (2).pdf
├── data/
│   └── vostok_clean.csv              # Processed dataset (generated)
├── models/
│   ├── dilated_cnn_regressor.h5      # Trained CNN model (generated)
│   └── bilstm_classifier.h5          # Trained LSTM model (generated)
└── figures/
    ├── dilated_cnn_arch.png          # Model architecture diagrams (generated)
    ├── bilstm_classifier_arch.png    # (generated)
    └── vostok_all_plots.png          # Results visualization (generated)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vostok-dl.git
cd vostok-dl
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter notebook:
```bash
jupyter lab
```

## Usage

### Running the Analysis

1. Open `Vostok_Icecore_Deeplearning_Project.ipynb` in Jupyter Lab
2. Execute cells sequentially to:
   - Load and preprocess the raw Vostok data
   - Create training/validation/test splits
   - Train the deep learning models
   - Evaluate performance and generate visualizations

### Key Outputs

- **Processed Dataset**: `data/vostok_clean.csv` (1,170 samples at 100-year resolution)
- **Trained Models**: Saved in `models/` directory for future use
- **Visualizations**: Performance plots and model architectures in `figures/`

### Model Performance

| Model | Task | Metric | Performance |
|-------|------|---------|-------------|
| Dilated CNN | CO₂ Regression | MAE | 2.7 ppm |
| Dilated CNN | CO₂ Regression | R² | 0.98 |
| Bidirectional LSTM | Climate Classification | AUC | 0.98 |
| Bidirectional LSTM | Climate Classification | F1-Score | 0.92 |

## Scientific Impact

This work demonstrates the potential of deep learning for paleoclimate reconstruction, offering:

- **High-precision CO₂ estimates** from indirect proxy measurements
- **Automated climate state detection** for large paleoclimate datasets
- **Methodological framework** applicable to other ice core and paleoclimate records
- **Validation approach** ensuring temporal generalization

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests. For major changes, please open an issue first to discuss the proposed modifications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NOAA Paleoclimatology Division for providing the Vostok ice core dataset
- The Vostok drilling team and Russian Antarctic Expedition
- TensorFlow and scikit-learn communities for the deep learning frameworks

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{vostok_deep_learning,
  title={Deep Learning Reconstruction of Atmospheric CO₂ and Climate State from Vostok Ice Core},
  author={[Mohammed Noushir]},
  year={2025},
  url={https://github.com/Noushir/Vostok_DL}
}
```

## Contact

For questions or collaborations, please open an issue or contact [mhmdnoushir.k@gmail.com].
