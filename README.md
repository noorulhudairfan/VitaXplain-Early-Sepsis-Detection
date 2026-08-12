
# VitaXplain — Early Sepsis Detection

## Overview

**VitaXplain** is an end-to-end machine learning and deep learning framework designed to detect the early onset of sepsis from time-series ICU patient data.

The system analyses up to **15 hours of accumulated physiological data before confirmed sepsis onset** and learns temporal patterns associated with physiological deterioration. The proposed framework combines synthetic time-series data generation, deep temporal modelling, ensemble learning, uncertainty estimation, and explainable AI.

The project was developed as a Final Year Project with a focus on building an AI-driven clinical decision-support system that can provide an early warning signal before a patient's condition becomes critical.

> **Research Disclaimer:** This project is an academic/research prototype and is not intended for direct clinical diagnosis or treatment.

---

## Key Features

* 15-hour ICU time-series prediction windows
* Patient-level data splitting to reduce data leakage
* Missing-value imputation and temporal feature engineering
* 108 engineered features per timestep
* GRU-based **TimeGAN** for synthetic sepsis sequence generation
* Clinically constrained synthetic data generation
* Proposed **Multi-Scale Temporal Attention (MSTA)** deep learning architecture
* BiGRU-based temporal representation learning
* Short-range and long-range temporal attention
* XGBoost, LightGBM, and Random Forest ensemble models
* Logistic Regression stacking meta-learner
* Monte Carlo Dropout for predictive uncertainty
* SHAP-based explainability
* Clinical Alignment Score (CAS) for measuring alignment with SOFA-related clinical features
* External evaluation using data from a different institutional source
* Early-warning analysis across different available lead times

---

## Dataset

The project combines publicly available ICU datasets obtained from **PhysioNet and Kaggle**.

Three dataset groups were used throughout the pipeline:

| Dataset | Description                                                                 | Role                                               |
| ------- | --------------------------------------------------------------------------- | -------------------------------------------------- |
| **DS1** | Static clinical records containing SOFA/APACHE-related information          | Explainability and clinical validation             |
| **DS2** | Hourly ICU time-series data                                                 | Primary training, validation, and internal testing |
| **DS3** | Patient-level ICU time-series records from a different institutional source | Training augmentation pool and external evaluation |

The datasets share a clinical feature space containing vital signs, laboratory measurements, demographic information, and the binary `SepsisLabel`.

### Clinical Features

The feature set includes measurements such as:

* Heart Rate (HR)
* Oxygen Saturation (O2Sat)
* Temperature
* Systolic, Mean and Diastolic Blood Pressure
* Respiratory Rate
* FiO2
* pH
* PaCO2
* Creatinine
* Bilirubin
* Lactate
* Glucose
* WBC
* Platelets
* Hemoglobin
* BUN
* Electrolytes
* Age and demographic variables
* ICU-related temporal variables

The raw clinical variables were transformed into a larger temporal representation through delta features and rolling statistics.

### Dataset Availability

The original datasets are not included in this repository because their redistribution depends on the terms and licensing conditions of their respective sources.

**Sources:**

* PhysioNet — publicly available clinical datasets
* Kaggle — publicly available dataset resources

> Add the exact dataset names and source links here if you want visitors to access the specific datasets used in the project.

---

## Methodology

The proposed pipeline consists of four major stages:

```text
Raw ICU Data
     │
     ▼
Data Preprocessing
     │
     ├── Missing-value imputation
     ├── 15-hour window construction
     ├── Feature engineering
     └── Robust normalization
     │
     ▼
TimeGAN Synthetic Data Generation
     │
     ├── GRU-based generator
     ├── Temporal representation learning
     └── ClinicalConstraintLoss
     │
     ▼
Multi-Scale Temporal Attention (MSTA)
     │
     ├── BiGRU encoder
     ├── Short-range attention
     ├── Long-range attention
     └── Deep classification head
     │
     ▼
Ensemble Stacking
     │
     ├── XGBoost
     ├── LightGBM
     ├── Random Forest
     └── MSTA
     │
     ▼
Logistic Regression Meta-Learner
     │
     ├── Prediction
     ├── Uncertainty estimation
     └── Explainability
```

---

## Data Preprocessing

Each patient is represented using a **15-hour temporal window**.

For sepsis patients, the window consists of the hours immediately preceding the first confirmed sepsis onset. Rows corresponding to confirmed onset are excluded from the feature window to reduce label leakage.

The preprocessing pipeline includes:

1. Patient-level train/validation/test splitting
2. 15-hour temporal window construction
3. Removal of insufficiently long patient records
4. Forward and backward filling of missing observations
5. MICE-based imputation for remaining missing values
6. Temporal delta feature generation
7. Rolling mean and standard deviation features
8. Fixed-length sequence construction
9. RobustScaler normalization

The original clinical feature space was expanded from **40 base variables to 108 engineered features per timestep**.

---

## Synthetic Data Augmentation — TimeGAN

Sepsis cases represented a minority of the training population.

The initial training pool had a class imbalance of approximately:

```text
Sepsis : Non-Sepsis = 1 : 16.7
```

To preserve the temporal characteristics of the clinical data, a GRU-based **TimeGAN** architecture was used instead of conventional oversampling methods.

The TimeGAN pipeline consists of:

* Embedder
* Recovery network
* Generator
* Discriminator

A custom **ClinicalConstraintLoss** was incorporated to encourage generated sequences to follow clinically plausible physiological trends.

Synthetic sepsis sequences were generated until the training pool reached approximately:

```text
Sepsis : Non-Sepsis = 1 : 4
```

### TimeGAN Results

| Quality Check       |  Result | Status |
| ------------------- | ------: | ------ |
| TSTR AUROC          |   0.708 | PASS   |
| Synthetic HR trend  |  +0.022 | PASS   |
| Synthetic MAP trend |  −0.083 | PASS   |
| Final class ratio   | 1 : 4.0 | PASS   |

The final augmented training set contained:

* **38,176 total sequences**
* **7,635 sepsis sequences**
* **30,541 non-sepsis sequences**
* Sequence shape: **38,176 × 15 × 108**

---

## Multi-Scale Temporal Attention (MSTA)

The primary deep learning model is the proposed **Multi-Scale Temporal Attention (MSTA)** network.

MSTA combines a **Bidirectional GRU (BiGRU)** encoder with dual-scale temporal attention to capture both short-term physiological instability and longer-term deterioration patterns.

### Architecture

```text
108-dimensional temporal features
            │
            ▼
Linear + LayerNorm + ReLU
            │
            ▼
2-Layer Bidirectional GRU
            │
      ┌─────┴─────┐
      ▼           ▼
Short-Range   Long-Range
 Attention     Attention
      │           │
      └─────┬─────┘
            ▼
     Feature Fusion
            │
            ▼
Feed-Forward Classifier
            │
            ▼
      Sepsis Risk
      Probability
```

### Training Configuration

| Parameter              | Value                         |
| ---------------------- | ----------------------------- |
| Hidden dimension       | 128                           |
| Dropout                | 0.3                           |
| Optimizer              | Adam                          |
| Learning rate          | 0.001                         |
| Weight decay           | 0.0001                        |
| Loss                   | Weighted Binary Cross-Entropy |
| Batch size             | 256                           |
| Gradient clipping      | 1.0                           |
| Early stopping         | 12 epochs                     |
| Uncertainty estimation | MC Dropout                    |
| MC Dropout passes      | 50                            |

The MSTA model contained approximately **973,377 trainable parameters** and converged at epoch 23 using early stopping.

---

## Ensemble Learning

Three tree-based models were trained alongside the MSTA model:

* **XGBoost**
* **LightGBM**
* **Random Forest**

Their predictions were combined with the MSTA prediction using a **Logistic Regression meta-learner**.

The meta-learner was trained using held-out validation predictions to reduce the risk of training-set memorization.

### Ensemble Structure

```text
                    ┌── XGBoost
                    │
                    ├── LightGBM
Patient Features ───┼── Random Forest
                    │
                    └── MSTA
                         │
                         ▼
                Logistic Regression
                  Meta-Learner
                         │
                         ▼
                Final Sepsis Risk
```

---

## Model Performance

### MSTA Performance

The MSTA model achieved an external DS3 AUROC of:

**0.8805**

TimeGAN augmentation improved the MSTA AUROC by:

* **+0.0748** on DS2 validation
* **+0.0243** on DS2 test
* **+0.1466** on DS3 external evaluation

| Evaluation Set | Baseline AUROC | MSTA + TimeGAN |    Gain |
| -------------- | -------------: | -------------: | ------: |
| DS2 Validation |         0.8621 |         0.9369 | +0.0748 |
| DS2 Test       |         0.8946 |         0.9189 | +0.0243 |
| DS3 External   |         0.7339 |         0.8805 | +0.1466 |

---

## Ensemble Results

The stacked ensemble produced the strongest overall performance.

### Primary Held-Out Evaluation

The merged held-out evaluation set contained:

* **8,068 patient windows**
* **427 sepsis cases**
* **7,641 non-sepsis cases**
* Sepsis prevalence: **5.3%**

The ensemble achieved:

**AUROC: 0.9244**

**AUPRC: 0.7099**

| Evaluation Set  |      AUROC |      AUPRC |
| --------------- | ---------: | ---------: |
| DS2 Validation  |     0.9844 |     0.9233 |
| DS2 Test        |     0.9990 |     0.9818 |
| DS3 External    |     0.9229 |     0.7064 |
| Merged Held-Out | **0.9244** | **0.7099** |

The merged held-out AUROC 95% confidence interval was:

**0.9087 – 0.9382**

---

## Clinical Decision Metrics

On the merged held-out evaluation set, the model achieved:

| Metric      |  Value |
| ----------- | -----: |
| Sensitivity | 0.5831 |
| Specificity | 0.9923 |
| PPV         | 0.8084 |
| NPV         | 0.9771 |
| F1 Score    | 0.6776 |

At an operating point corresponding to **90% specificity**, sensitivity reached:

**0.7916**

These metrics demonstrate the model's ability to maintain high specificity while detecting a substantial proportion of sepsis cases.

---

## Early Warning Analysis

The model was evaluated using different amounts of available patient history.

The external DS3 evaluation demonstrated that the model was able to discriminate sepsis cases using limited early observations.

| Hours Available | Hours Before Onset |      AUROC |      AUPRC |
| --------------: | -----------------: | ---------: | ---------: |
|          1 hour |               T−14 |     0.8233 |     0.2945 |
|         3 hours |               T−12 |     0.8284 |     0.3440 |
|         6 hours |                T−9 |     0.8313 |     0.3306 |
|         9 hours |                T−6 |     0.8378 |     0.3983 |
|        12 hours |                T−3 |     0.8585 |     0.4908 |
|        15 hours |                T−0 | **0.8805** | **0.5628** |

The AUROC increased from **0.8233 using the first available hour** to **0.8805 using the complete 15-hour window**, demonstrating the value of accumulated temporal information.

---

## Explainable AI

Explainability was incorporated using **SHAP (SHapley Additive exPlanations)**.

SHAP TreeExplainer was applied to the XGBoost component of the ensemble to identify the clinical features contributing to predictions.

The analysis considered both:

* Mean-level physiological measurements
* Temporal variability features

Important features included clinically relevant variables such as:

* FiO2
* MAP
* Creatinine
* Hemoglobin
* WBC

These features span respiratory, cardiovascular, renal, haematological, and immune-related physiological domains.

---

## Clinical Alignment Score (CAS)

A **Clinical Alignment Score (CAS)** was introduced to quantitatively evaluate the alignment between model explanations and established clinical organ-dysfunction indicators.

CAS was calculated using the Jaccard similarity between the model's top-10 SHAP features and a predefined SOFA-related feature set.

### CAS Results

| Measure               | Result |
| --------------------- | -----: |
| Mean CAS              | 0.1059 |
| Standard deviation    | 0.0362 |
| Theoretical maximum   |  0.385 |
| Percentage of maximum |  27.5% |
| SOFA domains covered  | 5 of 6 |

CAS showed a monotonic increase across predicted-risk deciles, with the highest-risk predictions demonstrating stronger alignment with clinically relevant organ-dysfunction biomarkers.

---

## Counterfactual Analysis

Counterfactual analysis was performed to investigate how model predictions respond to changes in influential features.

High-confidence predictions generally required only one to two feature modifications to move below the decision threshold, while moderate-risk predictions required more changes.

This analysis provides additional insight into the features driving individual predictions and complements the global SHAP analysis.

---

## Uncertainty Estimation

Monte Carlo Dropout was used to estimate predictive uncertainty.

The MSTA model performed **50 stochastic forward passes** during inference.

The analysis showed higher uncertainty among sepsis-positive cases compared with non-sepsis cases:

| Group      | Mean Uncertainty |
| ---------- | ---------------: |
| Sepsis     |           0.0698 |
| Non-Sepsis |           0.0490 |

The mean reported 95% confidence interval width was approximately **0.0501**.

---

## Project Structure

```text
VitaXplain-Early-Sepsis-Detection/
│
├── README.md
│
└── model_training.ipynb
```

The `model_training.ipynb` notebook contains the model development and experimental workflow, including preprocessing, feature engineering, synthetic data generation, deep learning, ensemble modelling, evaluation, and explainability analysis.

---

## Technologies Used

* Python
* Google Colab
* PyTorch / Deep Learning Frameworks
* TensorFlow / Keras (if applicable)
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Matplotlib
* TimeGAN
* GRU / BiGRU
* Attention Mechanisms

> Remove any library above that was not actually used in the final implementation.

---

## Reproducibility

The main implementation is provided in:

`model_training.ipynb`

The notebook was developed and executed using **Google Colab**.

To reproduce the experiments:

1. Open the notebook in Google Colab.
2. Obtain the required datasets from their original PhysioNet/Kaggle sources.
3. Configure the dataset paths.
4. Install the required Python dependencies.
5. Run the preprocessing and feature engineering pipeline.
6. Train the TimeGAN augmentation model.
7. Train the MSTA and ensemble models.
8. Evaluate the models on the designated validation and held-out datasets.

Because the original clinical datasets are not distributed with this repository, users must obtain them separately from their respective sources.

---

## Research Contributions

The main contributions of this project include:

1. A **15-hour temporal framework** for early sepsis detection.
2. A **GRU-based TimeGAN augmentation approach** for addressing severe sepsis-class imbalance.
3. A **ClinicalConstraintLoss** designed to encourage physiologically plausible synthetic trajectories.
4. A proposed **Multi-Scale Temporal Attention (MSTA)** architecture for learning short- and long-term physiological patterns.
5. A heterogeneous **stacked ensemble** combining deep learning and tree-based models.
6. **Monte Carlo Dropout** for predictive uncertainty estimation.
7. SHAP-based model interpretability.
8. A **Clinical Alignment Score (CAS)** for quantifying alignment between model explanations and clinical SOFA-related features.
9. Evaluation using an external dataset from a different institutional source to investigate model generalisability.

---

## Future Work

Potential future improvements include:

* Evaluation on additional multi-centre ICU datasets
* Further optimisation of the temporal attention architecture
* Improved synthetic-data generation and validation
* Prospective clinical validation
* Real-time streaming prediction
* Model calibration and threshold optimisation for clinical workflows
* Deployment as a clinical decision-support prototype
* Additional fairness and subgroup performance analysis

---

## Author

**Noor Ul Huda**

Final Year Project — VitaXplain: Early Sepsis Detection

---

## Disclaimer

This repository presents an academic research prototype. The model has not been clinically validated for real-world patient care and should not be used as a substitute for professional medical judgement, diagnosis, or treatment.

