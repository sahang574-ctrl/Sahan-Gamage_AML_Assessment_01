# SmartFactory: Predictive Maintenance & Casting Defect Detection

A machine learning project exploring two approaches to smart manufacturing quality monitoring: **sensor-based machine failure prediction** and **image-based casting defect detection**. The project compares classical machine learning, feed-forward neural networks, and convolutional neural networks (CNNs), with attention to reproducibility, class imbalance, evaluation, and practical deployment trade-offs.

## Project Overview

| Part | Task | Data | Models |
|---|---|---|---|
| **A — Sensor Classification** | Predict machine failure from manufacturing measurements | UCI AI4I 2020 Predictive Maintenance (10,000 synthetic observations) | Random Forest, feed-forward MLP |
| **B — Casting Inspection** | Classify casting photographs as **OK** or **defective** | Casting Product Image Data (1,300 original images: 519 OK, 781 defective) | Baseline CNN, revised CNN |
| **C — Comparison** | Compare model performance, computational requirements, interpretability, and deployment considerations | Outputs from Parts A and B | Cross-method evaluation and recommendations |

## Key Features

- Exploratory data analysis and visualizations
- Training-only preprocessing, with leakage controls for sensor data
- Separate training, validation, and held-out test partitions
- Random Forest hyperparameter selection and feature importance analysis
- PyTorch feed-forward neural network with weighted binary cross-entropy and early stopping
- Two CNN architectures for image-based casting inspection
- Validation-based decision thresholds emphasizing recall using the F2 score
- Evaluation using metrics such as precision, recall, F1/F2, average precision, ROC-AUC, and confusion matrices
- CNN training curves, prediction/error examples, and Grad-CAM visual explanations
- Saved evaluation artifacts and trained model weights
- Comparison of accuracy-related metrics, model complexity, training cost, and deployment trade-offs

## Repository Structure

```text
SmartFactory/
├── 01_SmartFactory_Sensor_Models(5).ipynb
├── 02_SmartFactory_Casting_CNN(3).ipynb
├── data/
│   ├── ai4i2020.csv
│   └── casting/
│       └── ... JPEG images in their original class folders
├── outputs/                 # Generated when notebooks run
│   ├── part_a/
│   └── part_b/
├── models/                  # Saved model artifacts
└── README.md
```

> The notebooks expect a `data/` directory in the current working directory. Dataset files, outputs, and model weights may need to be created or downloaded separately; the structure above shows the expected layout, not necessarily files included in this repository.

## Datasets

**Part A — UCI AI4I 2020 Predictive Maintenance**  
[Dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)  
Place the CSV at `data/ai4i2020.csv`. The target is `Machine failure` (0 = no failure; 1 = failure). Identifier fields and failure-mode flags are excluded from model inputs to reduce leakage. The dataset is synthetic, so results should not be interpreted as validated real-factory performance.

**Part B — Casting Product Image Data for Quality Inspection**  
[Dataset page](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)  
Place the **original 512×512 JPEG images** under `data/casting/`, retaining their class folders (`ok_front` and `def_front`). The notebook expects exactly **1,300** original images. Target labels are 0 = OK and 1 = defective. Check the source's **CC BY-NC-ND 4.0** licensing conditions before redistributing images.

## Tools & Technologies

Python, Jupyter Notebook, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, PyTorch, Pillow, and joblib.

## How to Run

1. Clone or download this repository and open a terminal in its root folder.
2. Install the required packages:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn torch pillow joblib jupyter
   ```

3. Download both datasets from the links above and arrange them under `data/` as described.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Run **`01_SmartFactory_Sensor_Models(5).ipynb`** first. This creates Part A outputs used in the final comparison.
6. Run **`02_SmartFactory_Casting_CNN(3).ipynb`** second. It trains and evaluates the CNNs and includes **Part C**, which reads Part A results.

The notebooks use a fixed random seed (`42`) and select CUDA when available; otherwise they use CPU. Training times depend on your hardware.

## Evaluation Approach

Models are selected and thresholds are tuned using **validation data**, with final evaluation on a **held-out test set**. The notebooks generate metrics tables, plots, and model artifacts under `outputs/` and `models/`. Consult the executed notebooks and generated CSV files for the actual numerical results; no performance numbers are claimed here without those outputs.

## Limitations

- The sensor dataset is **synthetic** and predicts machine failure, not a directly observed casting defect.
- Sensor records and casting photographs come from **different, unpaired datasets**; this is a comparison of approaches, **not a fused multimodal system**.
- Image splits cannot establish generalization to unseen production batches, cameras, lighting conditions, or factories.
- Grad-CAM highlights are explanatory aids, not precise defect segmentation.
- Real-world deployment would require additional validation on representative production data.

## Data Attribution

- UCI Machine Learning Repository: *AI4I 2020 Predictive Maintenance Dataset* (CC BY 4.0).
- Ravirajsinh Dabhi: *Casting Product Image Data for Quality Inspection*, with photographs supported by PILOT TECHNOCAST, Rajkot (CC BY-NC-ND 4.0).

## Project Purpose

An educational exploration of how **tabular sensor models** and **computer vision models** can support predictive maintenance and quality inspection in smart manufacturing.
