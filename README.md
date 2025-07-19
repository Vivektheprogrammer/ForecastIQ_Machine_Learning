# Machine Learning Models for Rainfall and Temperature Forecasting in India

---

## Overview

This repository presents a comparative study and implementation of various machine learning algorithms for accurate rainfall and temperature forecasting across diverse regions of India. Leveraging a comprehensive dataset from the Indian Weather Repository (2023) that integrates meteorological and air quality parameters, this project aims to enhance predictive capabilities for critical environmental variables. The models developed contribute to improved climate resilience, agricultural planning, and disaster risk mitigation.

---

## Project Goal

The primary goal of this research is to:
* Evaluate the performance of five prominent machine learning algorithms—Random Forest, XGBoost, LightGBM, Extra Trees, and Support Vector Machine (SVM)—for environmental prediction.
* Accurately forecast rainfall event occurrence (classification), precipitation value estimation (regression), and temperature value estimation and classification (regression and classification).
* Demonstrate the enhanced predictive capability achieved by incorporating air quality indicators into the models.

---

## Dataset

The dataset used in this research is sourced from the **Indian Weather Repository (2023)** and combines data from the Central Pollution Control Board (CPCB) and meteorological institutes. It comprises 42 attributes for various locations across India, including:

* **Meteorological Data:** Temperature ($^{\circ}$C and $^{\circ}$F), humidity (%), wind speed (km/h), atmospheric pressure (hPa), precipitation (mm), cloud cover, visibility.
* **Air Quality Parameters:** Carbon Monoxide (CO), Ozone ($O_3$), Nitrogen Dioxide ($NO_2$), Sulfur Dioxide ($SO_2$), Particulate Matter ($PM_{2.5}$ and $PM_{10}$), US EPA and GB DEFRA air quality indices.
* **Geographic Identifiers:** Latitude, longitude, region, time zone.
* **Astronomical Data:** Sunrise, sunset, moon phases, and illumination.

This rich and diverse dataset captures a wide range of weather conditions and environmental factors, making it ideal for robust predictive modeling.

---

## Methodology

### Data Preprocessing
1.  **Logarithmic Transformation:** A shifted logarithmic transformation was applied to raw precipitation values to address skewness and normalize the distribution, making it more suitable for regression models.
    $$
    \text{log\_precip\_mm} = \log(\text{precipitation\_mm} + 1)
    $$
2.  **Target Variable Encoding:** Continuous rainfall values were converted into a binary label (`rain_occurred`) for classification tasks, simplifying the prediction of whether it will rain or not.
    * `rain_occurred = 1 if precipitation_mm > 0 else 0`
3.  **Correlation Analysis:** Pearson's correlation coefficient was used to identify relationships between environmental features and target variables (`log_precip_mm`, `temperature_celsius`, `rain_occurred`). Key correlations identified include:
    * `cloud_cover` and `humidity` showed strong positive correlation with rainfall.
    * `air_quality_PM10` and `humidity` showed moderate negative correlation.

### Machine Learning Models Implemented

Five machine learning algorithms were utilized for both regression and classification tasks:

1.  **Random Forest:** An ensemble learning algorithm that builds multiple decision trees and combines their predictions.
    * **Performance:** Low MSE for rainfall and temperature regression, with 91.85% accuracy for rain occurrence and 91.85% accuracy for temperature classification.
2.  **XGBoost (Extreme Gradient Boosting):** A highly efficient and scalable implementation of gradient boosting.
    * **Performance:** Low MSE for rainfall and temperature regression, strong accuracy (92.08%) for rain occurrence, and outstanding accuracy (97.72%) for temperature classification.
3.  **LightGBM:** A gradient boosting framework that uses tree-based learning algorithms, known for speed and memory efficiency.
    * **Performance:** Low MSE for rainfall and temperature regression, robust accuracy (92.38%) for rain occurrence, and high accuracy (97.64%) for temperature classification.
4.  **Extra Trees (Extremely Randomized Trees):** An ensemble method that builds randomized decision trees.
    * **Performance:** Achieved the **lowest MSE** for both rainfall (3.60) and temperature (0.027) regression. High accuracy (92.99%) for rain occurrence and 96.19% accuracy for temperature classification.
5.  **Support Vector Machine (SVM):** A supervised machine learning model used for both classification and regression tasks by finding a hyperplane that best separates data points.
    * **Performance:** Moderate MSE for rainfall and temperature regression. Achieved 87.28% accuracy for rain occurrence (with high recall but lower precision) and 92.61% for temperature classification.

---

## Evaluation Metrics

Model performance was rigorously evaluated using a set of standard metrics:

* **Regression Tasks (Rainfall Amount & Temperature Value):**
    * **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values; lower MSE indicates higher accuracy.
* **Classification Tasks (Rain Occurrence & Temperature Categories):**
    * **Accuracy:** Overall proportion of correctly predicted instances.
    * **Precision:** Proportion of true positive predictions among all positive predictions (important when false positives are costly).
    * **Recall:** Proportion of actual positives correctly identified (important when missing positive cases is costly).
    * **F1-Score:** Harmonic mean of precision and recall, balancing both types of errors, especially useful for imbalanced datasets.
    * **Confusion Matrix:** Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

---

## Results Highlights

The study found that **ensemble tree-based models (Extra Trees, XGBoost, and LightGBM) consistently outperformed SVM** across all regression and classification tasks.

| Model                 | Rainfall Regression MSE | Temperature Regression MSE | Rain Occurrence Accuracy | Temperature Classification Accuracy |
| :-------------------- | :---------------------- | :------------------------- | :----------------------- | :---------------------------------- |
| Random Forest         | 3.95                    | 0.054                      | 91.85%                   | 91.85%                              |
| XGBoost               | 3.93                    | 0.050                      | 92.08%                   | **97.72%** |
| LightGBM              | 4.10                    | 0.051                      | 92.38%                   | 97.64%                              |
| **Extra Trees** | **3.60** | **0.027** | **92.99%** | 96.19%                              |
| SVM                   | 4.38                    | 0.35                       | 87.28%                   | 92.61%                              |

* **Extra Trees** demonstrated superior performance in **rainfall and temperature regression**.
* **XGBoost** and **LightGBM** delivered the highest accuracies in **temperature classification**.
* The inclusion of **air quality indicators significantly enhanced** the predictive capability across all models.

---

## Conclusion

This research underscores the immense potential of machine learning, particularly ensemble methods, in environmental prediction and operational weather forecasting. The successful integration of meteorological and air quality data provides a robust framework for creating highly accurate and scalable data-driven weather forecasting systems. These findings offer valuable insights for real-time climate monitoring, agricultural planning, and public health management, especially in climatically diverse regions like India.

---
