# Airline-Clustering-Analysis-using-K-Means-DBSCAN-Hierarchical-Clustering
Clustering analysis on airline passenger data using K-Means, DBSCAN, and Agglomerative algorithms with EDA, outlier handling, and performance evaluation using silhouette score.
# ✈️ Airline Clustering Analysis

## 📌 Overview
This project performs clustering on airline passenger data to identify patterns and group airlines based on flight frequency and passenger count.

The goal is to compare different clustering algorithms and evaluate their performance.

---

## 📊 Dataset
- Airline Passenger Traffic Data
- Features used:
  - Operating Airline
  - Passenger Count
  - Flights held (frequency)

---

## 🔍 Steps Performed

### 1. Data Collection & Storage
- Loaded dataset using Pandas
- Stored data into MySQL database using SQLAlchemy

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics
- Missing values & duplicates check
- Sweetviz report generation

### 3. Feature Engineering
- Aggregated:
  - Number of flights per airline
  - Total passenger count

### 4. Outlier Detection
- Scatter plot visualization
- Removed extreme outliers (e.g., United Airlines)

### 5. Clustering Algorithms Applied
- K-Means Clustering
- Agglomerative (Hierarchical) Clustering
- DBSCAN

### 6. Model Evaluation
- Silhouette Score used for performance comparison

### 7. Visualization
- Cluster plots for all algorithms

---

## 📈 Results
- K-Means and Agglomerative produced clear clusters
- DBSCAN required parameter tuning (eps, min_samples)
- Outlier removal significantly improved clustering performance

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- Sweetviz
- SQLAlchemy (MySQL)

---

## 📦 Model Saving
- DBSCAN model saved using Pickle

---

## 🚀 Key Learnings
- Importance of outlier handling in clustering
- Difference between density-based and centroid-based algorithms
- Real-world application of unsupervised learning


## 📌 Conclusion
This project demonstrates how different clustering algorithms behave on real-world data and highlights the importance of preprocessing and parameter tuning.
