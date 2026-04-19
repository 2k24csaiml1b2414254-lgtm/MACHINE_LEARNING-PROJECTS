Customer Segmentation Analysis

## Overview
This notebook performs customer segmentation using clustering techniques on the SmartCart customers dataset.

## Dataset
- **Source**: `smartcart_customers.csv`
- Contains customer demographic and behavioral data

## Steps Performed

### 1. Data Preprocessing
- Handled missing values in the Income column (filled with median)
- Created new features:
    - **Age**: Calculated from Year_Birth
    - **Customer_Tenure_Days**: Days since customer joined
    - **Total_Spending**: Sum of spending across product categories
    - **Total_Children**: Sum of kids and teens at home
- Encoded Education and Marital_Status

### 2. Outlier Removal
- Removed customers with Age > 90
- Removed customers with Income > 600,000

### 3. Feature Engineering
- Applied One-Hot Encoding to categorical variables
- Standardized features using StandardScaler

### 4. Dimensionality Reduction
- Applied PCA to reduce features to 3 principal components

### 5. Clustering Analysis
- Used **Elbow Method** and **Silhouette Score** to determine optimal K
- Applied **K-Means** and **Agglomerative Clustering** with K=4

### 6. Cluster Characterization
- Analyzed cluster patterns based on Income and Spending
- Generated cluster summary statistics

## Results
- Identified 4 distinct customer segments
- Visualized clusters in 3D PCA space
- Generated cluster profiles for marketing insights
