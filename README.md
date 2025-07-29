# Customer Churn Prediction Project: Driving Retention with Data Insights

This project aims to build a robust machine learning model to predict customer churn in a telecommunications company. By identifying customers at high risk of churning, the project provides actionable insights that enable the company to implement proactive retention strategies, reduce customer attrition, and ultimately enhance profitability.

## Project Objectives

* Perform comprehensive data preprocessing and exploratory data analysis to understand churn drivers.
* Develop and optimize a predictive model for churn classification.
* Interpret model results to derive key business insights and recommendations.

## Data Source & Tools Used

* **Data Source:** The dataset used is the Telco Customer Churn Dataset. It is included directly in this GitHub repository (`Telco-Customer-Churn.csv`) and contains various customer attributes, service subscriptions, contract details, billing information, and churn status.
* **Tools Used:**
    * **Python:** For all data manipulation, analysis, and machine learning tasks.
    * **Pandas & NumPy:** Essential libraries for efficient data handling and numerical operations.
    * **Matplotlib & Seaborn:** For creating static and statistical data visualizations.
    * **Scikit-learn:** For data splitting, feature scaling, model selection, and evaluation metrics.
    * **LightGBM:** A high-performance gradient boosting framework used for building the predictive model.
    * **Google Colab:** The cloud-based development environment for executing the Python code.

## Key Preprocessing Steps

Before analysis and modeling, the raw data underwent several crucial preprocessing steps to ensure data quality and suitability:

* **Data Loading:** The `Telco-Customer-Churn.csv` file was loaded into a Pandas DataFrame.
* **Handling Missing Values:** Empty strings in the `TotalCharges` column were converted to `NaN` (Not a Number), and rows containing these missing values were subsequently dropped.
* **Target Variable Transformation:** The `Churn` column, initially categorical ('Yes'/'No'), was converted into a numerical binary format (1 for 'Yes', 0 for 'No') for machine learning compatibility.
* **Dropping Unnecessary Columns:** The `customerID` column was removed as it serves merely as a unique identifier and holds no predictive power for the model.
* **Categorical Feature Encoding:** All remaining categorical features (object data type) were transformed into a numerical format with Binary Encoding for two-level columns and One-Hot Encoding for multi-level columns.

## Exploratory Data Analysis (EDA)

EDA involved examining variable distributions and their relationships with customer churn through various visualizations to gain deeper insights into data patterns. Below are some of the most insightful plots; all remaining plots can be found in the [plots folder](./plots) of this repository.

### Numerical Feature Analysis

* **Tenure Distribution and Churn:** The distribution of tenure shows a bimodal shape, with peaks at very low and very high tenures. The box plot tenure by Churn clearly indicates that customers with shorter tenures (new customers) have a significantly higher churn rate compared to long-term customers.
    ![Tenure Distribution and Churn](https://github.com/ZahraKheyrandish/Customer-Churn-Prediction-BI/blob/main/plots/download%20(4).png?raw=true)

* **MonthlyCharges Distribution and Churn:** MonthlyCharges shows a varied distribution. The MonthlyCharges by Churn box plot suggests that customers with higher monthly charges tend to churn more often.
    ![MonthlyCharges Distribution and Churn](https://github.com/ZahraKheyrandish/Customer-Churn-Prediction-BI/blob/main/plots/download%20(5).png?raw=true)

* **TotalCharges Distribution and Churn:** TotalCharges is skewed towards lower values. The TotalCharges by Churn box plot reveals that customers with lower total charges (often correlated with shorter tenure) are more prone to churn.
    ![TotalCharges Distribution and Churn](https://github.com/ZahraKheyrandish/Customer-Churn-Prediction-BI/blob/main/plots/download%20(6).png?raw=true)

### Key Categorical Feature Analysis and Overall Data Relationship

* **Churn Rate by gender:** There is no significant difference in churn rates between genders.
    ![Churn Rate by gender](https://github.com/ZahraKheyrandish/Customer-Churn-Prediction-BI/blob/main/plots/download%20(33).png?raw=true)

* **Correlation Matrix of Features:** This heatmap provides a quick overview of how features relate to each other and to the Churn target.
    ![Correlation Matrix of Features](https://github.P_USERNAME/YOUR_REPOSITORY/blob/main/plots/plt.jpg?raw=true)
For a complete overview of all generated visualizations and their detailed analysis, please refer to the [plots folder](./plots).
[Link to Google Colab Notebook with full EDA (if available)]

## Data Preparation for Modeling

* **Data Splitting:** The dataset was divided into training (80%) and testing (20%) sets. `stratify=y` was used to maintain the proportion of the Churn class in both sets.
* **Feature Scaling:** Numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) were scaled using `StandardScaler` to ensure optimal performance for machine learning models.

## Model Building and Training

The LightGBM (`LGBMClassifier`) model was chosen for churn prediction.

* **Class Imbalance Handling:** To address the imbalance between churned and non-churned customer classes, the `scale_pos_weight` parameter was calculated and applied to the LightGBM model.
* **Hyperparameter Tuning:** `RandomizedSearchCV` was employed to find the best combination of model parameters. This method efficiently samples random combinations from the parameter space to identify optimal settings based on the `roc_auc` scoring metric.

**Best Parameters Found:** 
{'learning_rate': 0.010078056549297373, 'max_depth': -1, 'n_estimators': 332, 'num_leaves': 22}
**Best ROC-AUC score:** `0.84619`
This ROC-AUC score from cross-validation on the training set indicates a very strong discriminative ability of the model.

## Model Evaluation

The final model (LightGBM with optimized parameters) was evaluated on the test set.

**Evaluation Metrics:**

| Metric    | Value  |
| :-------- | :----- |
| Accuracy  | 0.7960 |
| Precision | 0.6465 |
| Recall    | 0.5134 |
| F1-Score  | 0.5723 |
| AUC-ROC   | 0.8411 |

* **Confusion Matrix:** Provides a visual breakdown of the model's correct and incorrect classifications.
    * True Negatives (Correctly predicted non-churn): 928
    * False Positives (Incorrectly predicted churn): 105
    * False Negatives (Incorrectly predicted non-churn): 182
    * True Positives (Correctly predicted churn): 192
    ![Confusion Matrix](https://github.com/ZahraKheyrandish/Customer-Churn-Prediction-BI/blob/main/plots/download%20(35).png?raw=true)

* **Receiver Operating Characteristic (ROC) Curve and AUC:** Used to evaluate the model's ability to distinguish between classes.
    ![Receiver Operating Characteristic (ROC) Curve](https://github.com/ZahraKheyrandish/Customer-Churn-Prediction-BI/blob/main/plots/download%20(36).png?raw=true)
    * **AUC-ROC Interpretation:** An AUC-ROC value of 0.8411 signifies a very strong performance in distinguishing between churned and non-churned customers. This means the model can rank customers by their churn probability with high accuracy.

## Feature Importance Analysis

To understand which factors contribute most to churn prediction, feature importances were extracted from the LightGBM model.

**Top 10 Most Important Features:**

| Feature | Importance |
| :-------- | :--------- |
| tenure | 1455 |
| MonthlyCharges | 1440 |
| TotalCharges | 1008 |
| PaymentMethod_Electronic check | 304 |
| OnlineSecurity_Yes | 249 |
| Contract_Two year | 195 |
| Contract_One year | 188 |
| PaperlessBilling | 186 |
| InternetService_No | 184 |
| TechSupport_Yes | 179 |

![Top 10 Most Important Features for Churn Prediction](https://github.com/ZahraKheyrandish/Customer-Churn-Prediction-BI/blob/main/plots/download%20(37).png?raw=true)

**Key Insights:**

* **`tenure` (customer duration), `MonthlyCharges` (monthly fees), and `TotalCharges` (total fees):** These are the most critical factors in predicting churn. This suggests that newer customers and those with higher monthly charges are more prone to churn.
* **`PaymentMethod_Electronic check`:** This is a significant indicator of churn; customers using this payment method tend to churn more.
* **`OnlineSecurity_Yes` and `TechSupport_Yes`:** Customers subscribing to these value-added services are less likely to churn, highlighting the importance of these services for retention.
* **`Contract_Two year` and `Contract_One year`:** Long-term contracts are strongly associated with reduced churn, emphasizing the value of contract renewal strategies.
* **`PaperlessBilling`:** Customers with paperless billing enabled show a noticeably higher churn rate.
* **`InternetService_No`:** The absence of internet service generally correlates with very low churn (which might indicate a different segment of customers with different needs).

## Conclusion and Business Recommendations

This project delivers a robust model for customer churn prediction, demonstrating excellent discriminative ability with an AUC-ROC above 0.84. The insights derived from feature importance analysis provide actionable guidance for the business:

* **Focus on New Customers:** Given the high importance of `tenure`, customer retention programs should target customers in their initial months of service.
* **Review Payment Methods:** The customer service team should identify customers using "Electronic check" payment and investigate potential issues or offer alternative payment options.
* **Promote Long-Term Contracts and Value-Added Services:** Encouraging customers to sign one or two-year contracts and subscribe to OnlineSecurity and TechSupport services can significantly reduce churn rates.
* **Address High Monthly Charges:** Customers with `MonthlyCharges` high may require special attention to prevent dissatisfaction.
* **Investigate Paperless Billing Link:** A deeper dive into why paperless billing is associated with higher churn could yield further insights.

This model can help the company more effectively allocate resources to retain at-risk customers, ultimately increasing profitability.

## How to Run

To run this project:

1.  Clone this GitHub repository: `git clone https://github.com/ZahraKheyrandish/Customer-Churn-Prediction-BI.git`
2.  Navigate to the project directory.
3.  The dataset `Telco-Customer-Churn.csv` is already included in the repository.
4.  Open the Python notebook (e.g., in Google Colab or Jupyter Notebook) that contains the project's code.
    * If using Google Colab, you might need to mount your Google Drive if the notebook expects files from there, or ensure the data loading path is adjusted to directly read the CSV from the cloned repository.
5.  Run all cells in the notebook sequentially
