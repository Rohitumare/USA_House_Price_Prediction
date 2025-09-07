# USA_House_Price_Prediction

A beginner-friendly data analytics project that predicts USA house prices using Linear Regression.

## Project Overview  
This project performs an end-to-end Machine Learning workflow using a Linear Regression model to predict housing prices in the USA. A fantastic starter for learners exploring regression analysis in Python.


## Repository Contents  
- `USA_House_Price_Prediction.ipynb` — Jupyter Notebook walking through the full data pipeline and modeling process.  
- `USA_Housing.csv` — Original dataset containing features related to house pricing.  
- `Predicting_house_price_using_linear_regression_model.ipynb` — Jupyter Notebook showing how the model is used.  
- `USA house price pred model with 100k rmse on 25-08-25.pkl` — Saved/trained model with a root mean square error around 100k (as of August 25, 2025).  
- `ml_stats.py` — A self made python library using OOPs to automate descriptive statistics process.


## Dataset  
This dataset includes attributes like average area income, house age, number of rooms, bedrooms, area population, and the target variable house "price" for regression analysis. Linear Regression model has been trained using these features.


## Project Workflow  

### 1. Exploratory Data Analysis (EDA)  
- Load the dataset and examine basic statistics.  
- Visualize feature distributions and relationships with the target variable (e.g., scatter plots, histograms, correlation matrix).

### 2. Data Cleaning & Preprocessing  
- Check for missing or outlier values and apply necessary cleaning or imputation.  
- Ensure that all features are in numerical format and appropriate scale.

### 3. Feature Engineering  
- If needed, create new features—like per-room pricing or area-adjusted metrics.  
- Standardize or normalize input features if required by the model.

### 4. Model Training & Evaluation  
- Split data into training and testing sets.  
- Train a Linear Regression model on the training portion.  
- Evaluate performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).


## Results & Insights
Model performance: RMSE of approximately $100,000 (as of August 25, 2025).

### Findings:
- Higher income areas and larger house sizes tend to associate with increased house prices.
- Model may struggle with high-variance regions or outlier communities—worth exploring residuals.


## Conclusion
This project demonstrates a complete ML pipeline using Linear Regression to forecast USA house prices. Ideal for learners building an understanding of regression-based modeling techniques.


## Technologies Used
- Python (Pandas, NumPy, scikit-learn, Matplotlib/Seaborn)
- Jupyter Notebook
- Pickle for model serialization
- Custom statistics library
