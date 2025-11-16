# electricity-consumption-ml-project
Supervised Machine Learning project predicting household electricity consumption and monthly bill using regression models. Includes dataset (150 rows), preprocessing, model training, evaluation, and Streamlit dashboard.

Here’s a **short and clean blog version** for your ML project:

---

#  Electricity Consumption Prediction Using Machine Learning

*By Alan Alex*

In this project, I built a Machine Learning model that predicts **monthly electricity consumption (kWh)** based on factors like number of appliances, AC usage, family size, temperature, and daily usage duration. The aim was to understand how different variables affect electricity usage and to create a model that can assist homes in estimating their monthly power consumption.

---

## Project Workflow


DATA COLLECTION → PREPROCESSING → TRAIN/TEST SPLIT → MODEL TRAINING
                                                          
-> EXPLORATION → FEATURE SELECTION → MODEL TESTING → VISUALIZATION → RESULTS

**Models Used**

I trained and compared two regression models:

* **Linear Regression** – a simple baseline model
* **Random Forest Regressor** – a more advanced ensemble model

Random Forest performed better because electricity consumption is influenced by non-linear patterns.

 **Results**

Both models were evaluated using MAE, RMSE, and R² Score.
Random Forest showed:

* Lower error
* Higher accuracy
* Better fit to actual consumption values

Graphs comparing **Actual vs Predicted** values clearly showed Random Forest giving more reliable predictions.


 **Conclusion**

This project demonstrates a complete ML pipeline—from data cleaning to model evaluation. It also shows how machine learning can help in **smart energy management**, utility prediction, and home automation systems.
