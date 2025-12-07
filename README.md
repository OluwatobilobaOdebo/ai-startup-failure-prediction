# AI Startup Failure Prediction

This project builds an end-to-end machine learning pipeline to predict the probability that a startup will fail (close) or succeed (acquire) using historical startup funding, investment, milestone, and relationship data.

It includes:
- A full EDA + modeling Jupyter notebook  
- A production-ready Random Forest model  
- An interactive Streamlit dashboard  
- Clean project structure with datasets, models, and processed features  
- Deployment-ready setup for Streamlit Cloud

---

## Project Overview

The goal of this project is to help VCs, angel investors, accelerators, and founders understand which startups are likely to fail based on historical patterns.

The system:
- Cleans and processes raw startup data  
- Trains a Random Forest classifier  
- Generates predicted failure probabilities  
- Buckets startups into Low / Medium / High risk segments  
- Surfaces insights through an interactive Streamlit dashboard  
- Allows portfolio and individual startup analysis

This project showcases applied machine learning, data storytelling, analytics engineering, and interactive deployment.

---

## Features

### Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing  
- Missing value handling  
- Correlation and distribution analysis  
- Key feature exploration

### Machine Learning Model
- Random Forest classifier  
- Train/test evaluation  
- Probability outputs  
- Model metrics exported for transparency

### Streamlit Dashboard
- Portfolio-level KPIs  
- Risk distribution histogram  
- Funding vs predicted risk scatterplot  
- Filters for state/region and risk bucket  
- Startup-level risk page  
- Underlying filtered dataset table

---

## Dataset Description

### **Source**  
**Startup Success Prediction Dataset** 
https://www.kaggle.com/datasets/manishkc06/startup-success-prediction

| Feature | Description |
|--------|-------------|
| `state_code` | US state/region |
| `funding_rounds` | Total funding rounds |
| `funding_total_usd` | Total capital raised |
| `milestones` | Milestones achieved |
| `relationships` | Investor/advisor relationships |
| `is_top500` | Top 500 startup indicator |
| `target_failure` | 1 = Failed, 0 = Acquired |
| `pred_failure_prob_rf` | Model-predicted failure probability |
| `risk_bucket_rf` | Low / Medium / High risk |
