# 🚴 Bicycle Sales Forecasting and Datawarehouse Project

Welcome to the **Bicycle Sales Forecasting and Datawarehouse** repository! 🚀  
This project demonstrates a full-stack analytics solution combining data engineering, business intelligence, and machine learning. Developed as part of David Cruz Gómez Final Project in Big Data & Business Analytics, it showcases industry-grade practices in data warehousing, ETL processing, and predictive modeling.

---

## 📦 Project Overview

This end-to-end solution was designed to:

- Consolidate historical **bicycle sales data** with business, product, and employee information.
- Integrate **weather conditions** as external predictors of demand.
- Deliver **interactive dashboards** for business insights.
- Train and evaluate **machine learning models** to forecast future sales trends.

---

## 🏗️ Data Engineering & Architecture

### Objective  
Build a modern data platform using a **Medallion Architecture** (Bronze, Silver, Gold) that transforms raw sales and weather data into a clean, analytical data warehouse modeled in star schema.

### Pipeline Components

- **Bronze Layer:** Ingests raw sales and weather data from CSVs and NOAA files.
- **Silver Layer:** Cleanses, enriches, and joins datasets using SQL and Spark.
- **Gold Layer:** Structures data into facts and dimensions, optimized for BI tools.

### Tech Stack

- SQL Server (ETL & data warehousing)  
- Apache Spark (weather data processing)  
- Structured Star Schema design  
- Data Quality Checks in SQL & Python

---

## 📊 BI & Reporting

### Objective  
Enable business stakeholders to interactively explore key performance indicators (KPIs) through an intuitive dashboard.

### Features

- **Power BI dashboard** with slicers by country, product, employee, and date.
- **Geospatial heatmaps** for sales distribution.
- **Trend analysis** by seasonality and weather conditions.
- **Top product and employee performance** visualizations.

---

## 🤖 Machine Learning

### Objective  
Predict future bicycle sales based on historical demand, seasonality, and weather variables.

### Models Implemented

- Random Forest Regressor  
- XGBoost Regressor  
- LightGBM Regressor  
- Support Vector Regressor (SVR)

### Evaluation Metrics

- RMSE, MAE, R²  
- Cross-validation and grid search optimization  
- Feature importance analysis (e.g., temperature, precipitation, weekend)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, adapt, and share this project with proper credit.
