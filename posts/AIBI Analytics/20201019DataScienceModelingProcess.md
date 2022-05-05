---
title: "The Data Science ML Modeling Process"
date: "2020-10-19"
author: Alberto Gutierrez
description: Data Science Process, DSM, Data Science Modeling Process, Machine learning process, Exploratory Data Analysis, Normalization and Standardization, Model Deployment
...
<span style="display:block; color:blue; margin-top:-40px;"> </span>
[about me](../../about.md)  &nbsp;   &nbsp;  &nbsp;  &nbsp;   &nbsp;   &nbsp;  &nbsp;  &nbsp; [home](../../index.md)

<script type="text/javascript" charset="utf-8"
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML,
https://vincenttam.github.io/javascripts/MathJaxLocal.js"></script>

<h1 style="color:	#115BDC;">The Data Science ML Modeling Process - the key phases in developing, deploying, and managing a machine learning model </h1>

<figure>
 <img alt="Data Science Modeling Process" title="Data Science Modeling Process" src="/images/DataScienceModelingProcess/Data Science Modeling Process.png" width="700">
 </figure>

published October 19, 2020  
last updated January 17, 2021

# Introduction

While developing machine learning models, there is a set of best practices to deliver consistent and effective results. To begin with, one essential, yet often overlooked, best practice is to ensure there is a sufficiently predictive data set. Without such a dataset it is not possible to develop an effective model. It is useful for the best practices to address the entire ML model life-cycle. However, It seems there is no one definitive reference of best-practices for the ML model life-cycle. Several authors have published articles and books outlining methods for developing machine learning models. For example, few good sources are listed in the references. $^{1,2,3}$  Informed by these best practices, a data science modeling process for the entire ML model life-cycle is depicted in the figure above.

Another reason for documenting the data science ML modeling process is to facilitate communication and planning with non-data scientists. Though data scientists understand modeling best practices, the process is not well understood outside this small community. Project planning with other business functions significantly benefits from an understanding of the process. The overall project planning is improved by understanding the need for establishing clear data science oriented technical objectives, the role of exploratory data analysis, how a model is put into production, and managing and updating a model once it is in production. Understanding the modeling phases and methodology also helps non-data scientists appreciate how a model works and sets realistic expectations for what it takes to develop and manage ML models over their life-time.

Though in the diagram the phases appear to flow from start to finish, there are usually many iterations within and between the stages as insights data understanding is developed. Also, a project can focus on subsets of the life-cycle. Depending on the project specifics often  part of the life-cycle will fall within the project scope. For example, a discovery project may focus primarily in the data discovery, and insights generation. After initial data discovery, then BI insights and dashboards can be created. Additionally, a prototype model can go into the ML Operations phase moving from prototype to full-scale production with real customer data and management of the model in production. 

# Key Phases in Developing, Deploying, and Managing a Machine Learning Model

### Objective  
The first step is to delineate the business and technical objectives. The business objective is rarely to develop a predictive model. For example, the business objective might be to optimize product pricing and thereby increase profits. The technical goals are usually one part of broader business goals. For example, subordinate to optimizing product pricing, the technical goal will include an ML pricing model. Understanding the business goals helps define technical objectives, how the model will be used, and provides project guidance. A key output of this step is to identify the KPIs, measures and the targets that will define success of the project.

### Data Wrangling
The next step is to gather the data needed for developing the model. This step requires some domain expertise and collaboration between the data scientist and business domain expert to understand what types of data are useful or required to generate a successful model. Gathering the data will require some initial data manipulation, software coding for downloading, processing, and API (Application Programming Interface) access of data sources. At this and the next stage, it is essential to establish the requirements for necessary data. Too often, machine learning models are based on insufficient data, and thereby after very significant effort, it may ultimately lead to costly failure.

### Exploratory Data Analysis (EDA)
This step requires a systematic investigation of each variable to understand the statistics, relationship to other variables, the variable's usefulness in a predictive model, and the data's sufficiency to achieve the objectives. Many insights are usually discovered in this stage. Based on the data scientist's expertise, a hypothesis about the kind of model and data is established.

Data variables are transformed into a format that can be used by a machine. This process is known as feature extraction of the variables that the ML algorithm will use for prediction. Feature extraction requires de-trending, creating aggregations, statistical representations, and combining variables in a way that the machine can generalize and better predict the target (dependent variable). This step will also assess how the machine learning features are correlated to the target variable and identify multicollinearity between the variables.

A powerful way to communicate the data value during the EDA phase is to visualize data in dashboards. Dashboards include, for example, real-time visualization of actionable metrics for a product or service or BI dashboards for business processes, including sales, quality, and supply chain. Often a dashboard developed during EDA drives constructive communication between data scientists and business domain professionals. It is not uncommon for a dashboard created at this stage to morph into a production data product.

### Data Pre-processing
Once we have a set of machine learning features (independent variables), they will require preparation for ML. Preparing ML features requires standardization and normalization, for example, normalizing the features by putting the numerical features on the same scale. Standardization is transforming the variables to a standard deviation of one and a mean of zero. The pre-processing will depend on the variable (categorical or numerical) and statistics of the variable. This step will ensure, for example, that one feature with large numeric values does not overwhelm the predictive information of other variables with small numeric values. Depending on the type of model, dummy variable representations will be needed for categorical variables.

As data is updated, a data pipeline is often required to take raw data and process it so that it is put in normalized and standardized form as required by the ML algorithm. As the project proceeds, the data pipeline will mature to the point that it is put in production, operates automatically, at scale, and, if required, in real-time.

### Model Development
Model development takes the the pre-processed features, select which features will be used in the model, performs model training and test, and assess the model effectiveness in achieving the business objectives. Model development is an iterative cycle, cycling between feature selection, training, and model selection. It is important to point out that models such as computer vision and natural language models operating on unstructured data with deep-learning neural structures (RNN - Recursive Neural Networks or CNN - Convolutional Neural Networks) have built-in feature extraction.  However, even these models will require data preparation and potentially human feature engineering when receiving mixed inputs (structured and unstructured data). They will also require data pipelines that properly prepare machine learning data.

Model development is an iterative process, also including iteration with feature extraction to improve upon or create new features as insights are developed. Training and testing the machine learning model requires separating the data into a Test Set and Training Set. Several model types are often tried, such as Decision Tree, SVM (Support Vector Machine), Random Forest, Gradient Boost, or Deep Learning models. Not all the machine learning features will be useful to the model. Some variables will be redundant and cause adverse multi-collinearity effects. Trimming the feature set down to those useful for prediction is called feature selection. The results of too many variables is termed "the curse of dimensionality" and often results in poor model performance. There are few methods for reducing the impact of too many variables and multi-collinearity, such as feature reduction with PCA (Principal Component Analysis) or eliminating variables based on variable importance. Training the model against the training set reveals which variables are best used in the model.

After the model is trained, it is tested against the Test Set and evaluated against meeting the business objectives. Since the model is not exposed to the Test Set during training, model performance with the Test Set indicates that it can generalize for unseen data.

### Model Deployment
Following the model development, the model is ready for deployment. An essential step in the model deployment process is A/B testing. Though the model development process often emulates many real-life scenarios, it is still necessary to verify that the model performs as expected in the real world. Hypothesis testing (A/B test) compared to the existing production model (e.g., 2-sample t-Test, model A vs. model B) is used to ensure statistical verifiability that the new model performs as expected in the real world. The model is gradually put into production, starting with a small percentage of the target population and increasing to the entire customer population.

Model deployment requires integrating the model into the business process. This is often facilitated by including the model as part of the software development CI/CD (Continuous Integration and Deployment) process or uploading the model to a service such as Amazon Sagemaker. Additionally, the model may need to be enhanced so that it operates over larger data sets, optimized for operation in real-time, or handles real-world corner cases.

### Model Operations
Once the model is in production, it will need constant monitoring to ensure that it performs as expected. Some models learn (i.e., train) and adapt in production. However, several situations will require enhancement or tuning of the model. For example, such cases include the data statistics or user behavior changing such that improved model design, model hyper-parameters, or a different type of model is needed. New types of models may be discovered that bring better performance and ROI, such as deep learning AI-based models, which use additional information such as language and sentiment. If the model does not automatically retrain, the model performance may slowly drift to require a regular update. Another typical reason to update the model is new predictive data becoming available so that the data pipeline is updated and new features are available.


### Insights
Finally, insights are generated in almost every step of the process. Insights are often visualized in charts or graphs, become powerful when they are timely, drive action, and are integrated into stories.$^{4}$ A useful tool for communicating and discovering insights is a data dashboard, with drill-downs that facilitate data exploration by a non-data scientist or domain expert. Insights can also be operationalized by delivering personalized alerts and reports to functional business managers and or customers.


# References

[1] Aiden V. Johnson, [Data Science Method (DSM) - a framework on how to take your data science projects to the next level](https://aiden-dataminer.medium.com/the-data-science-method-dsm-modeling-56b4233cad1b), Medium, December 30, 2019

[2] Dr. Datman, [Data Science Modeling Process and Six Consultative Roles](https://towardsdatascience.com/data-science-modeling-process-fa6e8e45bf02), Towards Data Science, November 12, 2018

[3] Aurélien Géron, [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/), O'Reilly Media, Inc. 2nd Edition, September 2019.

[4] Bryent Dykes, [Actionable Insights](https://www.forbes.com/sites/brentdykes/2016/04/26/actionable-insights-the-missing-link-between-data-and-business-value/#39f705df51e5), Forbes, April 26, 2016.
