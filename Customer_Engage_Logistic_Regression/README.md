# __PREDICTING CUSTOMER SUBSCRIPTIONS IN BANKING: A LOGISTIC REGRESSION APPROACH__

![Image of Banking Transaction 1](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/9800577a-1011-490e-81a0-7d0b4489f540)

## __Below, you will find a walkthrough of the main modules from the notebook, highlighting the most relevant steps in this data science project. These steps enabled us to model the probability of customers contracting new financial services using a logistic regression model.__

#  __MAIN GOAL__
__The proposed project of modeling under Machine Learning approaches has been aimed at leveraging informed decision-making in the marketing areas of financial institutions. Therefore, the main objective seeks to predict the probability that institution's customers will contract new financial products and services during campaign events, thereby understanding how effective these campaigns have been.__

## __Dataset Description__
About Dataset the Bank Client Attributes and Marketing Outcomes dataset taken from Kaggle, offers a comprehensive insight into the attributes of bank clients and the outcomes of marketing campaigns. It includes details such as client demographics, employment status, financial history, and contact methods. Additionally, the dataset encompasses the results of marketing campaigns, including the duration, success rates, and previous interactions with clients. This dataset serves as a valuable resource for analyzing customer behavior, optimizing marketing strategies, and enhancing client engagement in the banking sector.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/8a251804-f936-4949-92ce-8bdd2a1ae84e)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/3765b7b5-ed25-4839-a106-e54aaaa60398)


## __Scheme of the general approach adopted for the development of the analysis__
To articulate a classification problem using the field “y” as a predictor, we need to identify the target variable or the variable we want to predict. In this dataset, it seems that the field y represents whether a client subscribed to a product or service offered by the bank (e.g., term deposit). Typically, in banking marketing datasets like this, y often represents the outcome of a marketing campaign, such as whether a client subscribed to a new banking service.
So, the classification problem here would be predicting whether a client will subscribe to the product or service (y = "yes") or not (y = "no").

Given this classification problem, we can use various machine learning approaches, but some common ones include the Logistic Regression: It's a simple and effective algorithm for binary classification problems like this.

## __Exploratory Data Analysis__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/52d422b6-8ab5-470f-9ea8-c0b8b06e7071)  

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/9dc25f5a-24b6-4674-99f5-bee4c953430f)

## __Summary of Data Exploration, Cleaning and Feature Engineering__
As I went in deep into my modeling process, each step was meticulously executed to ensure the robustness and reliability of this analyses. Here's a comprehensive overview of my approach for handling data that is utilized across several variant for  Logistic Regression approaches:

## __Data Exploration:__
Examined the dataset to understand its structure and characteristics.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/f73ae862-b30d-40de-a2d2-28582ed7e022)

### __Actions Taken for Data Cleaning:__
Handled missing values by dropping rows with missing values.
![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/8bd1d957-5ba2-42d6-a716-05f5a969bfcc)

### __Outliers Detection:__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/40bd4f95-9d27-4cf8-bb9c-d7f6faa0903f)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/8a7add41-6db7-4e43-9b7e-c05a7ac84996)

Employed Winsorization and robust treatment methods to mitigate outliers.

### __Feature Engineering:__
Defined numeric and categorical columns.
Created preprocessor pipelines for data preprocessing.

### __Dataset Split__
Let's split the dataset into a training and a testing dataset. Training dataset will be used to train and (maybe) tune models, and testing dataset will be used to evaluate the models. Note that I may also split the training dataset into train and validation sets where the validation dataset is only used to tune the model and to set the model parameters.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/958f0522-0bed-4b1f-aa16-ccb79c8e54fc)

### __Feature Engineering__
Next, let's process the raw dataset and construct input data X and label/output y for logistic regression model training.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/222ac2b5-8e9e-44b9-af98-0bf54672d313)

### __Correlation Analysis__

![newplot](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/02130693-8ebd-40ec-bf14-e67b0f9d6cd4)

After exploring the correlations between the target variable and all the final variables involved once processing is done, which is useful to get an idea of those that could most influence the response variable, however, this introduces complexity into the model.

Logistic regression models can suffer from overfitting when the number of predictors (features) is large relative to the number of observations. Since I have many encoded categorical variables, each potentially introducing multiple new features (after one-hot encoding), the risk of overfitting is increased. It's essential to consider if all variables are necessary or if feature selection techniques (like regularization) could help.

Regularization: Techniques like Lasso (L1 regularization) or Ridge (L2 regularization) regression can help mitigate overfitting by penalizing the coefficients of less important variables, effectively shrinking them towards zero. This encourages simpler models and reduces the impact of less relevant features.

## __BUILD LOGISTIC REGRESSION MODELS__

Now we have the training and testing datasets ready, let's start the model training task. We first define a sklearn.linear_model.LogisticRegression model with the following arguments, you can check the comment for each argument for what it means.

## __Logistic Regression with L2 Penalty__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/f5cd63d6-540c-4924-8bbc-2bbf37e71e54)
![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/7980ac00-9768-4579-8bf8-49b58f5a5c39)
![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/d1ac0f84-7ac1-4db4-aece-3e2371d28be1)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/b29435d8-4f44-4385-b72b-123bca13a811)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/2aa83656-1892-4156-84cb-2eb8a055df83)

## __Logistic Regression with L1 Penalty__
Next, let's try defining another logistic regression model with l1 penality this time, to see if our classification performance would be improved.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/fde2848e-23e5-40e5-96cb-dc53b648694e)
![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/15c053b5-4d08-4459-b913-8ea505817a8e)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/a98ce8bc-dc0d-4ba2-9eb7-4db323122f92)

Now, we can see this logistic regression with l1 penalty is pretty similar than l2.
As we can see from the above evaluation results, the logistic regression model has relatively good performance on this multinomial classification task. The overall accuracy is around 0.9 for L1 model and L2 model and the f1score is around 0.90 for both models as well. Note that for recall, precision, and f1score, we output the values for each class to see how the model performs on an individual class. And, we can see from the results, the recall for class=1 (More often) is not very good. This is actually a common problem called imbalanced classification challenge. I will test other alternatives for the next steps.

## __Confusion Matrix for Logistic Regression with L1 Penalty__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/1534b073-b06b-42e7-8bd9-7ca27285ade2)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/755f03dd-72e8-413f-9571-4f2c9c18155a)

## __For many machine learning tasks on imbalanced datasets, like this dichotomous phenomena, it recommends care more about recall than precision. As a baseline, we want the model to be able to find all frauds and we would allow the model to make false-positive errors because the cost of false positives is usually not very high (maybe just costs a false notification email or phone call to confirm with customers). On the other hand, failing to recognize positive examples (such as fraud or a deadly disease) can be life-threatening
As such, our priority is to improve the model's recall, then we will also want to keep precision as high as possible.__

## __ALTERNATE APPROACH TO DEALING WITH INTERST CLASS IMBALANCE__

SMOTE first creates many pairs or small clusters with two or more similar instances, the measure by instance distance such as Euclidean distance.
Then, within the boundary of each pair or cluster, SMOTE uniformly permutes features value, one feature at a time, to populate a collection of similar synthesized instances within each pair or cluster.

As a result, __SMOTE creates a class-balanced synthetic dataset without adding duplicated instances with minority labels.__ 

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/e0980528-df4b-44eb-8679-f2806664a6eb)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/e7108843-61c3-4595-9a93-fdba048b2f03)

## __Logistic Regression L1 with SMOTE Treatment__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/be21242a-4ceb-41a2-9a79-9e26de4b30b0)
![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/09078536-40e5-4a42-b214-4a90082ae308)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/b096a9f6-c719-4ce5-aa5f-38a08c979d5c)

After resampling the data, I retrained my model (l1_model) using the resampled data. This allows the model to learn from a more balanced dataset and potentially improve its performance, especially in terms of recall for the minority class.

The accuracy score of the model on the test set is approximately 0.82, indicating that around 82% of the predictions made by the model are correct.

While accuracy is an important metric, it may not provide a complete picture, especially in imbalanced datasets.

## __Confusion Matrix for Logistic Regression L1 with SMOTE Treatment__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/fd9c099e-af97-469a-bc47-bf2148d546f5)

Overall, it seems that the model trained on the resampled data has achieved a higher recall compared to the previous model, indicating an improvement in detecting positive examples (e.g., successful marketing outcomes).

Additionally, further analysis and fine-tuning of the model may be necessary to optimize its performance.

## __INTERPRETING LOGISTIC REGRESSION MODELS__

Once I have choosen the LR model with smote treatment. One way to interpret logistic regression models is by analyzing feature coefficients. Although it may not be as effective as the regular linear regression models because the logistic regression model has a sigmoid function, we can still get a sense for the importance or impact of each feature.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/b694d508-bbc4-431c-baf0-2c741cd8c3db)

The coef_ is a coefficients list with two elements, one element is the actual coefficent for class 0, 1. To better analyze the coefficients, let's use three utility methods to sort and visualize them.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/92bddcf8-d039-4dd9-8063-d9d07c679d53)

## __Visualizing Logistic Regression Coefficients__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/5fac040b-8c13-4a6b-b34d-696b092652a6)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/c87db220-f804-4a0b-b3c9-b40156560d65)

## __Interpreting the coeficients and Odds-ratios__

__Odds Ratios:__
An odds ratio represents the likelihood of an event occurring given a certain condition compared to the likelihood of the same event occurring without that condition. In logistic regression, odds ratios are derived from the predicted probabilities of the target variable being in a certain class.

In this case, the odds ratios are array([0.99201183, 0.00798817]). This means that the odds of the event (class 1) occurring are 0.99201183 / 0.00798817 = ~124 times higher than the odds of the event not occurring (class 0).

Based on the coefficients provided, here's an interpretation of some of the features:

__Housing:__ Having a housing loan increases the log-odds of contracting new products and services by approximately 0.61 units.

__Poutcome:__ A successful previous marketing campaign increases the log-odds of contracting new products and services by approximately 1.01 units.

__Education:__ Higher education levels significantly increase the log-odds of contracting new products and services. The specific increase depends on the education level.


## __CONCLUSIONS__

### Considering the business objective of predicting the probability that customers will contract new financial products during campaign events, the choice of model depends on the specific cost associated with misclassifications. If the cost of missing a potential customer (false negative) is high, a model with a high recall should be chosen. In this context, __Logistic Regression L1 with SMOTE technique is recommended due to its high recall values__.

### On the other hand, the higher impact of education on y in my logistic regression results compared to its correlation coefficient suggests that logistic regression captures nuances and interactions that simple correlation analysis may miss.

### Here other alternative strategies that might improve recall while maintaining a high precision  should be considered as well: 
### Some algorithms are inherently better at handling imbalanced datasets and prioritizing recall, such as Random Forest, Gradient Boosting, or Support Vector Machines. Experimenting with different algorithms may yield better results.














