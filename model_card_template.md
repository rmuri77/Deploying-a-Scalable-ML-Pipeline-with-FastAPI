# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classification model trained to predict whether an individual's income exceeds $50K per year based on U.S. Census data. The model was trained using a Logistic Regression classifier implemented with scikit-learn. Categorical features were encoded using one-hot encoding, and the model outputs a binary income classification <=50K or >50K.
## Intended Use
It is designed to showcase how a machine learning model can be trained, evaluated, deployed via an API, and tested using CI/CD practices. This model should not be used for real-world decision-making related to employment, finance, or income prediction.

## Training Data
The model was trained using the Census Income dataset, which includes a demographic and employment-related attributes such as age, education, workclass, occupation, marital status, race, sex, hours worked per week, and capital gains/losses. The dataset was split into training and test sets using an 80/20 train-test split.
## Evaluation Data
The evaluation data consists of the held-out 20% test split from the Census Income dataset. This dataset was not seen during training and was used to assess the model’s generalization performance.
## Metrics
The model was evaluated using the following classification metrics:
- Precision
- Recall
- F1 Score

On the test dataset, the model achieved approximately:
- Precision: 0.72
- Recall: 0.60
- F1 Score: 0.65

Additionally, performance was evaluated across categorical slices to identify potential performance disparities.

## Ethical Considerations
This model is trained on historical census data, which may contain societal biases related to income, race, gender, and occupation. As a result, the model may reflect or amplify these biases. Predictions should be interpreted with caution, and fairness assessments should be performed before any real-world application.
## Caveats and Recommendations
The model is limited by the quality training data. It does not account for changes in economic conditions or income distributions over time. Future improvements could include using more advanced models, addressing class imbalance, and applying fairness-aware machine learning techniques. This model should not be deployed in production systems without additional validation.