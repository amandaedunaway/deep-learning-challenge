# deep-learning-challenge
Module 21 Challenge

## Overview of the Analysis
The purpose of the analysis is to optimize a neural network model to predict whether a grant recipient will utilize funds successfully. 
- The dataset contains information about each grant recipient. It is imported into a DataFrame and preprocessed. 
- It is then analyzed by creating a sequential model using tensorflow: `.keras.models.Sequential()`.
- Layers are added using `.keras.layers.Dense()`.
- The model is compiled and fitted using `.compile()` and `.fit()`, respectively.
- Lastly, the model's accuracy is evaluated using the test data.   


## Results
### Data Preprocessing
#### Variable descriptions
- The target variable is a binary variable for whether the funding was effectively utilized.
- The feature variables include: 
    - categorical variables: the application type, the industry affiliation, the use case for funding, the organization type, and the income classification.
    - binary variables: whether the application is active and whether it has any special considerations.
    - a continous variable: the funding amount requested.
- The applicant name and identification number are not part of the analysis.
#### Preprocessing Outline
- To simplify two categorical variables, `APPLICATION_TYPE` and `CLASSIFICATION`, bins labelled "Other" are created for labels that don't meet a count threshold.
- The categorical variables are encoded using `.get_dummies()`.
- The data is divided into training and testing datasets using `train_test_split()` and scaled using a `StandardScaler()` instance.
### Compiling, Training, and Evaluating the Model
- The input layer for the initial model contains 43 input dimensions, given the number of features. It contains 80 neurons, which is just less than approximately twice the input dimensions. It utilizes the relu activation function, which can efficiently handle potential non-linearity.
- The second hidden layer contains 30 neurons, to determine how the model performs if the second hidden layer is less complex than the first, given the same activation function.
- The output layer contains 1 unit, since it is a binary classification model, and uses the sigmoid activation function, which produces probabilities.
- The model's accuracy of 0.725 is close but falls short of the target model performance of 0.75.
- Three attempts were made to increase model perfomance. 
    - The first alternate model has 200 epochs instead of 100.
    - The second alternate model has the original 100 epochs but utilizes 80 neurons in the second hidden layer.
    - The third alternate model utilizes 100 epochs and 30 neurons in the second hidden layer but adds a third hidden layer.


## Summary and Next Steps
Overall, the neural network model did not meet the benchmark of predicting with greater than 75% accuracy. The first alternate optimization model, which increased the number of epochs, performed slightly better than the original model, with 0.726 accuracy compared to 0.725 accuracy, although this is not a meaningful difference. The other optimization techniques resulted in slightly poorer accuracy.

Two possible next steps include further experimentation with other optimization methods, such as adjusting the bins for the relevant columns, or implementing a different model. Another possible model that could solve this classification problem is a Random Forest classifier. Random Forest builds decision trees instead of neural networks, can handle both linear and non-linear data, and can predict binary outcomes.

#### Resources
ChatGPT was used to learn more details about the differences between the relu and sigmoid activation functions.