
# [Project 1: Instagram Classification](https://github.com/ModelBehavior/tecator/blob/main/teactor.Rmd)
## Spam Account Detection
### Data Description
Fakes and spammers are a major problem on all social media platforms, including Instagram. [kaggler](https://www.kaggle.com/free4ever1/instagram-fake-spammer-genuine-accounts) has personally identified the spammer/fake accounts included in this dataset after carefully examining each instance and as such the dataset has high level of accuracy. The dataset has been collected using a crawler from 15-19, March 2019. The response variable is fake, and is coded 0=genuine and 1=spam. There are 11 predictors in the data:

+ profile picture, user has profile picture, or not (coded 0=no picture, 1=has picture)
+ nums length user, this is the ratio of number of numerical chars in username to its length
+ full name words, full name in word tokens
+ nums length full name, the ratio of the number of numerical characters in full name to its length
+ name username, are username and full name literally the same (coded 0=not same, 1=same)
+ description length, bio length in characters
+ external url, has external url or not (coded 0=no external url, 1=has external url)
+ private or not, is the account set to  private viewer ship (coded 0=not private, 1=private)
+ posts, number of posts
+ followers, number of followers
+ follows, number of accounts followed

### Split Data
Before building these models, we will split the data into one set that will be used to develop models, preprocess the predictors, and explore relationships among the predictors and the response (the training set) and another that will be the final arbiter of the predictor set/model combination performance (the test set). To partition the data, the splitting of the orignal data set will be done in a stratified manner by making random splits in each of the outcome classes. This will keep the proportion of fake accounts approximately the same. In the splitting, 82% of the data were allocated to the training set. 10-fold cross-validation were used to tune the models

### Prepprocess
The predictors were transformed using yeoJohnson transformations, centered and scaled, searched for near zero variance predictors, and searched for a predictor space with kendal correlation less than .80.

### Modeling
While often overlooked, the metric used to assess the effectiveness of a model to predict the outcome is very important and can influence the conclusions. The metrics select to evaluate model performance depend on the response variable. The outcome being a binary categorical variable. Accuracy, sensitivity, specificity, kappa, and area under the curve were used to evaluate each model's performance. \

The models tried on this data include logistic regression, linear discriminant analysis, regularized discriminant analysis, flexible discriminant analysis, k-nearest neighbors, single-layer neural network, and C5 boosted trees.

### Logistic Model
This models was created usinhg the penalized version (elastic net) that conducts feature selection during model training. 10 values of the ridge and lasso penalty combinations were examined, using a space filling design.

### LDA Model
This models was created using the penalized version that conducts feature selection during model training. 10 values of the ridge penalty was investigated, using a space filling design.

### RDA Model
This model was created using a space filling design with 10 combination values for fraction common variance that toggles between LDA and QDA by putting a penalty on the covariance matrix $\lambda \Sigma_l + (1-\lambda)\Sigma$ when $\lambda$ is zero we get LDA and when $\lambda$ is 1 we get QDA. For $\lambda$ values between 0 and 1 we get something between LDA and QDA. Fraction identity buts a penalty on the pooled covariance matrix and allows the matrix to morph from its observed value to one where the predictors are assumed to be indepenent. Tuning an RDA model over these parameters enables the training data to decide the most appropriate  assumptions for the model.

### FDA Model
This model, first-degree and second-degree MARS hinge functions were used and the number of retained terms was varied from 2 to 12.

### KNN Model
this model, k varied from 1 to 10.

### NNet Model
Models were fit with hidden units ranging from 2 to 10 and 10 weight decay values determined using a space filling design.

### C5.0 Boosted Model
This model was evaluated with tree-based models, up to 100 iterations of boosting and tuned over sample_size and min_n.

### Training Results
The model results are shown below. The plot show the confidence interval of the resamples over the different metrics. The linear models did just as well as the non-linear models. The best non-linear model is the neural network and the best linear model is the logistic regression. To see if these two models are statistically different a t-test can be performed. I did not do this in this analysis.

![](https://github.com/ModelBehavior/Shawn_Portfolio/blob/main/images/Screen%20Shot%202021-09-25%20at%2012.59.51%20PM.png)

# Results
The best two models were logistic regression and the single-layer neural network. We can see our models dropped in accuracy quite a bit on the testing data, and the logistic regression predicted one more class correctly than the single-layer neural network. The logistic regression model misclassified 12 observations, while the single-layer neural network misclassified 13. Given the simplicity of the logistic regression model, it would be chosen as the overall model. The logistic regression has a false positive rate of 10.77%. And a true positive rate of 91.8%. Meaning when the model predicts genuine, it is right 91.8% of the time, and when the model predicts spam, it is wrong 10.77% of the time.

![](https://github.com/ModelBehavior/Shawn_Portfolio/blob/main/images/Screen%20Shot%202021-09-25%20at%201.02.23%20PM.png)

# [Project 2: Fat Content With IR](https://github.com/ModelBehavior/tecator/blob/main/teactor.Rmd)
## Regression Analysis of Fat content using IR as Predictors
### Data Description
Infrared (IR) spectroscopy technology is used to determine the chemical makeup of a substance. The device measures the absorbance of the sample at each individual frequency. This series of measurements creates a spectrum profile which can then be used to determine the chemical makeup of the sample material. A Tecator Infratec Food and Feed Analyzer intrument was used to analyze 215 samples of meat across 100 frequencies. In addition to an IR profile, analytical chemistry determined the percent fat for each sample.

### Goals of Analysis
If we can establish a predictive relationship between IR spectrum and fat content, then food scientist could predict a sample's fat content, then food scientist could predict a sample's fat content with IR instead of analytical chemistry. This could provide cost savings, since analytical chemistry is  a more expensive, time-consuming process.

### Methodology
The data was split into a testing and training set, and different preprocessing methods were done. \
The predictors are highly correlated, so PCA was used to reduce the dimension of the predictor space. \
Cross-validation was done to find the optimal value of the tuning parameters for models that required this. \
The different types of models that fit the data were: bagged trees, boosted trees, cubist, linear regression, decision trees, MARS, neural networks, KNN, random forest, and SVM. \
The neural network model performed the best on the training data with an RMSE of .85088724 and a standard error of 0.03248912, followed by the cubist model.

![](/images/project1_1)

### Results 
Applying the best model to the test set, we get an RMSE of .7274025 with an r-squared of 0.9968596.

# [Project 3: Whats on The Dollar Menu?](https://github.com/ModelBehavior/McDonalds_EDA)
## Exploratory Data Analysis on McDonald's Data.
### Questions:
- How many calories does the average McDonald's value meal contain?
- How much do beverages, like soda or coffee, contribute to the overall caloric intake?
- Does ordering grilled chicken instead of crispy increase a sandwich's nutritional value?
### Data:
This [dataset](https://www.kaggle.com/mcdonalds/nutrition-facts) provides a nutrition analysis of every menu item on the US McDonald's menu, including breakfast, beef burgers, chicken and fish sandwiches, fries, salads, soda, coffee and tea, milkshakes, and desserts. The menu items and nutrition facts were scraped from the McDonald's website.
### Methods:
Group data by category to get summarizations for graphics and reorder levels to make data more presentable. Filter through data to find Category for Chicken and remove observations that are not grilled or crispy chicken sandwiches such as, removing Filet-O-Fish and Nuggets. Then search through strings to find keyword Grilled, then creating a new variable type with value Grilled if a string contains word Grilled else Crispy. Used t-tests to compare grilled versus crispy chicken means for calories, protein, and total fat. Used t-tests to compare grilled versus crispy chicken means for calories, protein, and total fat. The results of the t-tests show that there is no significant mean difference between the mean calories for grilled and crispy chicken sandwiches. There is a significant mean difference between the protein of crispy and grilled chicken sandwiches. There is a significant mean difference between total fat daily %  of crispy and grilled chicken sandwiches. I used ggplot2 to graph the data to make it presentable and understandable to interested parties.
- Since there is no difference in calories between crispy and grilled chicken on average, we are worried about protein and total fat. From the t-tests, we can see grilled chicken has more protein on average and less total fat, so we can conclude that grilled chicken increases a sandwich's nutritional value.
### Results:

![](/images/project2_image)

### Limitations and Next Steps:
The graphs can be improved using shiny to make an interactive dashboard, which can boost the interpretability of the findings to interested parties.

# [Project 4: Ask a Manager](https://github.com/ModelBehavior/TidyTuesday_Ask_a_Manager)
## Analysis of Manager salaries
# Questions
+ Is there a significant difference bewteen salary by gender?
+ Is there a sig diff between salary by race?
+ Do Mangers on average get paid more with more education?
+ Do older mangers get paid more on average?
+ Do mangers with more years of experiance get paid more on average?

# Data
The data this week comes from the Ask a Manager Survey. H/t to Kaija Gahm for sharing it as an issue!
The salary survey a few weeks ago got a huge response — 24,000+ people shared their salaries and other info, which is a lot of raw data to sift through. Reader Elisabeth Engl kindly took the raw data and analyzed some of the trends in it and here’s what she found. (She asked me to note that she did this as a fun project to share some insights from the survey, rather than as a paid engagement.)
This data does not reflect the general population; it reflects Ask a Manager readers who self-selected to respond, which is a very different group (as you can see just from the demographic breakdown below, which is very white and very female).
[Link to data and extra information](https://github.com/rfordatascience/tidytuesday/blob/master/data/2021/2021-05-18/readme.md)

# Methodology
Using R ggplot, I will explore this weeks tidytuesday data set to answer questions about manager salaries.

# Results

![](/images/Project3_image.png)

# Limitations and Next steps
The presentation of the data can be made better using shiny or a BI tool such as Power BI or IBM Cognos for interactivity, and further exploration. Significance can be further tested through statistical tests such as ANOVA or t-tests, to confirm graphical findings.

# [Project 5: Wait Thats Spam!](https://github.com/ModelBehavior/Spam_Detection/blob/main/Spam%20Prediction.Rmd)

## Sentiment Analysis and Prediction of SMS Messages

# Questions:
+ Can we use this data to predict if a SMS message is spam?
+ What are the most common positive words?
+ What are the most common negative words?
+ What are the most important positive and negative words as described by the model? 

# Data
The dataset can be found [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset). The SMS Spam Collection Dataset is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

# Methods
I used comparison word cloud for the most common positive and negative words. The size of a word's text is in proportion to its frequency within its sentiment. I used bing word bank for the sentiment analysis. I chose a lasso logistic model. This model does well on text data. I used single word tokens, convert tokens into weights using tfidf, only kept 500 tokens after removing stop words. The model is sensitive to centering and scaling, so I normalized the data. I used grid search to find the best penalty. I used bootstrap resampling to test the model before running the final model on the test set. It's easier to detect non-spam SMS messages than it is to detect spam SMS messages. Our overall accuracy rate is good at 96%

# Results
![](/images/project4_img1)
![](/images/project4_img2)

# Limitations and Next Steps
Things we can do to get better results: include not only unigrams but bi-grams, tri-grams, what stopwords make the most sense for my data, include more words in the word bank (I only included 500), we could choose a different weighting other than tfidf, we could try other types of models such as SVM or Naive Bayes.

# [Project 6: Mario Kart 64 World Records](https://github.com/ModelBehavior/Mario_Kart/blob/main/Mario_Kart.Rmd)
## Exploratory Analysis

# Questions
+ How did the world records develop over time?
+ Which track is the fastest?
+ For which track did the world record improve the most?
+ For how many tracks have shortcuts been discovered?
+ When were shortcuts discovered?
+ On which track does the shortcut save the most time?
+ Which is the longest standing world record?
+ Who is the player with the most world records?
+ Who are recent players?

# Data
Tidytuesday data set can be found [here](https://github.com/rfordatascience/tidytuesday/blob/master/data/2021/2021-05-25/readme.md). 

# Methodology
Use R tidyverse to explore and make vizualizations for tidytuesday data

# Results
![](/images/project5_img)

# Limitations and Next Steps
Making interactive plots would be the next step.
