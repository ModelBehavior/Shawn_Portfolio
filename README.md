# Ra'Shawn Howard
Data Science Portfolio 
# [Project 1: Car Seats Cost What!?](https://github.com/ModelBehavior/Car_Seat_Sales/blob/main/Project1.Rmd)
## Regression Analysis of Car Seat Sales
### Questions:
- What is the relationship between Sales and Price?
- Can we explain car seat sales by Price, urban(YES or NO), and US(YES or NO)?
- What can we tell from the coefficients of the model?
- Can any predictors be removed from the model?
- How well can this model predict sales?
- Can this model be extended (can we improve this analysis)?
### Data:
The data used for this project comes from the ISLR package. There are 400 observations and 11 variables. 3 of which are factors, and the other 8 are numeric.
### Methods and Results:
Regression analysis with Sales as the response and Price, Urban and the US as the explanatory variables. 
10-fold Cross-validation to see how well the model will fit new data using RMSE as the metric.
I dropped the Urban variable because of its non-significance in the model.
Residual plots to check assumptions of the linear model.
From the coefficients, we can conclude that the average unit sales among stores not in the US are 13.031.
The average car seat sales among stores in the US is 1200 units higher than stores not in the US.
With all other variables held constant, on average sales will fall by roughly 54 seats for every $1 increase in price.
Using 10-fold cross-validation, we have an average RMSE of 2.4478658, which means we can expect our predictions to be off by 2.4478658 on average when applied to a new set of data.

![](/images/project1_image)

### Limitations and Next Steps:
This analysis only included 2 explanatory variables out of 10. It could be possible for this analysis to give different or better results using more or varying subsets of the explanatory variables, such as the income variable.

# [Project 2: Whats on The Dollar Menu?](https://github.com/ModelBehavior/McDonalds_EDA)
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

# [Project 3: Ask a Manager](https://github.com/ModelBehavior/TidyTuesday_Ask_a_Manager)
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

# [Project 4: Wait Thats Spam!](https://github.com/ModelBehavior/Spam_Detection/blob/main/Spam%20Prediction.Rmd)

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

# [Project 5: Mario Kart 64 World Records](https://github.com/ModelBehavior/Mario_Kart/blob/main/Mario_Kart.Rmd)
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
![](/images/project5_image)

# Limitations and Next Steps
Making interactive plots would be the next step.
