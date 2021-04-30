# Shawn_Portfolio
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

# [Project 2: Whats on The Dollar Menue?](https://github.com/ModelBehavior/McDonalds_EDA)
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
