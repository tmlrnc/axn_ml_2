"""
Market Basket Analysis for Association Rules

<img src="images/market1.png" alt="DIS">



Step 1
----------
    READ FILE_IN_RAW.CSV
    Drop all rows that have blanks

    python -m zeroblank --file_in RAW.csv      --file_out RAW_NO_BLANKS.csv






Step 2
----------
    Transform floats to integer categories

    python -m transform --file_in RAW_NO_BLANKS.csv      --file_out RAW_NO_BLANKS_TRANSFORM.csv



Step 3
----------
    One hot encode all strings and integers to categories

    python -m ohe --file_in RAW_NO_BLANKS_TRANSFORM.csv      --file_out RAW_NO_BLANKS_TRANSFORM_OHE.csv --ignore ID



Step 4
----------
    Calculate apriori frequency item sets for all consequents of all antecedents

    python -m market_basket_analysis --file_in RAW_NO_BLANKS_TRANSFORM_OHE.csv --file_out RAW_NO_BLANKS_TRANSFORM_OHE_RESULTS.csv




Step 5
----------
   Calculate association rules: support, confidence, lift for all consequents of all antecedents




Step 6
----------
    Sort by consequents then by antecedents then by Confidence



Step 7
----------
   Write CSV report

The most common interpretation of r-squared is how well the regression model fits the observed data. For example,
an r-squared of 60% reveals that 60% of the data fit the regression model. Generally, a higher r-squared indicates a better fit for the model.


Correlation Coefficient, denoted by r, tells us how closely data in a scatterplot fall along a straight line.
The closer that the absolute value of r is to one, the better that the data are described by a linear equation.
If r =1 or r = -1 then the data set is perfectly aligned. Data sets with values of r close to zero show little to no straight-line relationship.
We begin with a few preliminary calculations. The quantities from these calculations will be used in subsequent steps of our calculation of r:
Calculate x̄, the mean of all of the first coordinates of the data xi.
Calculate ȳ, the mean of all of the second coordinates of the data
yi.
Calculate s x the sample standard deviation of all of the first coordinates of the data xi.
Calculate s y the sample standard deviation of all of the second coordinates of the data yi.
Use the formula (zx)i = (xi – x̄) / s x and calculate a standardized value for each xi.
Use the formula (zy)i = (yi – ȳ) / s y and calculate a standardized value for each yi.
Multiply corresponding standardized values: (zx)i(zy)i
Add the products from the last step together.
Divide the sum from the previous step by n – 1, where n is the total number of points in our set of paired data.
The result of all of this is the correlation coefficient r.
Generally, a value of r greater than 0.7 is considered a strong correlation. Anything between 0.5 and 0.7 is a
moderate correlation, and anything less than 0.4 is considered a weak or no correlation.

SUPPORT is how frequent an Antecedent is in all the transactions
SUPPORT = (Num Transactions with Antecedent AND Consequent )/Total Num Transaction

CONFIDENCE is likeliness of occurrence of Consequent Given the Antecedent
CONFIDENCE = (Num Transactions with Antecedent AND Consequent )/ Num Transactions with Antecedent

Lift is how much better a Antecedent is at predicting the Consequent than just assuming the Consequent in the first place.
Lift = ((Num Transactions with Antecedent AND Consequent )/ Num Transactions with Antecedent) )
/ ((Num Transactions with Consequent) / Total Num Transaction)



Market Basket Analysis also called Association analysis is light on the math concepts and easy to explain to non-technical people.
In addition, it is an unsupervised learning tool that looks for hidden patterns so there is limited need for data prep and feature engineering.
It is a good start for certain cases of data exploration and can point the way for a deeper dive into the data using other approaches.

Association rules are normally written like this: {Diapers} -> {Beer} which means that there is a strong relationship between customers
that purchased diapers and also purchased beer in the same transaction.

In the above example, the {Diaper} is the antecedent and the {Beer} is the consequent.
Both antecedents and consequents can have multiple items.
In other words, {Diaper, Gum} -> {Beer, Chips} is a valid rule.

Support is the relative frequency that the rules show up.
In many instances, you may want to look for high support in order to make sure it is a useful relationship.
However, there may be instances where a low support is useful if you are trying to find “hidden” relationships.

Confidence is a measure of the reliability of the rule.
A confidence of .5 in the above example would mean that in 50% of the cases where Diaper and Gum were purchased,
the purchase also included Beer and Chips. For product recommendation, a 50% confidence may be perfectly acceptable.

Lift is the ratio of the observed support to that expected if the two rules were independent.
The basic rule of thumb is that a lift value close to 1 means the rules were completely independent.
Lift values > 1 are generally more “interesting” and could be indicative of a useful rule pattern.
List > 6 is a HIT


Leverage computes the difference between the observed frequency of A and C appearing together and the frequency
that would be expected if A and C were independent.
An leverage value of 0 indicates independence.


A high conviction value means that the consequent is highly depending on the antecedent. For instance, in the case of
a perfect confidence score,
the denominator becomes 0 (due to 1 - 1) for which the conviction score is defined as 'inf'. Similar to lift,
if items are independent, the conviction is 1



The support metric is defined for itemsets, not assocication rules.
The table produced by the association rule mining algorithm contains three different support metrics: 'antecedent support',
'consequent support', and 'support'.
Here, 'antecedent support' computes the proportion of transactions that contain the antecedent A, and
'consequent support'  computes the support for the itemset of the consequent C.

The 'support' metric then computes the support of the combined itemset A ∪ C -- note that 'support' depends on
'antecedent support' and 'consequent support' via min('antecedent support', 'consequent support').

Here is one of 5 shopping cart optimization techniques I used.
I increased click through to Walmart Shopping Carts using my custom Deep Reinforcement Learning model with a
Deep Q-network on top of TensorFlow libraries.
As you know, reinforcement learning is the area of machine learning that is focused on training agents to take
certain actions at certain states from within an
environment to maximize rewards.
My RF DQN is a combination of deep learning and reinforcement learning. My model target is to approximate Q(s, a),
and is updated through back propagation.
Assuming the approximation of Q(s, a) is y(hat) and the loss function is L, we have: prediction: y(hat) = f(s, θ)
loss: L(y, y(hat)) = L(Q(s, a), f(s, θ)).
My DQN has 3 convolutional layers and 2 fully connected layers to estimate Q values directly from shopping carts.
Every day for 14 months I ran a batch training workflow to update my Walmart Shopping Cart Reinforcement
Learning Agent and generated times to send the next push notifications.
The workflow gathers the following data for training:

    The State of the users from 2 days ago
    The Action the hour a notification was sent from 2 days ago
    The next State of the user from 1 day ago
    User engagement values the Reward from 1 day ago

For each Walmart shopping cart, this is assembled into a trajectory of (State, Action, Next State, Reward)
This set of trajectories is used to update the existing RL DQN Agent. The update means running through a deep
learning workflow on
Tensorflow: the values of the neural network representing the Agent are updated to better reflect the relationships
between States, Actions and long term rewards.
My RL Agent keeps track of the long term reward implicitly. I am only telling the Agent about the immediate reward from the following day.
The RL Algorithms build up their own internal estimates of the long term reward, based on the immediate rewards and the next State.
This is a batch RL DQN. Over the span of 14 months of data collection and training, the agent increased the click through rate by 10% .
It algorithmically tests different strategies to personalize values for each user and learns
how to optimize for key metrics such as long term retention and engagement.
The agent was able to answer this question: what hour do I send each user a message in order to raise engagement.
It automates new strategies.


<img src="images/market2.png" alt="DIS">




<img src="images/market3.png" alt="DIS">



Steps and Parameters:
----------

 python -m zeroblanks --file_in RAW.csv      --file_out RAW_NO_BLANKS.csv


 python -m transform --file_in RAW_NO_BLANKS.csv      --file_out RAW_NO_BLANKS_TRANSFORM.csv


 python -m ohe --file_in RAW_NO_BLANKS_TRANSFORM.csv      --file_out RAW_NO_BLANKS_TRANSFORM_OHE.csv --ignore ID


  python -m market_basket_analysis --file_in RAW_NO_BLANKS_TRANSFORM_OHE.csv --file_out RAW_NO_BLANKS_TRANSFORM_OHE_RESULTS.csv


Support:
----------
Support is the relative frequency that the rules show up.
In many instances, you may want to look for high support in order to make sure it is a useful relationship.
However, there may be instances where a low support is useful if you are trying to find “hidden” relationships.

SUPPORT is how frequent an Antecedent is in all the transactions
SUPPORT = (Num Transactions with Antecedent AND Consequent )/Total Num Transaction

Confidence:
----------
Confidence is a measure of the reliability of the rule.
A confidence of .5 in the above example would mean that in 50% of the cases where Diaper and Gum were purchased,
the purchase also included Beer and Chips. For product recommendation, a 50% confidence may be perfectly acceptable.

CONFIDENCE is likeliness of occurrence of Consequent Given the Antecedent
CONFIDENCE = (Num Transactions with Antecedent AND Consequent )/ Num Transactions with Antecedent

Lift:
----------
Lift is the ratio of the observed support to that expected if the two rules were independent.
The basic rule of thumb is that a lift value close to 1 means the rules were completely independent.
Lift values > 1 are generally more “interesting” and could be indicative of a useful rule pattern.
List > 6 is a HIT

PROCESS:
Preparing the Dataset of CATEGORIES using OHE
A one hot encoding is a representation of categorical variables as binary vectors. This first requires that the categorical values be mapped to integer values. Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.
Extract Frequent Itemsets
Extract Association Rules
Extract Rules
Define Threshold and extract the final associations



Example 1.
---------------------

INPUT:
<img src="images/ex1in.png" alt="DIS">


OUTPUT:
<img src="images/ex1out.png" alt="DIS">





TEST 1 - Data Input CSV File:
----------------------------


cd /Users/tomlorenc/Sites/VL_standard/ml/axn/ml

Married_Single_Health_IN_TEST.csv -
<img src="images/1.png" alt="DIS">


1) Zero all rows with blanks
ADD ID COLUMN

python -m zeroblank --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/Married_Single_Health_IN_TEST.csv
--file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/Married_Single_Health_IN_TEST_zero.csv

Married_Single_Health_IN_TEST_zero.csv -
<img src="images/2.png" alt="DIS">


2) One hot encode - all strings and integers to categories
REMOVE ID COLUMN

python -m ohe --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/Married_Single_Health_IN_TEST_zero.csv
--file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/Married_Single_Health_IN_TEST_zero_ohe.csv --ignore ID

Married_Single_Health_IN_TEST_zero.csv -
<img src="images/3.png" alt="DIS">

3) MBA

python -m market_basket_analysis --file_in  /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/Married_Single_Health_IN_TEST_zero_ohe.csv
--file_out  /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/Married_Single_Health_IN_TEST_zero_ohe_results.csv

Married_Single_Health_IN_TEST_zero_ohe_results.csv -
<img src="images/4.png" alt="DIS">





TEST 2 -
----------------------------


1) Zero all rows with blanks
ADD ID COLUMN

python -m zeroblank --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6.csv
--file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6_zero.csv



2) One hot encode - all strings and integers to categories
REMOVE ID COLUMN

python -m ohe --file_in //Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6_zero.csv
--file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6_ohe.csv --ignore ID

3) MBA

python -m market_basket_analysis --file_in  /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6_ohe.csv
--file_out  /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6_ohe_results.csv.csv


REMOVE COLUMN
sed 's/^[^,]*,//' 9_29_MBA_IN_V6_ohe_results.csv > 9_29_MBA_IN_V6_ohe_results2.csv

4) MBA Report



python -m mba_report --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6_ohe_results2.csv
--col_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6.csv
--count_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6_ohe.csv
 --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_29_MBA_IN_V6_ohe_results_2_report.csv


9_22_MBA_IN_zero_ohe_results_2_report.csv -
<img src="images/report.png" alt="DIS">

TEST 2 -
----------------------------


1) Zero all rows with blanks
ADD ID COLUMN

python -m zeroblank --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_23_MBA_IN.csv
--file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_23_MBA_IN_zero.csv



2) One hot encode - all strings and integers to categories
REMOVE ID COLUMN

python -m ohe --file_in //Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_23_MBA_IN_zero.csv
--file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_23_MBA_IN_zero_ohe.csv --ignore ID

3) MBA

python -m market_basket_analysis --file_in  /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_23_MBA_IN_zero_ohe.csv
--file_out  /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_23_MBA_IN_zero_ohe_results.csv


REMOVE COLUMN
sed 's/^[^,]*,//' 9_23_MBA_IN_zero_ohe_results.csv > 9_22_MBA_IN_zero_ohe_results_3.csv

4) MBA Report



python -m mba_report --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_22_MBA_IN_zero_ohe_results_2.csv
--col_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_22_MBA_IN.csv
--count_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_22_MBA_IN_zero_ohe.csv
 --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/9_22_MBA_IN_zero_ohe_results_2_report.csv



"""
# pylint: disable=unused-variable
# pylint: disable=duplicate-code

import argparse

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def main():
    """
MBA


      """
    # pylint: disable=duplicate-code

    ######################################################################
    #
    # read run commands
    #
    # pylint: disable=duplicate-code


    ######################################################################

    #
    #
    # pylint: disable=duplicate-code
