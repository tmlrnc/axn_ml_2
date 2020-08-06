"""
Market Basket Analysis

<img src="images/market.png" alt="DIS">



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
