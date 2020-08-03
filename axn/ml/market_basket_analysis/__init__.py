"""
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
"""
# pylint: disable=unused-variable
# pylint: disable=line-too-long
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
