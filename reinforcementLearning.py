# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:23:55 2019

@author: ajseshad
"""

#Used to optimize the multi-armed bandit class of problems
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numAds = 10
numUsers = 10000

def ImportData(filename):
    return pd.read_csv(filename)

def RandomChoice(dataset):
    import random
    ads_selected = []
    total_reward = 0
    for i in range(0,numUsers):
        ad = random.randrange(numAds)
        ads_selected.append(ad)
        total_reward += dataset.values[i, ad]
    PlotHistogram(ads_selected, 'Histogram of AD selection - Random Choice')
    return total_reward

#Upper confidence bound
def UCB(dataset):
    import math
    numADSelection = [0] * numAds
    sumADRewards = [0] * numAds
    selectedAds = [0] * numUsers
    totalRewards = 0
    for i in range(0, numUsers):
        maxUB = 0
        for j in range(0, numAds):
            if numADSelection[j] > 0:
                avg_reward = sumADRewards[j] / numADSelection[j]
                delta_reward = math.sqrt(3/2 * math.log(i + 1) / numADSelection[j])
                upper_bound = avg_reward + delta_reward
            else:
                upper_bound = 1e400
            if (upper_bound > maxUB):
                selectedAds[i] = j
                maxUB = upper_bound

        selectedAd =  selectedAds[i]
        numADSelection[selectedAd] += 1
        reward = dataset.values[i, selectedAd]
        sumADRewards[selectedAd] += reward
        totalRewards += reward

    PlotHistogram(selectedAds, 'Histogram of AD selection - Upper Confidence Bound')
    return totalRewards

def ThomsonSampling(dataset):
    import random
    numADSelection = [0] * numAds
    numRewards_1 = [0] * numAds
    numRewards_0 = [0] * numAds
    selectedAds = [0] * numUsers
    totalRewards = 0
    for i in range(0, numUsers):
        # Draw is random but based on the probability distribution
        maxDraw = 0
        for j in range(0, numAds):
            randomDraw = random.betavariate(numRewards_1[j] + 1, numRewards_0[j] + 1)
            if (randomDraw > maxDraw):
                selectedAds[i] = j
                maxDraw = randomDraw

        selectedAd =  selectedAds[i]
        numADSelection[selectedAd] += 1
        reward = dataset.values[i, selectedAd]
        numRewards_1[selectedAd] += reward
        numRewards_0[selectedAd] = numADSelection[selectedAd] - numRewards_1[selectedAd]
        totalRewards += reward

    PlotHistogram(selectedAds, 'Histogram of AD selection - Thomson Sampling')
    return totalRewards
    

def PlotHistogram(ads_selected, title):
    plt.hist(ads_selected)
    plt.title(title)
    plt.xlabel('Ads')
    plt.ylabel('No of times each Ad was selected')
    plt.show()

dataset = ImportData("Ads_CTR_Optimisation.csv")
total_rewards1 = RandomChoice(dataset)
total_rewards2 = UCB(dataset)
total_rewards3 = ThomsonSampling(dataset)