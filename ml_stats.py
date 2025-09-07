#!/usr/bin/env python
# coding: utf-8

# In[103]:


import numpy as np
import stats
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ml_stats:
    def __init__(self, user_data):
        self.data = user_data

    def total(self):
        self.total = sum(self.data)
        print("Total:", self.total)
    
    def length(self):
        self.length = len(self.data)
        print("Length:", self.length)
    
    def maximum(self):
        self.maximum = max(self.data)
        print("Maximum Value:", self.maximum)
    
    def minimum(self):
        self.minimum = min(self.data)
        print("Minimum value:", self.minimum)

    def rang(self):
        self.rang = max(self.data) - min (self.data)
        print("Range of Data:", self.rang)
    
    def mean(self):
        self.mean = np.mean(self.data)
        print("Mean:", self.mean)
    
    def median(self):
        self.median = np.median(self.data)
        print("Median:", self.median)
    
    def mode(self):
        try:
            self.mode = stats.mode(self.data)
            print("Mode:", self.mode)

        except:
            print("No Mode found. Computer is fine")
    
    def IQR(self):   
        self.q1 = np.percentile(self.data, 25)
        print("Quartile 1:", self.q1)
        
        self.q3 = np.percentile(self.data, 75)
        print("Quartile 3:", self.q3)
        
        self.IQR = self.q3 - self.q1
        print("IQR:", self.IQR)
    
    def LowerUpper(self):
        self.lower = self.q1 - self.IQR * 1.5
        print("Lower Whisker:", self.lower)
        
        self.upper = self.q3 + self.IQR * 1.5
        print("Upper Whisker:", self.upper)

    def outliers_count(self):
        outliers= len([i for i in self.data if i>self.upper or i <self.lower])
        print('Total Outliers Count:', outliers)
    
    def variance(self):
        self.variance = np.var(self.data)
        print("Variance", self.variance)
    
    def std_dev(self):
        self.std_dev = np.std(self.data)
        print("Standard Deviation", self.std_dev)
    
    def emperical_rule(self):
        for i in range(1,4):
            self.sd_minus_1 = self.mean - i*self.std_dev
            self.sd_1 = self.mean + i*self.std_dev
            self.perc = ((self.data >= self.sd_minus_1) & (self.data <= self.sd_1)).sum() / self.length
            print(f"{i} Standard Deviation Range: {self.sd_minus_1:.2f} to {self.sd_1:.2f} --> {self.perc*100:.2f}% of data")
    
    def SkewKurt(self):
        self.skew = stats.skewness(self.data)
        print("Skewness:", self.skew)
        self.kurt = stats.kurtosis(self.data)
        print("Kurtosis:", self.kurt)
        
    def normal_dist(self):
        self.stat_val, self.p_values = shapiro(self.data)
        if self.p_values >= 0.05:
            print("Likely follows Normal Distribution")
        else:
            print("Likely does NOT follows Normal Distribution")

    def confidence_interval(self):
        self.standard_error = self.std_dev/np.sqrt(self.length)
        
        for confi, z in [(95, 1.96), (97, 2.17), (99, 2.576)]:
            self.moe = self.standard_error * z
            self.lower = self.mean - self.moe
            self.upper = self.mean + self.moe
            print(f"{confi}% Confidence Interval: {self.lower:.2f} to {self.upper:.2f}")

    def visuals(self):
        if len(set(self.data))<7:
            print("Detected Discrete Data")
            plt.figure(figsize = (6,6))
            sns.countplot(self.data)
            plt.show()

            plt.figure(figsize = (6,6))
            data_count = pd.Series(self.data).value_counts()
            plt.pie(data_count, labels = data_count.index, autopct = "%0.2f%%")
            plt.show()
        else:
            print("Detected Continuous Data")
            plt.figure(figsize = (6,6))
            sns.boxplot(self.data)
            plt.show()

            plt.figure(figsize = (6,6))
            sns.histplot(self.data, kde = True)
            plt.show()
            
    def stats_process(self):
        self.total()
        self.length()
        self.maximum()
        self.minimum()
        self.rang()
        self.mean()
        self.median()
        self.mode()
        self.IQR()
        self.LowerUpper()
        self.outliers_count()
        self.variance()
        self.std_dev()
        self.emperical_rule()
        self.SkewKurt()
        self.normal_dist()
        self.confidence_interval()
        self.visuals()