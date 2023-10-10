#import the panda library that is useful for data cleaning (1st step of our work)
import pandas as pd

#STEP 1: COLLECTION

#we input ourdata to Python and make it a DataFrame that we can work on
df = pd.read_csv("dataset_project_eHealth20232024.csv")

#The following instruction provides the essential details about the dataset, such as the number of rows and
#columns, number of non-null values, type of data of each column and the memory usage of the DataFrame.
df.info()
print ("\n")

#STEP 2: CLEANING

#From slide 21 ppt.02 python
#Common data problems:
# A. Inconsistent column names -> IT'S NOT OUR CASE
# B. Duplicate rows
# C. Column types signaling unexpected data values -> IT'S NOT OUR CASE BECAUSE WHEN WE MADE THE COMMAND df.info()
#ALL ARE WEEL DEFINED AS INT64 O FLOAT64, IF THEY WERE MIXED WE WOULD HAVE SEEN IN THE COLUMN TYE OF THE info
#OBJECT
# D. Outliers -> WE SAID THAT IS NOT OUR CASE
# E. Missing data
# And much, much more…

# B. let's deal with duplicate rows
df = df.drop_duplicates()
df.info() #Printing we see that the number of rows goes from 160 to 150 so 10 duplicates dropped
print ("\n")

# E. Let's deal with missing values;
#Looking at slide 26 ppt.2 of Python there are several chances
# E.1 the removal as said by Tauro is not the best option since we loose information on the other columns
# -> WE TRIED TOGETHER AND WE SAW THAT THERE ARE 48 ROWS THAT HAVE AT LEAST A NaN ELEMENT SO REMOVE THEM 48/150
# WOULD MEAN LOOSE TO MUCH INFO SO IS NOT AN OPTION
# E.2 The one of machine learning, we have not seen it, he said we would have done a lesson on it but for now
# it's not an option
# E.3 The one on the interpolation has the problem on the choice of the number of k that we have seen for the
#for the cluster but not for that step so not
# E.4 the one on forward and backward fill I do not consider the best

# E.5 Let's try with IMPUTATION replacing with the median the Nan values
df.fillna(df.median(), inplace=True)
#I make again the check with the info to see that there are 150 non null values not like the info before
#that on some columns were 147 non null or 149
df.info()
print ("\n")

# STEP .3 EXPLORATORY DATA ANALYSIS

#We have to understand previously the columns to take into account if all or just the ones we are interested in
#As regards the distinction between numerical and categorical data we have to understand which correlation
#analysis to do ...

