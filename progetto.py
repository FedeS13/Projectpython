import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# D. Outliers
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
# WOULD MEAN LOOSE TOO MUCH INFO SO IS NOT AN OPTION
# E.2 The one of machine learning, we have not seen it, he said we would have done a lesson on it but for now
# it's not an option
# E.3 The one on the interpolation has the problem on the choice of the number of k that we have seen for the
# cluster but not for that step so not
# E.4 the one on forward and backward fill I do not consider the best

# E.5 Let's try with IMPUTATION replacing with the median the Nan values
#In particular the strategy is to replace for columns only for the social values, while the scores will be treated
#for rows

#Let's do for social values
# Select the first 5 columns
columns_to_fill = df.columns[:5]
# Calculate the median for each column
medians = df[columns_to_fill].median()
# Replace NaN values with medians
df[columns_to_fill] = df[columns_to_fill].fillna(medians)

#Now let's focus on the columns for each parameter, replacing the Nan with the median for each group
#this means take a row consider the group of phq, find the nan and fill the median of these elements
#do the same for the other groups and iterate for each row of the dataframe
#Define a function to replace the Nan with the median: The median is rounded so to avoid decimal numbers
#not admissible for the possible response to questionnaire
def replace_nans_with_median(row, start_idx, end_idx):
    selected_values = row.iloc[start_idx:end_idx]  # Select columns in the specified interval
    median = np.nanmedian(selected_values)  # Compute the median excluding the Nan
    for i in range(start_idx, end_idx):
        if np.isnan(row.iloc[i]):
            row.iloc[i] = round(median)
    return row

#Define the intervals of columns
column_ranges = [(5,14), (14, 21), (21, 29), (29, 42), (42, 54)]

#Apply the function to each row of the DataFrame for each interval of column
for start_idx, end_idx in column_ranges:
    df = df.apply(lambda row: replace_nans_with_median(row, start_idx, end_idx), axis=1)

df.info()

print('\n',df.to_string()) # I printed out all just to check was correct this algorithm


#NOW LET'S COMPUTE THE SCORES ELIMINATING ALL OTHER COLUMNS
for start, end in column_ranges:
    # Create a name for the new column result of the sum
    column_name = f'sum_{start+1}_{end}'
    # Compute the sum of the elements in the interval of columns specified for each row
    sums = df.iloc[:, start: end].sum(axis=1)
    # Add the resulting column to dataframe
    df[column_name] = sums

#The columns with the previous for have been added at the end of the dataframe so eliminating all the single paramters columns
# the ones with scores they shift to correct position
df = df.drop(df.columns[5:54], axis=1)

#DO UNA MIGLIORE DENOMINAZIONE ALLE COLONNE, NON SAPEVO FARLO DA DENTRO A QUEL CICLO FOR DI PRIMA QUANDO LE HO
#CREATE
new_name_columns = ['Sum_phq', 'Sum_gad', 'Sum_eheals', 'Sum_heals', 'Sum_ccs']
# Rename the desired columns
df = df.rename(columns={columnn: new_name for columnn, new_name in zip(df.columns[5:10], new_name_columns)})
#Print the new dataframe changes
print('\nNEW DATAFRAME\n',df.to_string())


#D. Let's deal with outliers -> Find the outliers in the columns
def find_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column < lower_bound) | (column > upper_bound)]

# Create a list to store outliers for each column
outliers_list = []

# Loop through each column and find outliers
for column_name in df.columns:
    outliers = find_outliers_iqr(df[column_name]) #call the previous function for each column in dataframe
    # to find outliers
    if not outliers.empty:  # Check if there are outliers
        outliers_list.append((column_name, outliers)) #if there are we are in the loop and
        # create a list with the column and the respective outliers

# Print outliers for columns that have outliers
for column_name, outliers in outliers_list:
    print("\nOutliers in column {}: \n{}".format(column_name, outliers)) #I visualize it below + also graphically
    # Creare a subplot of 2 graphs
    plt.figure(figsize=(12, 5))  # Set dimension of the figure
    # 1st subplot is the boxplot to visualize the outliers
    plt.subplot(1, 2, 1)
    df.boxplot(column=column_name) #boxplot to visualize outliers
    plt.title(f'Box Plot for {column_name}')
    # 2nd subplot is instead the histogram and distibution to understand why we obtained that value
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=column_name, kde=True, color='skyblue', bins=50) #histplot to see distribution
    plt.title(f'Histplot for {column_name}')
    # Show the subplot
    plt.tight_layout()
    plt.show()

#let's have  also an overview of all scores
# Create a subplot of 5 graphs so to see the 5 scores
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
# Design the histplot for the score columns of the dataframe
for i, col in enumerate(df.columns[5:10]):
    sns.histplot(data=df, x=col, ax=axes[i], kde=True, bins=50)
    axes[i].set_title(f'Histogram of {col}')
# Add spaces between graphs for better view and show
plt.tight_layout()
plt.show()


# STEP .3 EXPLORATORY DATA ANALYSIS
#We have to understand previously the columns to take into account if all or just the ones we are interested in
#As regards the distinction between numerical and categorical data we have to understand which correlation
#analysis to do ...


