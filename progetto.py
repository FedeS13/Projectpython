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

#E.5 Let's try with IMPUTATION replacing with the median the Nan values apart from
# the categorical nominal variables where replace with the mode
# In our dataset the categorical nominal variables are the gender, the civil status
# In detail, the strategy is to replace for columns only for the social values,
# while the scores will be treated for rows

#Let's do for social values
# Replace NaN values with the mode for the categorical nominal columns
mode_cols = ['gender', 'marital']
for col in mode_cols:
    mode_value = df[col].mode().values[0]
    df[col].fillna(mode_value, inplace=True)
# Replace NaN values with the median for the other social columns
median_cols = ['age', 'education', 'income']
for col in median_cols:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)
df.info()
print("\n")

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

print(df.to_string()) # I printed out all just to check was correct this algorithm

#The first step of our analysis is to discriminate which is the problem of our population
#so understand which problem to treat, if phq,gad,heas,eheals or ccs to do so we sum all scores of each
# and analize which are the problems mostly evident in out population

#Before evaluating the scores we have to pay attention to responses to ccs questionnaire which were evaluated reversly.
# The columns reverse scored are:
columns_to_modify = ['ccs_3', 'ccs_6', 'ccs_7', 'ccs_12']
# Create a dictionary to map values:
values = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}
# replace
df[columns_to_modify] = df[columns_to_modify].replace(values)

#NOW LET'S COMPUTE THE SCORES OF COLUMNS
#FOR PHQ
# Select the columns from which compute the score
selected_columns = df.iloc[:, 5:14]
# Define a function to compute the value for each row
def calcola_valore_riga(row):
    count_2 = (row >= 2).sum()  # Count values >=2
    count_2_6_7 = (row.iloc[0:2] >= 2).sum()  # Count values >=2 in column 6 and column 7
    if count_2 >= 5 and count_2_6_7 >= 1:
        return 2
    elif 2 <= count_2 <= 4 and count_2_6_7 >= 1:
        return 1
    else:
        return 0
# Apply the function to each row and have a new column
df['phq_score_normalized'] = selected_columns.apply(calcola_valore_riga, axis=1)


#FOR GAD, HEAS AND EHEALS -> SUM ALL THE VALUES
df['gad_score'] = df.iloc[:, 14: 22].sum(axis=1)
df['heas_score'] = df.iloc[:, 22: 30].sum(axis=1)
df['eheals_score'] = df.iloc[:, 30: 43].sum(axis=1)

#A regards GAD we define a coding

def assign_value_gad(value):
    if 0 <= value <= 4:
        return 0
    elif 5 <= value <= 9:
        return 1
    elif 10 <= value <= 14:
        return 2
    else:
        return 3

# Apply the function to desider column
df['gad_score'] = df['gad_score'].apply(assign_value_gad)

#As regards the eheals and heals the coding is another
#and for both is the same since maximum score of eheals
#while for heals is 39 so use the same subdivision
#in 3 scales (eheals will just have a surplus element ut the subdivision is coherent)
def assign_value(value):
    if 0 <= value <= 12:
        return 0
    elif 13 <= value <= 25:
        return 1
    else:
        return 2

# Apply the function to desired column
df['heas_score'] = df['heas_score'].apply(assign_value)
df['eheals_score'] = df['eheals_score'].apply(assign_value)


# FOR CCS
# Create a new column with the mean of elements for column 43 to column 54 for each row of the dataframe
#For ccs as much is higher much you are skeptic
df['ccs_score'] = df.iloc[:, 43:55].mean(axis=1)

def assign_value_ccs(value):
    if 0 <= value <= 3:
        return 0
    else:
        return 1

# Apply function to desired column
df['ccs_score'] = df['ccs_score'].apply(assign_value_ccs)


#The score columns have been added at the end of the dataframe so if we eliminate all the single parameters
# columns, the ones with scores will shift to correct position
#Before deleting save a copy of the original dataframe
df2 = df #the original dataframe is saved in df2
df = df.drop(df.columns[5:54], axis=1) #drop columns

print(df.to_string())

#FOR THE OUTLIERS RESULTS WE HAVE NOT DECIDED WHAT TO DO
#THE PROBLEM WE WERE TALKING ABOUT WAS OUTLIERS OF CATEGORICAL DATA
#TO TREAT AS FREQUENCY ANALYSIS AND NOT IQR
#WE WERE DECIDING IF TO DO  DIRECTLY IN THE CLUSTER
#D. Let's deal with outliers -> Find the outliers in the columns
#def find_outliers_iqr(column):
    #Q1 = column.quantile(0.25)
    #Q3 = column.quantile(0.75)
    #IQR = Q3 - Q1
    #lower_bound = Q1 - 1.5 * IQR
    #upper_bound = Q3 + 1.5 * IQR
    #return column[(column < lower_bound) | (column > upper_bound)]

# Create a list to store outliers for each column
#outliers_list = []

# Loop through each column and find outliers
#for column_name in df.columns:
    #outliers = find_outliers_iqr(df[column_name]) #call the previous function for each column in dataframe to find outliers
    #if not outliers.empty:  # Check if there are outliers
        #outliers_list.append((column_name, outliers)) #if there are we are in the loop and
        # create a list with the column and the respective outliers

# Print outliers for columns that have outliers
#for column_name, outliers in outliers_list:
    #print("\nOutliers in column {}: \n{}".format(column_name, outliers)) #I visualize it below + also graphically
    #plt.figure(figsize=(6, 4))  # Adjust figsize as needed
    #df.boxplot(column=column_name) #boxplot to visualize outliers
    #plt.title(f'Box Plot for {column_name}')
    #plt.figure(figsize=(6, 4))
    #sns.histplot(data=df, x=column_name, kde=True, color='skyblue', bins=50) #histplot to see distribution
    #plt.title(f'Histplot for {column_name}')
    #plt.show()


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

#Now the objective is to analize the correlation between the social values with the scores
#Fot the numerical variables we will use the heatmap to analize correlations
#but for the categorical we will use pairplot
#The strategy is to make a pair plot varying each categorical variabel
#The categorical variables we have considered are: education, marital and gender
#So we create new dataframes that will preserve only one categorical
df_gender=df.drop(columns = ['education', 'marital']) #here is preserved the gender and dropped the other two
df_education=df.drop(columns = ['gender', 'marital']) #here education
df_marital=df.drop(columns = ['education', 'gender']) #here marital
df_numerical=df.drop(columns = ['education','gender','marital']) #and here i delete all categoricals, so to make the heatmap only with numerical

sns.pairplot(df_gender,hue='gender')
plt.title("Gender pairplot")
plt.show()
sns.pairplot(df_education,hue='education')
plt.title('education pairplot')
plt.show()
sns.pairplot(df_marital,hue='marital')
plt.title('marital pairplot')
plt.show()
sns.heatmap(df_numerical.corr(), annot=True)
plt.title('heatmap numerical variables')
plt.show()

# STEP .3 EXPLORATORY DATA ANALYSIS

#We have to understand previously the columns to take into account if all or just the ones we are interested in
#As regards the distinction between numerical and categorical data we have to understand which correlation
#analysis to do ...
