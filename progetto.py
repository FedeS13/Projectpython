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
print("\n")

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

#Firstly we observe that we have to pay attention to responses to ccs questionnaire which were evaluated reversly.
# The columns reverse scored are:
columns_to_modify = ['ccs_3', 'ccs_6', 'ccs_7', 'ccs_12']
# Create a dictionary to map values:
values = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}
# replace
df[columns_to_modify] = df[columns_to_modify].replace(values)

# B. let's deal with duplicate rows
df = df.drop_duplicates()
df.info() #Printing we see that the number of rows goes from 160 to 150 so 10 duplicates dropped
print("\n")


#D. Let's deal with outliers -> Find the outliers in the columns

#We will distinguish between nominal variables for which we will do a frequency analysis
#from numerical and ordinal where do the IQR

# Identify the columns to exclude
nominal_cols = ['gender', 'marital'] #the categorical nominal that we will treat later
#the education like also the scores that are categorical ordinal he said is correct to treat them as numbers!
# Calculate IQR and identify outliers for the remaining columns
columns_without_nominal = [col for col in df.columns if col not in nominal_cols]
def find_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column < lower_bound) | (column > upper_bound)]

# Create a list to store outliers for each column
outliers_list = []
#create also a list where to save only columns that will be useful for winsorizing
outliers_column= []

# Loop through each column and find outliers
for column_name in columns_without_nominal:
    outliers = find_outliers_iqr(df[column_name]) #call the previous function for each column in dataframe to find outliers
    if not outliers.empty:  # Check if there are outliers
        outliers_column.append(column_name) #this we will use in winsorizing
        outliers_list.append((column_name, outliers)) #if there are we are in the loop and
        # create a list with the column and the respective outliers

# Print outliers for columns that have outliers
for column_name, outliers in outliers_list:
    print("\nOutliers in column {}: \n{}".format(column_name, outliers)) #I visualize it below + also graphically
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=column_name, color='skyblue', bins=50) #histogram to see distribution
    plt.title(f'Histogram for {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

#For the nominal columns excluded instead use the frequencies distribution to understand if outliers
for column_name in nominal_cols:
    sns.histplot(df[column_name], bins=50)
    plt.title(f'Histogram for {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

#From the graphs obtained we will perform winsorizing,which means that data that are out of ranges
#will assume lower and upper bound and so be no more outliers. This strategy is so not to lose all the other information
#about a candidate (we do it only for numerical and ordinal , for nominal chosen not outliers)

def winsorize_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace values outside the IQR range with the nearest values within the range
    column = np.where(column < lower_bound, lower_bound, column)
    column = np.where(column > upper_bound, upper_bound, column)

    return column
# Apply Winsorizing based on IQR to the columns identified as outliers stored in outliers_column
for column in columns_without_nominal:
    df[column] = winsorize_iqr(df[column])

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
# In our dataset the categorical nominal variables are the gender, the marital status
# In detail, the strategy is to replace for columns only for the social values,
# while the scores will be treated for rows

#Let's do for social values
# Replace NaN values with the mode for the categorical nominal columns
for col in nominal_cols:
    mode_value = df[col].mode().values[0]
    df[col].fillna(mode_value, inplace=True)
# Replace NaN values with the median for the other social columns
median_cols = ['age', 'education', 'income']
for col in median_cols:
    median_value = df[col].median()
    median_value = round(median_value)
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

#Define the intervals of columns of each score
column_ranges_questions = [(5,14), (14, 21), (21, 29), (29, 42), (42, 54)]

#Apply the function to each row of the DataFrame for each interval of column
for start_idx, end_idx in column_ranges_questions:
    df = df.apply(lambda row: replace_nans_with_median(row, start_idx, end_idx), axis=1)

print(df.to_string())# I printed out all just to check was correct this algorithm


#After having performed OUTLIERS NOW WE EVALUATE SCORES
# So to simplify the Exploratory data Analysis we sum all scores of each
#I create the vector of the names of new columns of scores
column_name_scores = ['phq_score', 'gad_score', 'heas_score', 'eheals_score', 'ccs_score']
# Create new columns in the DataFrame with names from column_name
for i, (start, end) in enumerate(column_ranges_questions):
    new_column_name = column_name_scores[i]
    selected_columns = df.iloc[:, start:end]
    df[new_column_name] = selected_columns.sum(axis=1)
print(df.to_string())

'''In origine avevamo pensato di fare fede alle tabelle degli score o crearle per vedere se la popolazione era sana o no 
ma questo approccio non andava bene. Le funzioni qui sotto erano le codifiche scelte per le valutazioni con i questionari. 
Tra l'altro Tauro ci ha detto che per i gad e phq poteva andare bene perchè avevamo le tabelle ma poi per gli altri
dovevamo tracciare la distribuzione della popolazione con le somme e stabilire eventualmente noi sulla base della popolazione 
dei target. -> Sicomme non dobbiamo diagnosticare abbiamo scelto di evitarlo '''
'''FOR PHQ
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

df['gad_score'] = df.iloc[:, 14: 21].sum(axis=1)

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


df['heas_score'] = df.iloc[:, 21: 29].sum(axis=1)
df['eheals_score'] = df.iloc[:, 29: 42].sum(axis=1)
#As regards the eheals and heals the coding is another
#and for both is the same since maximum score of eheals
#while for heals is 39 so use the same subdivision
#in 3 scales (eheals will just have a surplus element ut the subdivision is coherent)
#def assign_value(value):
    #if 0 <= value <= 12:
        #return 0
    #elif 13 <= value <= 25:
        #return 1
    #else:
        #return 2

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
df['ccs_score'] = df['ccs_score'].apply(assign_value_ccs) '''

#3. EXPLORATORY DATA ANALYSIS

#Firstly to do the EDA Univariate Analysis we use the function df.describe()
# That function provides summary statistics for all data belonging to numerical datatypes such as int or float.
#The EDA is performed only on the dataset with the scores apart from single values of questionnaire that would be impossible
df_scores = df.drop(columns=df.columns[5:54]) #I create this new dataframe df_score
# We choose not to consider in this analysis the nominal variable for which the median or quartiles would not make sense
df_analysis = df_scores.drop(nominal_cols, axis=1) #in this manner i define a new dataframe called df_analysis
# so to drop all columns we do not want
print(df_analysis.describe().to_string())
#Evaluations are on the report which takes the median, 25th and 75th percentiles and the maximum of the scores

#let's have an overview of all scores
# Create a subplot of 5 graphs so to see the 5 scores
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
# Design the histogram for the score columns of the dataframe
bin_v = [27, 21, 40, 39, 72]
'''PERCHE' LI ABBIAMO MESSI COSI' SPIEGAZIONE ->SAMU '''
for i, col in enumerate(df_analysis.columns[3:8]):
    sns.histplot(data=df_analysis, x=col, ax=axes[i], bins=bin_v[i])
    axes[i].set_title(f'Histogram of {col}')
# Add spaces between graphs for better view and show
plt.tight_layout()
plt.show()

#Now the objective is to analize the correlation between the social values with the scores
#Fot the numerical variables we will use the heatmap to analize correlations
#but for the nominals we will use pairplot
#The strategy is to make a pair plot varying each nominal variable
#The nominal variables are: marital and gender
#So we create new dataframes that will preserve only one nominal
df_gender = df_scores.drop(columns = 'marital') #here is preserved the gender and dropped the other
df_marital = df_scores.drop(columns = 'gender') #here marital
df_numerical = df_scores.drop(['gender','marital'], axis=1) #and here i delete all nominals, so to make the heatmap only with numerical


sns.pairplot(df_gender, hue='gender')
plt.show()
sns.pairplot(df_marital, hue='marital')
plt.show()
sns.heatmap(df_numerical.corr(), annot=True)
plt.show()


'''DOBBIAMO SCRIVERE QUESTI RISULTATI NELLA ANALYSIS'''

#3. CLUSTER
#Riapriamo per fare i CLUSTER
#devo eliminare gli scores ->
'''HO PROVATO A CREARE IN REALTA' PRIMA UN NUOVO DATAFRAME QUANDO HO SOMMATO E AGGIUNTO GLI SCORES 
# #COSI' DA MANTENERE QUELLO ORIGINALE MA MI STAMPAVA CMQ IL TUTTO STRANO QUINDI ELIMINO E FACCIO UNO STEP IN PIU'''
df = df.drop(columns=column_name_scores,axis=1)
print(df.to_string())
