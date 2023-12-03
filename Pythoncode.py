import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
import statistics

#--------------------------------------------------------------------------------------------------------------

'''STEP 1: COLLECTION'''

#We input our data to Python and make it a DataFrame that we can work on
df = pd.read_csv("dataset_project_eHealth20232024.csv")

#The following instruction provides the essential details about the dataset, such as the number of rows and
#columns, number of non-null values, type of data of each column and the memory usage of the DataFrame.
df.info()
print("\n")

#--------------------------------------------------------------------------------------------------------------

'''STEP 2: CLEANING'''

# data problems:
#  Duplicate rows
#  Outliers
#  Missing data

#Firstly we observe that we have to pay attention to responses to ccs questionnaire which were evaluated reversly.
# The columns reverse scored are:
columns_to_modify = ['ccs_3', 'ccs_6', 'ccs_7', 'ccs_12']
# Create a dictionary to map values:
values = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}
# replace
df[columns_to_modify] = df[columns_to_modify].replace(values)


''' Let's deal with duplicate rows'''
df = df.drop_duplicates()
df.info() #Printing we see that the number of rows goes from 160 to 150 so 10 duplicates dropped
print("\nCOMPLETE DATASET WITHOUT DUPLICATES ROW\n",df.to_string())
print("\n")


''' Let's deal with outliers'''

#We will distinguish between :
#nominal variables for which we will do analysis through a bar plot -> which are GENDER and MARITAL
#from numerical and ordinal where do the IQR
#For education and the scores, that are categorical ordinal, is common to treat them as numerical!

# Identify the columns to exclude
nominal_cols = ['gender', 'marital'] #the categorical nominal that we will treat later

# Calculate IQR and identify outliers for the remaining columns
# define the remaining columns:
columns_without_nominal = [col for col in df.columns if col not in nominal_cols]

#Function to find the outliers for each column passed with IQR method
def find_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column < lower_bound) | (column > upper_bound)]
#The function returns a subset of the column that includes only the values that are less than lower_bound
#or greater than upper_bound.
#These values are considered outliers according to the IQR method.

# Create a list to store outliers for each column
outliers_list = []
# create also a list where to save columns name with outlier that will be useful for winsorizing
outliers_column= []

# Loop through all numerical columns to apply the function that find outlier
for column_name in columns_without_nominal:
    outliers = find_outliers_iqr(df[column_name]) #call the previous function for each column in dataframe to find outliers
    if not outliers.empty:  #Since the for cicle goes for all columns of numerical dataframe, for the ones that the vector returned from the function is not empty
        outliers_column.append(column_name) #save the column which has the outlier; this we will use in winsorizing
        outliers_list.append((column_name, outliers)) #if there are we create a list whose elements are
        #couples of variables containing the column that we found has the outlier and the respective outliers

# Print outliers for columns that have outliers
for column_name, outliers in outliers_list:
    print("\nOutliers in column {}: \n{}".format(column_name, outliers)) #I print the column
    # where I find them and which are; these are the couples saved in outliers_list at previous step
    #Also I print them GRAPHICALLY with histograms
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=column_name, color='skyblue', bins=20)
    plt.title(f'Histogram for {column_name}')
    plt.xlabel(column_name)
    plt.show()

#For the nominal columns excluded instead use the bar plots to understand if outliers
for column_name in nominal_cols:
    sns.barplot(x= df[column_name].unique(), y=df[column_name].value_counts())
    plt.title(f'Barplot for {column_name}')
    plt.xlabel(column_name)
    plt.show()

# From the graphs obtained we will perform WINSORIZING, which means that data that are outliers
# will assume lower and upper bound and so be no more outliers. This strategy is so not to lose
# all the other information about a candidate

# We do it only for numerical and ordinal, for nominal we saw not outliers!

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
for column in outliers_column:
    df[column] = winsorize_iqr(df[column])

print("\nComplete dataset with outliers solved\n",df.to_string())


'''# Let's deal with missing values;'''
# We choose the IMPUTATION replacing with the median the Nan values
# apart for the categorical nominal variables where replace with the mode

# In detail, the strategy is for the social values columns make the analysis to replace along the column,
# while the scores will be treated for rows for each group of score
# So consider for the selected row, the columns of the phq, and replace the nan in one of the phq's with the
# median among that phq's

#Let's do for social values (so analysis for columns)
# Replace NaN values with the mode for the categorical nominal columns
for col in nominal_cols:
    mode_value = df[col].mode().values[0]
    df[col].fillna(mode_value, inplace=True) #fill na is the function that fills Nan value, in this case with the mode computed
# Replace NaN values with the median for the numerical social columns
numerical_social_cols = ['age', 'education', 'income']
for col in numerical_social_cols:
    median_value = np.nanmedian(df[col]) #in the specified column compute the median among that elements without taking into consideration the nan
    median_value = round(median_value) #round the median otherwise we can obtain values not accetable like 13.5 to education which has no meaning in questionnaires
    df[col].fillna(median_value, inplace=True) #fill Nan with the median computes
df.info() #check that the first 5 rows now have 150 non null values
print("\n")

#Now let's focus on the columns on each questionnaire
# take a row consider for example the group of phq, find the nan and fill it with the median of these elements
# do the same for the other groups gad, eheals, heas and ccs
# and iterate for each row of the dataframe

#Define a function to replace the Nan with the median:
def replace_nans_with_median(row, start_idx, end_idx):
    selected_values = row.iloc[start_idx:end_idx]  # for the row passed select columns in the specified interval
    median = np.nanmedian(selected_values)  # Compute the median excluding the Nan
    for i in range(start_idx, end_idx): #loop for each column
        if np.isnan(row.iloc[i]): #if in that column thare is a nan
            row.iloc[i] = round(median) #replace it with the median computed again rounded as discussed before
    return row

#Define the intervals of columns of each score
column_ranges_questions = [(5,14), (14, 21), (21, 29), (29, 42), (42, 54)]
#columns form 5 to 14 are phq, 14 to 21 gad and so on...

#Apply the function to each row of the DataFrame for each interval of column
for start_idx, end_idx in column_ranges_questions:
    df = df.apply(lambda row: replace_nans_with_median(row, start_idx, end_idx), axis=1)

print("\nDataset with Nan solved\n",df.to_string())# print out all to check the algorithm was correct

#----------------------------------------------------------------------------------------------------------

'''STEP 3 : EXPLORATORY DATA ANALYSIS'''

# To simplify the Exploratory data Analysis
# given a row we sum all scores for each group (phq, gad, eheals, heas, ccs)
# and evaluate the total score to each questionnaire

# Create the vector of the names of new columns of scores
column_name_scores = ['phq_score', 'gad_score', 'eheals_score', 'heas_score', 'ccs_score']
# Create new columns in the DataFrame with names from column_name_scores
for i, (start, end) in enumerate(column_ranges_questions): #for cycle that runs for each group of columns (phq, gad .. as defined before)
    new_column_name = column_name_scores[i] #give the new name of the column from the vector above defined
    selected_columns = df.iloc[:, start:end] #select the group of columns
    df[new_column_name] = selected_columns.sum(axis=1) #sum all values in the selected colum along the same row
    #and do it for each each row
print("\nDataset with the score columns added in the end\n",df.to_string()) #print the dataframe to which was added the columnss of scores

'''MONOVARIATE EXPORATORY DATA ANALYSIS'''
#Firstly  we use the function df.describe()
# That function provides summary statistics for all data belonging to numerical datatypes such as int or float.
# The EDA is performed on the dataset with the scores apart from single values of questionnaire that would be impossible
df_scores = df.drop(columns=df.columns[5:54]) #so I create this new dataframe df_score
#that has only the social columns and the score columns
print('\n DF SCORE \n', df_scores.to_string())

# We choose not to consider in this analysis the nominal variable for which the median or quartiles would not make sense
df_analysis = df_scores.drop(nominal_cols, axis=1) #in this manner i define a new dataframe called df_analysis
# so to drop all columns we do not want
print(df_analysis.describe().to_string())
#Evaluations are on the report which takes the median, 25th and 75th percentiles and the maximum of the scores

#Let's have an overview of all scores

# Create a subplot of 5 graphs so to see the 5 scores
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
# Design the histogram for the score columns of the dataframe
bin_rng = [(0,27), (0,21), (8,40), (0,39), (0,72)]
for i, col in enumerate(df_analysis.columns[3:8]):
    sns.histplot(data=df_analysis, x=col, ax=axes[i], binwidth=1, binrange=bin_rng[i])
    axes[i].set_title(f'Histogram of {col}')
# Add spaces between graphs for better view and show
plt.tight_layout()
plt.show()

bin_wdt = [1, 1, 1000]
#also I create subplots in which visualize the histograms of the numerical social variables
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
for i, col in enumerate(df_analysis.columns[0:3]):
    sns.histplot(data=df_analysis, x=col, ax=axes[i], binwidth=bin_wdt[i])
    axes[i].set_title(f'Histogram of {col}')
plt.show()

#The barplots for nominals have already been done in the outliers analysis


#BI/MULTIVARIATE

#Now the objective is to analize relationships between the social values with the scores
#For the numerical variables we will use the heatmap to analize correlations
#but for the nominals we will use pairplot

#The strategy is to make a pair plot varying each nominal variable
#The nominal variables are: marital and gender
#So we create new dataframes that will preserve only one each time
df_gender = df_scores.drop(columns = 'marital') #here is preserved the gender and dropped marital
df_marital = df_scores.drop(columns = 'gender') #viceversa

sns.pairplot(df_gender, hue='gender')
plt.show()
sns.pairplot(df_marital, hue='marital')
plt.show()

# For the numericals, to analyse correlations between each numerical variable,
# we have the dataframe df_analysis already defined above
sns.heatmap(df_analysis.corr(), annot=True)
plt.title("Heatmap for correlations")
plt.show()

#--------------------------------------------------------------------------------------------------------------

'4. DATA ANALYSIS'

'4.1 DATA PREPARATION'

# Let's take again the original dataframe without the scores for this part
df = df.drop(columns=column_name_scores,axis=1) #so drop the scores column i have added


df = df.reset_index(drop=True) #With this command i reset the indexes of the dataframe
# because when we dropped the duplicate rows the original indexes were maintained
# so for example if 153 was shifted the resulting dataframe
# was still at 159 with 153 missing, in this way instead we rescale correctly the indexes from 0 to 149.
# This is done otherwise we will have problems with nexts concatenations


# For the nominal variables we have to perform the ONE HOT ENCODING

# Consider a dataframe only of  'gender' and 'marital' columns, which are the nominal variables
# on which performing one hot encoding

df_only_numerical=df.drop(columns=nominal_cols,axis=1) #dataframe without gender and marital
df_only_categorical = df[['gender', 'marital']] #dataframe only with gender and marital
print("\Dataset with only categorical\n", df_only_categorical.to_string())
print("\nDataset with only numerical\n", df_only_numerical.to_string())


encoder = OneHotEncoder(handle_unknown='ignore') #create an object encoder

# Reshape 'gender' and 'marital' columns to do encoding
gender_data = df.loc[:, 'gender'].values.reshape(-1, 1)
marital_data = df.loc[:, 'marital'].values.reshape(-1, 1)

#Consider gender
encoder.fit(gender_data) #fit
dummy_gender = encoder.transform(gender_data).toarray() #transform, dummies
dummy_gender = pd.DataFrame(dummy_gender, columns=encoder.get_feature_names_out(['gender']))

#The same for marital
encoder.fit(marital_data)
dummy_marital = encoder.transform(marital_data).toarray()
dummy_marital = pd.DataFrame(dummy_marital, columns=encoder.get_feature_names_out(['marital']))

df_conc = pd.concat([dummy_gender,dummy_marital, df_only_numerical], axis=1)
#new dataframe, with dummy of gender and marital (encoded) + the original with only the numericals

print("\nDataset with one hot enconding\n",df_conc.to_string()) #print to see the dataframe after one hot encoding


#STANDARDIZE
scaler = StandardScaler(copy=False)
scaler.fit(df_conc.astype(float))
scaler.transform(df_conc.astype(float))
df_scaled = pd.DataFrame(scaler.transform(df_conc.astype(float)))


#PCA
#PCA is the most commonly used data reduction method. It works on numerical data
#and binary data only.
#Thanks to one got encoding we have binary for nominals while
#age income are numerical
#and education + questionnaires questionnaires are categorical ordinal
#that we treat as numerical

pca_1 = PCA()
pca_1.fit(df_scaled)
df_pca = pd.DataFrame(pca_1.transform(df_scaled))
explained_variance = pd.DataFrame(pca_1.explained_variance_ratio_).transpose()
ax = sns.barplot(data=explained_variance)
plt.title("Explained variance")
plt.show()
cum_explained_variance = np.cumsum(pca_1.explained_variance_ratio_)
cum_explained_variance = pd.DataFrame(cum_explained_variance).transpose()
mx = sns.barplot(data=cum_explained_variance)
mx.axhline(0.75)
plt.title("Cumulative explained variance")
plt.show()

# Each principal component obtained  explains a certain percentage of
# variance within the dataset. Most times the focus is not on the number of principal
# components, but on the amount of variance explained
# Reduced datasets should explain a percentage of variance from 70% to 90% of the original dataset.
# We chose above the threshold of 75%
# from the 0.75 seen in last plot threshold we select the 1st 23 principal components
# So
df_pca = df_pca.iloc[:,0:23]


# 4.2 CLUSTERING

#K-medoids

# First of all we have to understand which is the number of clusters.
# To do so we use the graphical and the analytical method.

#Calculating the total within sum of square distances (inertia) for a varying number of clusters,
# it is possible to graphically identify the elbow in the figure.
# The number of clusters where the elbow is found is the optimal number of clusters
# to be used.
distortions = []  # Empty list
for i in range(1, 10):
    max_iter = 0 if i <= 2 else 300  # Set max_iter to 0 for i <= 2, 300 otherwise
    #strategic choice to fix max iter otherwise fixing from range 1 without this row and max iter =300
    #we have the warning that range has to start from 2
    km = KMedoids(n_clusters=i, metric='euclidean', method='pam', init='random', max_iter=max_iter, random_state=123)
    km.fit(df_pca)
    distortions.append(km.inertia_)

# The average silhouette is a measure of how similar an object is to its cluster,
# compared to other clusters in the same partition.
# The higher the value, the better.
silhouette_scores = []
for i in range(2, 10):
    km = KMedoids(n_clusters=i, metric='euclidean', method='pam', init='random', max_iter=300,
                      random_state=123)
    y_km = km.fit_predict(df_pca)
    silhouette_scores.append(silhouette_score(df_pca, y_km))

#We plot both:

plt.plot(range(1,10), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(2,10), silhouette_scores, marker='o')
plt.title("Silhouette Score for KMedoids Clustering")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

#We plotted the graphical method and the analytical method
#We saw from the graphical the elbow at three while the analitical is max at 2 clusters
#Since the difference between 2 and 3 of the silhouette is minimal we choose 3 clusters

#So we apply k-Medoids for 3 clusters:

km = KMedoids(n_clusters=3, metric='euclidean', method='pam', init='random', max_iter=300, random_state=123)
# abbiamo fatto un controllo con un ciclo for per vedere che i medoidi che uscissero fossero corretti
# e non cadessimo in un minimo locale! basta prima del km=KMedoids fare un ciclo for i in range (1,10)
# e tutte le funzioni di seguito insieme con i grafici dentro questo ciclo for
km.fit(df_pca)
y_km= km.predict(df_pca)

# Plot data points for each cluster
plt.scatter(df_pca.iloc[y_km==0,0],df_pca.iloc[y_km==0,1], s=50, c='green', marker='o')
plt.scatter(df_pca.iloc[y_km==1,0],df_pca.iloc[y_km==1,1], s=50, c='orange', marker='+')
plt.scatter(df_pca.iloc[y_km==2,0],df_pca.iloc[y_km==2,1], s=50, c='blue', marker='*')
# Plot cluster centroids
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, c='red', marker='x', label='Medoids')
plt.legend()
plt.grid()
plt.show()

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot data points for each cluster in 3D
ax.scatter(df_pca.iloc[y_km==0, 0], df_pca.iloc[y_km==0, 1], df_pca.iloc[y_km==0, 2], s=50, c='green', marker='o', label='Cluster 0')
ax.scatter(df_pca.iloc[y_km==1, 0], df_pca.iloc[y_km==1, 1], df_pca.iloc[y_km==1, 2], s=50, c='orange', marker='+', label='Cluster 1')
ax.scatter(df_pca.iloc[y_km==2, 0], df_pca.iloc[y_km==2, 1], df_pca.iloc[y_km==2, 2], s=50, c='blue', marker='*', label='Cluster 2')
# Plot cluster centroids in 3D
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 2], s=250, c='red', marker='x', label='Medoids')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.legend()
plt.grid()
plt.show()

#We define a new dataframe with the label obtained
df_labeled=df #The new dataframe is given from the original one
df_labeled['Cluster']=y_km #and is added an additional column at the end which represents the clusters obtained
print("\n Complete dataframe with last column indicating cluster belonging\n",df_labeled.to_string())


'4.2 STATISTICAL ANALYSIS'
#In this case we choose to do it with a dataframe
#that contains all social values , the resulting scores of the questionnaires and the cluster column obtained
#As for the EDA we tought would not be effective to do it for each single result of the questionnaire
#So we take again the dataframe with social columns and scores we used before (df_scores)
# and we add the cluster column
df_scores['Cluster']=y_km

n_cluster=3 #we found in last point
alpha_corrected = 0.05/n_cluster #defined coefficient to perform bonferroni correction

#Let's start with gender and marital
#Since this is a nominal -> we evaluate the frequency from the contingency table

contingency_table_marital= pd.crosstab(df_scores['marital'],df_scores['Cluster'])
print("\n\nContingency table marital\n",contingency_table_marital.to_string())

# From the contingency table we have that the hypothesis of Chi square of marital are not satisfied (value in the table >5)
# For this reason so Fisher test was required.
# Since on Python could be performed only 2x2, Pairwise Fisher-square test was developed in R, with Bonferroni correction


# Instead for gender we do Chi Square suitable for multiple variables and not 2x2
# How it works Chi square
# Hyphothesis H0: Variable 1 and variable 2 are not related in the population; the proportions of variable 1 are
# the same for different values of variable 2
# If p<0.05/n I reject H0 -> there are not the same ->it discriminates
# If p>0.05/n I accept H0 -> there are the same -> so the nominal variable does not discriminate among
# clusters, which means that we will not use it


#We will save the values of p in the vector p_values if it would be useful for results to write in the report ->
p_values=[]
p_index=0 #index 0 in which we will save the ones of marital; to index 1 we will save the ones of gender


#Define a function that computes the result of chi square
def different(p_value,alpha):
    if p_value < alpha:
        print(f'There ARE differeces p_value: { p_value} < alpha: {alpha}')
    else:
        print(f'There ARE NOT difference p_value: { p_value} > alpha: {alpha})')


#Let's do for gender

contingency_table_gender= pd.crosstab(df_scores['gender'],df_scores['Cluster'])
print("\nContingency table gender\n",contingency_table_gender.to_string())

chi2, p, _, _, = chi2_contingency(contingency_table_gender)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}\n")
different(p_values[p_index],0.05)

#We apply in general the Chi square to the whole, but in reality we have to perform a pairwise:

#Pairwise cluster 0/1

contingency_subtable_gender_0_1 = contingency_table_gender.iloc[:,[0,1]]
print(contingency_subtable_gender_0_1)

p_index +=1

chi2, p, _, _, = chi2_contingency(contingency_subtable_gender_0_1)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}\n")
different(p_values[p_index],alpha_corrected)

#Pairwise cluster 0/2

contingency_subtable_gender_0_2= contingency_table_gender.iloc[:,[0,2]]
print(contingency_subtable_gender_0_2)

p_index +=1

chi2, p, _, _, = chi2_contingency(contingency_subtable_gender_0_2)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}\n")
different(p_values[p_index],alpha_corrected)

#Pairwise cluster 1/2

contingency_subtable_gender_1_2= contingency_table_gender.iloc[:,[1,2]]
print(contingency_subtable_gender_1_2)

p_index +=1

chi2, p, _, _, = chi2_contingency(contingency_subtable_gender_1_2)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}\n")
different(p_values[p_index],alpha_corrected)

#For example in this case we find that it doesn't discriminate which corresponds to the results we gained with EDA

#Values in the final table fo the statistical analysis are reported as mode for nominal variables!!!
#Therefore we will look to the mode for each cluster in the contingency table of gender and marital
#to discriminate it for each cluster


#Now let's focus on ordinary and quantitative variable, so all other elements of our dataframe

ordinal = ['age', 'education', 'income', 'phq_score', 'gad_score', 'eheals_score', 'heas_score', 'ccs_score']

#Let's do for each column KRUSKAL-WALLIS TEST and PAIRWISE MANN-WHITNEY U TESTS
for att in ordinal:
    print(f'\n\n{att}:')
    cluster_0 = df_scores[df_scores['Cluster'] == 0 ][att] #we take only the elements of the attribute
    # column selected from for cycle which have cluster = 0, i.e. if is age we take only the elements in the
    #dataframe of column age that have the cluster =0

    # What is to compute for discriminative variables?
    # Values are reported as median (25th-75th) for continuous variable in the final statistical analysis table
    # so we find the median age for cluster 0 , 1 and 2  and so on with the other numerical variables, together with the percentiles

    value_0 = statistics.median(cluster_0) #we compute the median on this column
    value_0 = round(value_0) #we round it because with score otherwise we will obtain no sense values
    q1_0 = np.percentile(cluster_0, 25) #we compute the percentiles, useful in the construction of the statistical table
    q3_0 = np.percentile(cluster_0, 75)
    print(f'Cluster 0 -> median: {value_0}, 1st percentiles {q1_0}, 3rd percentile: {q3_0}')
    cluster_1 = df_scores[df_scores['Cluster'] == 1 ][att] #same but the ones with have cluster =1
    value_1 = statistics.median(cluster_1)
    value_1= round(value_1)
    q1_1 = np.percentile(cluster_1, 25)
    q3_1 = np.percentile(cluster_1, 75)
    print(f'Cluster 1 -> median: {value_1}, 1st percentiles {q1_1}, 3rd percentile: {q3_1}')
    cluster_2 = df_scores[df_scores['Cluster'] == 2 ][att] # cluster =2
    value_2 = statistics.median(cluster_2)
    value_2 = round(value_2)
    q1_2 = np.percentile(cluster_2, 25)
    q3_2 = np.percentile(cluster_2, 75)
    print(f'Cluster 2 -> median: {value_2}, 1st percentiles {q1_2}, 3rd percentile: {q3_2}')

    ### KRUSKAL WALLIS
    #Hypothesis H0: all groups come from the same distribution
    #so if we reject it (p<0.05 in this case no bonferroni) it means come from different so we can
    #use the same function of before to read results
    print('\nKruskal Wallis:')
    #Perform kruskal wallis
    stat, p = kruskal([cluster_0], [cluster_1], [cluster_2], axis = 1)
    p_index = p_index + 1
    p_values = p_values + [p]
    #call the previous function to discriminate the result with values found
    different(p_values[p_index], 0.05)  # Krustal-Wallis doesn't want Bonferroni correction

    ### PAIRWISE MANN WHITNEY U
    # it is a pairwise -> I must consider all pairs cluster 0-1, 1-2, 0-2
    #Hyphothesis H0: the two have the identical distribution
    #rejecting (p<0.05/n) are different and so discrimimate
    #also in this case we call the function of before

    # 0 - 1
    print('\nMann Whitney - clusters 0 - 1:')
    #Perform mannwhitney u
    stat, p = mannwhitneyu(cluster_0, cluster_1)
    #call the function for discriminate
    different(p, alpha_corrected)

    #Do the same for other couples of clusters
    # 0 - 2
    print('\nMann Whitney - clusters 0 - 2:')
    stat, p = mannwhitneyu(cluster_0, cluster_2)
    different(p, alpha_corrected)

    # 1 - 2
    print('\nMann Whitney - clusters 1 - 2:')
    stat, p = mannwhitneyu(cluster_1, cluster_2)
    different(p, alpha_corrected)


# Kruskal wallis gives a general evaluation
# while pairwise Mann Whitney is more detailed outlining where are differences
# When we obtain for example in age that it differentiate for 0-1 , 0-2 but not for 1-2
# it means that when creating our personas
# cluster 0 and cluster 1 not have the same age
# cluster 0 and cluster 2 not have the same age
# but cluster 1 and cluster 2 have the same age
# and so on

#So from all this, we can build our final table of statistical analysis








