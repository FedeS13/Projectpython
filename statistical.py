import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
import statistics
import numpy as np

df_scores = pd.read_csv("data.csv")

#LET'S PERFORM STATISTICAL ANALYSIS

n_cluster=3 #we found in last point
alpha_corrected = 0.05/n_cluster #defined coefficient to perform bonferroni correction

contingency_table_marital= pd.crosstab(df_scores['marital'],df_scores['Cluster'])
print("\n\n",contingency_table_marital.to_string())

#First of all examine nominal variables: marital and gender

#We do Chi Square and not Fisher because Chi square is suitable only for (2x2) while for multiple variables is
# used Chi
#How it works Chi square
#Hyphothesis H0: Variable 1 and variable 2 are not related in the population; the proportions of variable 1 are
#the same for different values of variable 2
#If p<0.05/n I reject H0 -> there are not the same ->it discriminates
# If p>0.05/n I accept H0 -> there are the same -> so the nominal variable does not discriminate among
# clusters, which means that we will not use it


#We will save the values of p in the vector p_values if it would be useful for results to write in the report ->
'''WE WILL SEEE IT'''
p_values=[]
p_index=0 #index 0 in which we will save the ones of marital; to index 1 we will save the ones of gender


#Define a function that computes the result of chi square
def different(p_value,alpha):
    if p_value< alpha:
        print(f'There ARE differeces p_value: { p_value} < alpha: {alpha}')
    else:
        print(f'There ARE NOT difference p_value: { p_value} > alpha: {alpha})')


'''Values are reported as median (25th-75th) for continuous variable, % for binary variables, and mode for nominal variables!!!
-> FROM TAURO REPORT TO DO ON OUR TO CREATE TABLE AT PAGE 16 OF PPT3'''
#Since this is a nominal -> we evaluate the frequency from the contingency table

#Let's do for gender

contingency_table_gender= pd.crosstab(df_scores['gender'],df_scores['Cluster'])
print("\n\n",contingency_table_gender.to_string())

chi2, p, _, _, = chi2_contingency(contingency_table_gender)
p_values=p_values+[p]
print(f"\n0_1\nchi: { chi2} ,p : {p_values[p_index]}")
different(p_values[p_index],0.05)


contingency_subtable_gender_0_1= contingency_table_gender.iloc[:,[0,1]]
print(contingency_subtable_gender_0_1)

p_index +=1

chi2, p, _, _, = chi2_contingency(contingency_subtable_gender_0_1)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}")
different(p_values[p_index],alpha_corrected)

contingency_subtable_gender_0_2= contingency_table_gender.iloc[:,[0,2]]
print(contingency_subtable_gender_0_2)

p_index +=1

chi2, p, _, _, = chi2_contingency(contingency_subtable_gender_0_2)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}")
different(p_values[p_index],alpha_corrected)

contingency_subtable_gender_1_2= contingency_table_gender.iloc[:,[1,2]]
print(contingency_subtable_gender_1_2)

p_index +=1

chi2, p, _, _, = chi2_contingency(contingency_subtable_gender_1_2)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}")
different(p_values[p_index],alpha_corrected)

#For example in this case we find that it doesn't discriminate which corresponds to the results we gained with EDA


#Now let's focus on ordinary and quantitative variable, so all other elements of our dataframe

ordinal = ['age', 'education', 'income', 'phq_score', 'gad_score', 'eheals_score', 'heas_score', 'ccs_score']
#quando ritorni nell'altro script questo vettore è la somma di numerical_social_cols + column_name_scores
#these are the attributes column we are interested in examining

#Let's do for each column KRUSKAL-WALLIS TEST and PAIRWISE MANN-WHITNEY U TESTS
for att in ordinal:
    print(f'\n\n{att}:')
    cluster_0 = df_scores[df_scores['Cluster'] == 0 ][att] #we take only the elements of the attribute
    # column selected from for cycle which have cluster =0, i.e. if is age we take only the elements in the
    #dtaframe of column age that have the cluster =0

    # What is to compute for discriminative variables?
    '''Values are reported as median (25th-75th) for continuous variable, % for binary variables, and mode for nominal variables!!!
    -> FROM TAURO REPORT '''
    # are the medians and percentiles
    # so we find the median age for cluster 0 , 1 and 2  and so on with the other numerical variables

    value_0 = statistics.median(cluster_0) #we compute the median on this column
    value_0 = round(value_0) #we round it because with score otherwise we will obtain no sense values
    q1_0 = np.percentile(cluster_0, 25) #we compute the percentiles, useful in the construction of the statistical table
    #see tauro summary results
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
    #so if we reject it (p<0.05 in this case no bonferron) it means come from different so we can
    #use the same function of before to read results
    print('\nKruskal Wallis:')
    #Perform kruskal wallis
    stat, p = kruskal([cluster_0], [cluster_1], [cluster_2], axis = 1)
    '''CI SERVIRA' SALVARE I PVALUES? TANTO COMUNQUE LI STAMPIAMO E LA TABELLA LA COSTRUIAMO IN EXCEL??'''
    p_index = p_index + 1
    p_values = p_values + [p]
    #call the previous function to discriminate the result with values found
    different(p_values[p_index], 0.05)  # Krustal-Wallis doesn't want Bonferroni correction

    ### PAIRWISE MANN WHITNEY U
    # it is a pairwise -> I must consider all pairs cluster 0-1, 1-2, 0-2
    #Hyphothesis H0: the two have the identical distribution
    #rejecting (p<0.05/n) are different and so discrimimate
    #also in this case we call the function of before

    ''' QUI AD ESEMPIO NON RIUSCIAMO A SLAVARE I PVALUES AVENDO SCISSO I CLUSTER QUINDI NON POSSIAMo CANCELLARE
    lA COSA DI SALVARE I PVALUES? '''

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

#CREIAMO UN FILE EXCEL IN CUI DA QUESTI DATI RACCOGLIAMO I RISULTATI
'DOMANDA: MICA DOBBIAMO FARLO SU PYTHON PERCHè SAREBBE UN CAOS ?'


