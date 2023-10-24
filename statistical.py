import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
import statistics

df_scores = pd.read_csv("data.csv")

#LET'S PERFORM STATISTICAL ANALYSIS

n_cluster=3 #we found in last point
alpha_corrected = 0.05/n_cluster #defined coefficient to perform bonferroni correction

#First of all examine nominal variables: marital and gender

#Let's create the contingency table for marital with clusters
contingency_table_marital= pd.crosstab(df_scores['marital'],df_scores['Cluster'])
print(contingency_table_marital.to_string())

#We will save the values of p in the vector p_values if it would be useful for results to write in the report ->
'''WE WILL SEEE IT'''
p_values=[]
p_index=0 #index 0 in which we will save the ones of marital; to index 1 we will save the ones of gender

#We do Chi Square and not Fisher because Chi square is suitable only for (2x2) while for multiple variables is used Chi
#How it works Chi square
#Hyphothesis H0: there are no differences between the cluster and the nominal variable
#If p<0.05/n I reject H0 -> there are differences
# If p>0.05/n I accept H0 -> there are no differences -> so the nominal variabile does not discriminate among clusters
#which means that we will not use it

#Aplly chi square
chi2, p, _, _, = chi2_contingency(contingency_table_marital)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}")

#Define a function that computes the result of chi square
def different(p_value,alpha):
    if p_value< alpha:
        print(f'There ARE differeces p_value: { p_value} < alpha: {alpha}')
    else:
        print(f'There ARE NOT difference p_value: { p_value} > alpha: {alpha})')

different(p_values[p_index],alpha_corrected)

'''Values are reported as median (25th-75th) for continuous variable, % for binary variables, and mode for nominal variables!!!
-> FROM TAURO REPORT TO DO ON OUR TO CREATE TABLE AT PAGE 16 OF PPT3'''

#Since this is a nominal -> we evaluate the frequency from the contingency table
# we have single for cluster 0 (28), married for cluster 1 (24), and also for cluster 2 (31)


#Let's do the same for gender
p_index +=1

contingency_table_gender= pd.crosstab(df_scores['gender'],df_scores['Cluster'])
print("\n\n",contingency_table_gender.to_string())

chi2, p, _, _, = chi2_contingency(contingency_table_gender)
p_values=p_values+[p]
print(f"\nchi: { chi2} ,p : {p_values[p_index]}")

different(p_values[p_index],alpha_corrected)
#Fpr example in this case we find that it doesn't discriminate which corresponds to the results we gained with EDA


#Now let's focus on ordinary and quantitaive variable, so all other elements of our dataframe

ordinal=['age', 'education', 'income', 'phq_score', 'gad_score', 'heas_score', 'eheals_score', 'ccs_score'] #these are the attributes column
#we are interested in examining

#Let's do for each column KRUSKAL-WALLIS TEST and PAIRWISE MANN-WHITNEY U TESTS
for att in ordinal:
    print(f'\n\n{att}:')
    cluster_0 = df_scores[df_scores['Cluster'] == 0 ][att] #we take only the elements of the attribute column selected from for cycle which have cluster =0
    value_0= statistics.median(cluster_0)
    value_0=round(value_0)
    print(value_0)
    cluster_1 = df_scores[df_scores['Cluster'] == 1 ][att] #same but the ones with have cluster =1
    value_1= statistics.median(cluster_1)
    value_1=round(value_1)
    print(value_1)
    cluster_2 = df_scores[df_scores['Cluster'] == 2 ][att] # cluster =2
    value_2= statistics.median(cluster_2)
    value_2=round(value_2)
    print(value_2)

    ### KRUSKAL WALLIS
    print('\nKruskal Wallis:')
    stat, p = kruskal([cluster_0], [cluster_1], [cluster_2], axis = 1)
    p_index = p_index + 1
    p_values = p_values + [p]
    different(p_values[p_index], 0.05)  # Krustal-Wallis doesn't want Bonferroni correction

    ### PAIRWISE MANN WHITNEY U
    # it is a pairwise -> I must consider all pairs

    # 0 - 1
    print('\nMann Whitney - clusters 0 - 1:')
    stat, p = mannwhitneyu(cluster_0, cluster_1)
    different(p, alpha_corrected)

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
# When we obtain for example in age that it differentiate for 0-1 , 0-2 but not for 1-2 it means that when creating our personas
# cluster 0 and cluster 1 not have the same age
# cluster 0 and cluster 2 not have the same age
# but cluster 1 and cluster 2 have the same age
# What is to compute
'''Values are reported as median (25th-75th) for continuous variable, % for binary variables, and mode for nominal variables!!!
-> FROM TAURO REPORT '''
# are the medians
# so we find the median age for cluster 0 , 1 and 2  and so on with the other numerical variables

