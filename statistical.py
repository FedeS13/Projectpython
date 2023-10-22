import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

df_scores = pd.read_csv("data.csv")

#NOMINALI: GENDER/ MARITAL
n_cluster=3
alpha_corrected = 0.05/n_cluster

#Creiamo la contingency table per marital con i cluster
contingency_table_marital= pd.crosstab(df_scores['marital'],df_scores['Cluster'])
print(contingency_table_marital.to_string())

p_values=[]
p_index=0

#Facciamo Chi square perchè non è 2x2 (Fisher) ma più (6x3)
chi2, p, _, _, = chi2_contingency(contingency_table_marital)
p_values=p_values+[p]
print(chi2,p_values[p_index])

def different(p_value,alpha,variable):
    print(f'\n {variable}')
    if p_value< alpha:
        print(f'\n Ci sono differenze p_value: { p_value} < alpha: {alpha}')
    else:
        print(f'\n Non ci sono differenze p_value: { p_value} > alpha: {alpha})')

different(p_values[p_index],alpha_corrected,'marital')

#Values are reported as median (25th-75th) for continuous variable, % for binary variables, and mode for nominal variables
#Since this is a nominal -> we evaluate the frequency from the contingency table
# we have single for cluster 0 (28), married for cluster 1 (24), and also for cluster 2 (31)


#Creiamo la contingency table per marital con i cluster
p_index +=1


contingency_table_gender= pd.crosstab(df_scores['gender'],df_scores['Cluster'])
print(contingency_table_gender.to_string())
#Facciamo Chi square perchè non è 2x2 (Fisher) ma più (6x3)
chi2, p, _, _, = chi2_contingency(contingency_table_gender)
p_values=p_values+[p]
print(chi2,p_values[p_index])

different(p_values[p_index],alpha_corrected,'gender')





ordinal=['age', 'education', 'income', 'phq_score', 'gad_score', 'heas_score', 'eheals_score', 'ccs_score']
for att in ordinal:
    print(f'\n\n{att}:')
    cluster_0 = df_scores[df_scores['Cluster'] == 0 ][att]
    cluster_1 = df_scores[df_scores['Cluster'] == 1 ][att]
    cluster_2 = df_scores[df_scores['Cluster'] == 2 ][att]
    ### KRUSKAL WALLIS
    print('\nKruskal Wallis:')
    stat, p = kruskal([cluster_0], [cluster_1], [cluster_2], axis = 1)
    p_index = p_index + 1
    p_values = p_values + [p]
    different(p_values[p_index], 0.05, att)  # Krustal-Wallis doesn't want Bonferroni correction

    ### PAIRWISE MANN WHITNEY U
    # it is a pairwise -> I must consider all pairs

    # 0 - 1
    print('\nMann Whitney - clusters 0 - 1:')
    stat, p = mannwhitneyu(cluster_0, cluster_1)
    different(p, alpha_corrected, att)

    # 0 - 2
    print('\nMann Whitney - clusters 0 - 2:')
    stat, p = mannwhitneyu(cluster_0, cluster_2)
    different(p, alpha_corrected, att)

    # 1 - 2
    print('\nMann Whitney - clusters 1 - 2:')
    stat, p = mannwhitneyu(cluster_1, cluster_2)
    different(p, alpha_corrected, att)




'''for i in range(p_index+1, 9):
    df_new = df_scores.iloc[0,10].groupby('Cluster')
    stat,p= kruskal()'''


