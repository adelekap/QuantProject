import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# Load Data
journalData = pd.read_csv('../data/Journals.csv')
articleData = pd.read_csv('../data/Articles.csv')

# Binary masks
journalData.open_access = pd.Categorical(journalData.open_access)
journalData['open_access'] = journalData.open_access.cat.codes

# Create Box Plot
box = sns.boxplot('journal_id','n_authors',data=articleData)
plt.tight_layout()
plt.title('Number of Authors')
plt.savefig('../plots/nAuthorsBoxplot.png')
plt.show()

# Create Joint Plot for factor scores
joint = sns.jointplot('Eigenfactor_Score', 'Journal_Impact_Factor', kind="kde", size=7, space=0, data=journalData)
plt.savefig('../plots/EigenImpactJointplot.png')
plt.show()

# Create Joint Plot for citations
citations = articleData[['journal_id','n_citations_2018_Google']].groupby('journal_id').apply(np.mean)
citationsData = journalData.merge(citations,on='journal_id')
sns.jointplot('Eigenfactor_Score', 'n_citations_2018_Google', kind="kde", size=7, space=0, data=citationsData,color='red')
plt.savefig('../plots/citationsEigenJointplot.png')
plt.show()
sns.jointplot('Journal_Impact_Factor', 'n_citations_2018_Google', kind="kde", size=7, space=0, data=citationsData,color='green')
plt.savefig('../plots/citationsImpactJointplot.png')
plt.show()

pages = articleData[['journal_id','n_pages']].groupby('journal_id').apply(np.mean)
pagesData = citationsData.merge(pages,on='journal_id')
sns.jointplot('n_pages','n_citations_2018_Google',data=pagesData,kind='kde',color='purple')
plt.savefig('../plots/citationNPages.png')
plt.show()
sns.jointplot('n_pages','n_authors',data=articleData,kind='kde',color='black')
plt.savefig('../plots/authorsNPages.png')
plt.show()
sns.jointplot('n_authors','n_citations_2018_Google', data=articleData, kind='kde',color='brown')
plt.savefig('../plots/authorsCitations.png')
plt.show()

# Create Distribution
dist = sns.distplot(articleData['n_pages'])
plt.title('Number of Pages')
plt.savefig('../plots/nPages.pdf')
plt.show()

sns.factorplot('open_access','n_citations_2018_Google',data=citationsData,kind='bar',palette="RdBu_r")
plt.savefig('../plots/openCitations.png')
plt.show()
sns.factorplot('open_access','Eigenfactor_Score',data=citationsData,kind='bar',palette="RdBu_r")
plt.savefig('../plots/openEigen.png')
plt.show()
sns.factorplot('open_access','Journal_Impact_Factor',data=citationsData,kind='bar',palette="RdBu_r")
plt.savefig('../plots/openImpact.png')
plt.show()

