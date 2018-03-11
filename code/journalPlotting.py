import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# Load Data
journalData = pd.read_csv('../data/Journals.csv')
articleData = pd.read_csv('../data/Articles.csv')


journalData.open_access = pd.Categorical(journalData.open_access)
journalData['open_access'] = journalData.open_access.cat.codes

# Create Box Plot
box = sns.boxplot('journal_id','n_authors',data=articleData)
plt.tight_layout()
plt.title('Number of Authors')
plt.savefig('../plots/nAuthorsBoxplot.png')
plt.show()

# Create Joint Plot
joint = sns.jointplot('Eigenfactor_Score', 'Journal_Impact_Factor', kind="kde", size=7, space=0, data=journalData)
plt.savefig('../plots/EigenImpactJointplot.png')
plt.show()

# Create Distribution
dist = sns.distplot(articleData['n_pages'])
plt.title('Number of Pages')
plt.savefig('../plots/nPages.pdf')
plt.show()
