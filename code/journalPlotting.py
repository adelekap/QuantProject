import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# Load Data
journalData = pd.read_csv('../data/Journals.csv')
articleData = pd.read_csv('../data/Articles.csv')

# Make open_access field binary category
journalData.open_access = pd.Categorical(journalData.open_access)
journalData['open_access'] = journalData.open_access.cat.codes

# Create Box Plot
box = sns.boxplot('journal_id','n_authors',data=articleData)
plt.tight_layout()
plt.savefig('../plots/nAuthorsBoxplot.png')
plt.show()

# Create Joint Plot
joint = sns.jointplot('Eigenfactor_Score', 'Journal_Impact_Factor', kind="kde", size=7, space=0, data=journalData)
plt.savefig('../plots/EigenImpactJointplot.png')
plt.show()

