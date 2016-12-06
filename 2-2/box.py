import matplotlib.pyplot as plt
import numpy as np
import df as pd
import scipy.stats as stats

df = pd.read_csv("pesos_alturas_english.csv")
df.columns = ['altura', 'peso']

print df

# plt.boxplot(df['altura'])
# plt.show()

# plt.scatter(df['altura'], df['peso'])
# plt.show()

h = df['altura']
fit = stats.norm.pdf(h, np.mean(h), np.std(h))

plt.hist(h, normed=True)
plt.show()