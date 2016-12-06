import matplotlib.pyplot as plt
import df as pd

df = pd.read_table('web_traffic.tsv')

x = df.iloc[:, 0]
y = df.iloc[:, 1]

print x
# print y

plt.scatter(x, y, s=10)
plt.title("Trafego Web Mensal")
plt.xlabel("Tempo")
plt.ylabel("Hits/Hora")
plt.xticks([w*7*24 for w in range(10), ['semana %i' % w for w in range(10)]])
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')
plt.show()
