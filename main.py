import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Lecture des données pour les deux domaines
data1 = np.loadtxt('convergence_domaine1.txt')
data2 = np.loadtxt('convergence_domaine2.txt')
x1, y1 = data1[:, 0], data1[:, 1]
x2, y2 = data2[:, 0], data2[:, 1]

# Régression linéaire pour les deux domaines
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)

# Tracé
plt.figure(figsize=(10, 6))

# Domaine 1
plt.plot(x1, y1, 'bo-', label='Erreur Domaine 1')
plt.plot(x1, slope1*x1 + intercept1, 'b--', label=f'Pente D1 = {slope1:.2f}')

# Domaine 2
plt.plot(x2, y2, 'ro-', label='Erreur Domaine 2')
plt.plot(x2, slope2*x2 + intercept2, 'r--', label=f'Pente D2 = {slope2:.2f}')

plt.grid(True)
plt.xlabel('log10(h)')
plt.ylabel('log10(erreur)')
plt.legend()
plt.title('Analyse de convergence - Comparaison des deux domaines')
plt.show()