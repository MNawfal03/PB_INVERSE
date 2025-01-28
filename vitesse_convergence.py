import numpy as np
import matplotlib.pyplot as plt

# Lecture des données
data = np.loadtxt('cost_evolution.txt')
iterations = data[:, 0]
cost = data[:, 1]

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(iterations, cost, 'b-', label='Cost Evolution')
plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Evolution of Cost Function During Convergence')
plt.legend()

# Ajout d'une grille en arrière-plan
plt.grid(True, which="both", ls="-", alpha=0.2)

# Afficher le graphique
plt.show()
