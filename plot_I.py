import matplotlib.pyplot as plt

# Lire le fichier valeurs_I.txt
file_name = "valeurs_I.txt"

# Charger les données
iterations = []
valeurs_I = []

with open(file_name, 'r') as file:
    # Lire les lignes du fichier
    lines = file.readlines()
    for line in lines:
        # Ignorer les lignes vides ou les en-têtes
        if not line.strip() or "Iteration" in line:
            continue
        # Extraire les valeurs
        parts = line.split()
        iterations.append(int(parts[0]))  # Première colonne : Iteration
        valeurs_I.append(float(parts[1]))  # Deuxième colonne : Valeur de I

# Tracer les données
plt.figure(figsize=(10, 6))
plt.plot(iterations, valeurs_I, label="Valeur de I", linewidth=1.5)
plt.title("Convergenece de I - méthode de l\'équation adjointe", fontsize=16)
plt.xlabel("Itération", fontsize=14)
plt.ylabel("Valeur de I", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
# Sauvegarde du graphe en tant que fichier PNG
plt.savefig('Convergence_I_adjoint.png', dpi=300)  # Le paramètre dpi ajuste la qualité de l'image




# Afficher le graphique
plt.show()
