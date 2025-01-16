#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

using namespace std;
using namespace Eigen;

// Définition des constantes
const double xmin = 0.0, xmax = 1.0;
const double ymin = 0.0, ymax = 1.0;

// Fonction source
double source_terme(double x, double y) {
    return -sin(M_PI*x) * sin(M_PI*y);
}

// Solution exacte
double solution_exacte(double x, double y) {
    return sin(M_PI*x) * sin(M_PI*y) / (2*M_PI*M_PI);
}

// Fonction pour calculer les tailles des sous-domaines
int* charge(int Nx) {
    static int N[2];
    N[0] = floor(Nx/2);
    N[1] = (Nx%2 == 0) ? N[0] : N[0] + 1;
    return N;
}

// Construction de la matrice
SparseMatrix<double> Matrice(int Nx, int Ny, double alpha, double beta, double gamma) {
    int N = (Nx-1) * (Ny-1);
    SparseMatrix<double> A(N, N);
    std::vector<Triplet<double>> coefficients;
    
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            int k = (j-1) * (Nx-1) + i-1;
            coefficients.emplace_back(k, k, alpha);
            
            if (i > 1) coefficients.emplace_back(k, k - 1, gamma);
            if (i < Nx - 1) coefficients.emplace_back(k, k + 1, gamma);
            if (j > 1) coefficients.emplace_back(k, k - (Nx-1), beta);
            if (j < Ny - 1) coefficients.emplace_back(k, k + (Nx-1), beta);
        }
    }
    
    A.setFromTriplets(coefficients.begin(), coefficients.end());
    return A;
}

// Construction du second membre pour le domaine complet
VectorXd second_membre(int Nx, int Ny, double dx, double dy) {
    VectorXd B((Nx-1) * (Ny-1));  // Correction de la dimension
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            double x = xmin + i * dx;
            double y = ymin + j * dy;
            B((j-1) * (Nx-1) + i-1) = source_terme(x, y);  
        }
    }
    return B;
}

// Construction du second membre pour la méthode de Schwarz
VectorXd second_membre_schwarz(int Nx, int Ny, double dx, double dy, int Nr, int domain, const VectorXd& g1, const VectorXd& g2) {
    int *N = charge(Nx);
    int Nx1 = N[0], Nx2 = N[1];
    
    if (domain == 1) {
        // Domaine 1 (gauche)
        VectorXd B((Nx1 + Nr-1) * (Ny-1));  // Correction de la dimension
        for (int j = 1; j < Ny; ++j) {
            for (int i = 1; i < Nx1 + Nr; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                int k = (j-1) * (Nx1 + Nr-1) + i-1;  // Correction de l'indice
                B(k) = source_terme(x, y);
                
                // Condition à l'interface droite
                if (i == Nx1 + Nr - 1) {
                    B(k) = -g2(j-1)/(dx*dx);  // Correction de l'indice
                }
            }
        }
        return B;
    } else {
        // Domaine 2 (droite)
        VectorXd B((Nx-Nx2+ Nr-1) * (Ny-1));  // Correction de la dimension
        for (int j = 1; j < Ny; ++j) {
            for (int i = Nx2-Nr+1; i < Nx; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                int k = (j-1) * (Nx-Nx2 + Nr-1) + (i-Nx2+Nr-1);
                B(k) = source_terme(x, y);
                
                // Condition à l'interface gauche
                if (i == Nx2-Nr+1) {
                    B(k) = -g1(j-1)/(dx*dx);  // Correction de l'indice
                }
            }
        }
        return B;
    }
}

double calcul_erreur_L2(const VectorXd& u_num, int Ny, int domain, int Nr, double dx, double dy, int Nx_total) {
    double error = 0.0;
    int *N = charge(Nx_total);
    int Nx1 = N[0];
    int Nx2 = N[1];
    
    if (domain == 1) {
        for(int j = 1; j < Ny; ++j) {
            for(int i = 1; i < Nx1 + Nr; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                double u_exact = solution_exacte(x, y);
                int k = (j-1) * (Nx1 + Nr-1) + i-1;  // Correction de l'indice
                if (k < u_num.size()) {  // Vérification de l'indice
                    error += pow(u_num(k) - u_exact, 2);
                }
            }
        }
    }
    else {
        for(int j = 1; j < Ny; ++j) {
            for(int i = Nx2 - Nr + 1; i < Nx_total; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                double u_exact = solution_exacte(x, y);
                int k = (j-1) * (Nx_total-Nx2 + Nr-1) + (i-Nx2+Nr-1);
                if (k < u_num.size()) {  // Vérification de l'indice
                    error += pow(u_num(k) - u_exact, 2);
                }
            }
        }
    }
    return sqrt(error * dx * dy);
}

int main() {
    vector<int> N_values = {5, 10, 20, 40, 80};
    // vector<int> N_values = {5};
    ofstream convergence1("convergence_domaine1.txt");
    ofstream convergence2("convergence_domaine2.txt");
    
    double previous_h = 0;
    double previous_error1 = 0;
    double previous_error2 = 0;
    
    cout << "Analyse de convergence par sous-domaine" << endl;
    cout << "----------------------------------------" << endl;
    
    for(int N : N_values) {
        int Nx = N, Ny = N;
        int Nr = 2;
        int *Ns = charge(Nx);
        int Nx1 = Ns[0];
        int Nx2 = Ns[1];
        
        double dx = (xmax - xmin) / (Nx);
        double dy = (ymax - ymin) / (Ny);
        double h = max(dx, dy);
        
        double alpha = -2.0 * (1.0/(dx*dx) + 1.0/(dy*dy));
        double beta = 1.0 / (dy * dy);
        double gamma = 1.0 / (dx * dx);
        
        // Initialisation pour le domaine complet
        SparseMatrix<double> A = Matrice(Nx, Ny, alpha, beta, gamma);
        VectorXd f = second_membre(Nx, Ny, dx, dy);
        SparseLU<SparseMatrix<double>> solver;
        solver.compute(A);
        VectorXd u = solver.solve(f);
        
        // Construction des conditions aux limites pour Schwarz
        VectorXd g1(Ny-1), g2(Ny-1);  // Correction de la dimension
        for (int j = 1; j < Ny; ++j) {
            double x1 = xmin + (Nx1+Nr-1)*dx;
            double y = ymin + j*dy;
            g1(j-1) = solution_exacte(x1,y);
            
            double x2 = xmin + (Nx2-Nr+1)*dx;
            g2(j-1) = solution_exacte(x2,y);
        }

        // Résolution sur les sous-domaines
        SparseMatrix<double> A1 = Matrice(Nx1 + Nr, Ny, alpha, beta, gamma);
        SparseMatrix<double> A2 = Matrice(Nx-Nx2+ Nr , Ny, alpha, beta, gamma);
        
        VectorXd f1 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, g1, g2);
        VectorXd f2 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, g1, g2);
        
        SparseLU<SparseMatrix<double>> solver1, solver2;
        solver1.compute(A1);
        solver2.compute(A2);
        
        VectorXd u1 = solver1.solve(f1);
        VectorXd u2 = solver2.solve(f2);
        
        // Calcul de l'erreur
        double error1 = calcul_erreur_L2(u1, Ny, 1, Nr, dx, dy, Nx);
        double error2 = calcul_erreur_L2(u2, Ny, 2, Nr, dx, dy, Nx);
        
        // Affichage des résultats
        cout << "N = " << N << ", h = " << h << endl;
        // cout << "Domaine 1 - Erreur L2: " << error1;
        // cout << "Domaine 2 - Erreur L2: " << error2;

        if(previous_h > 0) {
            double slope1 = log10(error1/previous_error1) / log10(h/previous_h);
            cout << "\tPente: " << slope1;
        }
        cout << endl;
        
        if(previous_h > 0) {
            double slope2 = log10(error2/previous_error2) / log10(h/previous_h);
            cout << "\tPente: " << slope2;
        }
        cout << endl;
        
        // Sauvegarde pour le tracé
        convergence1 << log10(h) << " " << log10(error1) << endl;
        convergence2 << log10(h) << " " << log10(error2) << endl;
        
        previous_h = h;
        previous_error1 = error1;
        previous_error2 = error2;
    }
    
    convergence1.close();
    convergence2.close();
    cout << "\nDonnées de convergence écrites dans 'convergence_domaine1.txt'" << endl;
    cout << "\nDonnées de convergence écrites dans 'convergence_domaine2.txt'" << endl;
    
    return 0;
}