#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

using namespace Eigen;
using namespace std;

const double xmin = 0.0, xmax = 1.0;
const double ymin = 0.0, ymax = 1.0;

double source_terme(double x, double y) {
    return -sin(M_PI*x) * sin(M_PI*y);
}

double solution_exacte(double x, double y) {
    return sin(M_PI*x) * sin(M_PI*y) / (2*M_PI*M_PI);
}

int* charge(int Nx) {
    static int N[2];
    N[0] = floor(Nx/2);
    N[1] = (Nx%2 == 0) ? N[0] : N[0] + 1;
    return N;
}

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

VectorXd second_membre(int Nx, int Ny, double dx, double dy) {
    VectorXd B((Nx-1) * (Ny-1));
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            double x = xmin + i * dx;
            double y = ymin + j * dy;
            B((j-1) * (Nx-1) + i-1) = source_terme(x, y);
        }
    }
    return B;
}

// prise en compte de g1 et g2 
// Construction du second membre pour la mÃ©thode de Schwarz
VectorXd second_membre_schwarz(int Nx, int Ny, double dx, double dy, int Nr, int domain, const VectorXd& g1, const VectorXd& g2) {
    int *N = charge(Nx);
    int Nx1 = N[0], Nx2 = N[1];
    
    if (domain == 1) {
        // Domaine 1 (gauche)
        VectorXd B1((Nx1 + Nr-1) * (Ny-1));  // Correction de la dimension
        for (int j = 1; j < Ny; ++j) {
            for (int i = 1; i < Nx1 + Nr; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                int k = (j-1) * (Nx1 + Nr-1) + i-1;  // Correction de l'indice
                B1(k) = source_terme(x, y);
                
                // Condition Ã  l'interface droite
                if (i == Nx1 + Nr - 1) {
                    B1(k) = -g1(j-1)/(dx*dx);  // Correction de l'indice
                }
            }
        }
        return B1;
    } else {
        // Domaine 2 (droite)
        VectorXd B2((Nx-Nx2+ Nr-1) * (Ny-1));  // Correction de la dimension
        for (int j = 1; j < Ny; ++j) {
            for (int i = Nx2-Nr+1; i < Nx; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                int k = (j-1) * (Nx-Nx2 + Nr-1) + (i-Nx2+Nr-1);
                B2(k) = source_terme(x, y);
                
                // Condition Ã  l'interface gauche
                if (i == Nx2-Nr+1) {
                    B2(k) = -g2(j-1)/(dx*dx);  // Correction de l'indice
                }
            }
        }
        return B2;
    }
}

VectorXd Concatener(const VectorXd& u1, const VectorXd& u2) {
    int N1 = u1.size();
    int N2 = u2.size();
    VectorXd u(N1 + N2);
    u << u1, u2; // Concatenation directe avec Eigen
    return u;
}
// Décomposition LU


// Résolution du système AU = b(alpha)
VectorXd solve_system(const SparseLU<SparseMatrix<double>>& lu, const VectorXd& b) {
    return lu.solve(b);
}

// Calcul de I(alpha)
double compute_I(const VectorXd& U, int Nx ,int Nr, int Ny) {
    double r;
    int k1,k2;
    int *Ns = charge(Nx);
    int N1 = Ns[0];
    int N2 = Nx- Ns[1] ;
    
    for(int j=1; j< Ny -1; ++j){
        for (int i = 2; i < 2* Nr; i++)
        {
            k1=(j-1)*(N1+Nr-1)+(i-1)+(N1-Nr);
            k2=(N1+Nr-1)*(Ny-1)+(j-1)*(N2+Nr-1)+(i-1);
            r+=(U[k1]-U[k2])*(U[k1]-U[k2]);           
        }        
    }
    return r;
}

// Calcul du gradient de I
VectorXd compute_G(const SparseLU<SparseMatrix<double>>& solver1,
                    const SparseLU<SparseMatrix<double>>& solver2,
                    const VectorXd& alpha1,const VectorXd& alpha2, 
                    int Nx, int Ny, int Nr,double dx , double dy ,
                    const VectorXd& U, double eps = 1e-6) {
    int n1 = alpha1.size();
    int n2 = alpha2.size();
    VectorXd G(n1+n2);
    
    double I_current = compute_I(U,Nx,Nr,Ny);
    VectorXd alpha_eps1 = alpha1;
    VectorXd alpha_eps2 = alpha2;
    for (int p = 0; p < n1; p++)
    {
        alpha_eps1[p] = alpha_eps1[p] +eps ;
    }

    for (int p = 0; p < n2; p++)
    {
        alpha_eps2[p] = alpha_eps2[p] +eps ;
    }

    for (int i = 0; i < n1+n2; ++i) {
        
        VectorXd f1_eps = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, alpha_eps1, alpha_eps2);
        VectorXd f2_eps = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, alpha_eps1, alpha_eps2);
        
        VectorXd U1_eps = solve_system(solver1, f1_eps);
        VectorXd U2_eps = solve_system(solver2, f2_eps);
        VectorXd U_eps=Concatener(U1_eps,U2_eps);

        double I_plus = compute_I(U_eps,Nx,Nr,Ny);
        G(i) = (I_plus - I_current) / eps;
    }
    cout<< "G " << G[0] << endl ;
    return G;
}

// Critère de convergence
bool convergence_criteria(const VectorXd& G, double tol = 1e-6) {
    return G.norm() < tol;
}

// Résolution du problème inverse
// Résolution du problème inverse avec suivi de I
VectorXd inverse_problem_sensitivity(const SparseLU<SparseMatrix<double>>& solver1,
                                     const SparseLU<SparseMatrix<double>>& solver2,
                                     const SparseMatrix<double>& A1,
                                     const SparseMatrix<double>& A2,
                                     VectorXd alpha1,
                                     VectorXd alpha2,
                                     int Nx, int Ny, int Nr, double dx, double dy,
                                     int max_iter = 1000, double tol = 1e-6, double sigma = 0.01) {
    int n1 = alpha1.size();
    int n2 = alpha2.size();
    VectorXd alpha_old;
    
    // Ouvrir le fichier pour écrire les valeurs de I
    ofstream file_I("valeurs_I.txt");
    if (!file_I.is_open()) {
        cerr << "Erreur: Impossible d'ouvrir le fichier valeurs_I.txt" << endl;
        throw runtime_error("Erreur d'ouverture de fichier");
    }
    
    // Écrire l'en-tête du fichier
    file_I << "# Iteration\tValeur de I" << endl;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd f1 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, alpha1, alpha2);
        VectorXd f2 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, alpha1, alpha2);
        
        VectorXd U1 = solve_system(solver1, f1);
        VectorXd U2 = solve_system(solver2, f2);
        VectorXd U = Concatener(U1, U2);
        
        // Calculer I pour cette itération
        double I_current = compute_I(U, Nx, Nr, Ny);
        
        // Écrire dans le fichier
        file_I << iter << "\t" << I_current << endl;
        
        VectorXd alpha = Concatener(alpha1, alpha2);
        VectorXd G = compute_G(solver1, solver2, alpha1, alpha2, Nx, Ny, Nr, dx, dy, U);

        if (convergence_criteria(G, tol)) {
            cout << "Convergence atteinte après " << iter + 1 << " itérations." << endl;
            
            // Écrire la valeur finale de I
            file_I << "# Convergence atteinte" << endl;
            file_I << iter + 1 << "\t" << I_current << endl;
            break;
        }
        else {
            alpha1 -= sigma * G.head(n1);
            alpha2 -= sigma * G.tail(n2);
        }
    }
    
    // Fermer le fichier
    file_I.close();
    
    return alpha1, alpha2;
}
int main() {
    int Nx=10,Ny=10;
    int Nr = 2;
    int *Ns = charge(Nx);
    int Nx1 = Ns[0];
    int Nx2 = Ns[1];
    int N1=Nx1, N2=Nx-Nx2;
    // vectorXd u, derive_I_U,lambda;
    
    double dx = (xmax - xmin) / (Nx);
    double dy = (ymax - ymin) / (Ny);
    double h = max(dx, dy);
    
    double alpha = -2.0 * (1.0/(dx*dx) + 1.0/(dy*dy));
    double beta = 1.0 / (dy * dy);
    double gamma = 1.0 / (dx * dx);
    
    // Initialisation des matrices
    SparseMatrix<double> A1 = Matrice(N1+Nr, Ny, alpha, beta, gamma);
    SparseMatrix<double> A2 = Matrice(N2+Nr, Ny, alpha, beta, gamma);
    
      
    
    // SparseMatrix<double> A=Matrice_globale(A1,A2);

    //Décomposition LU
    SparseLU<SparseMatrix<double>> solver1, solver2;
    solver1.compute(A1);
    solver2.compute(A2);
    
    //initialisation de alpha1 et alpha2
    VectorXd alpha_vec1 = VectorXd::Random(Ny-1); //g1
    VectorXd alpha_vec2 = VectorXd::Random(Ny-1); //g2
    // VectorXd U_exact1 = second_membre(N1, N1, (xmax - xmin) / n, (ymax - ymin) / n).array() / (-4.0);
    
    //initialisation des seconds membres
    VectorXd f1 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, alpha_vec1, alpha_vec2);
    VectorXd f2 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, alpha_vec1, alpha_vec2);
    // VectorXd f =Concatener(f1,f2);

    VectorXd alpha_solution = inverse_problem_sensitivity(solver1,solver2, A1, A2 , alpha_vec1, alpha_vec2, Nx , Ny ,Nr ,dx,dy);
    
    cout << "Solution alpha :\n" << alpha_solution << endl;
    return 0;
}
