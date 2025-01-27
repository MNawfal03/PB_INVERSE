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
// Construction du second membre pour la mÃƒÂ©thode de Schwarz
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
                
                // Condition Ãƒ  l'interface droite
                if (i == Nx1 + Nr - 1) {
                    B(k) += -g1(j-1)/(dx*dx);  // Correction de l'indice
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
                
                // Condition Ãƒ  l'interface gauche
                if (i == Nx2-Nr+1) {
                    B(k) += -g2(j-1)/(dx*dx);  // Correction de l'indice
                }
            }
        }
        return B;
    }
}

VectorXd Concatener(const VectorXd& u1, const VectorXd& u2) {
    int N1 = u1.size();
    int N2 = u2.size();
    VectorXd u(N1 + N2);
    u << u1, u2; // Concatenation directe avec Eigen
    return u;
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
                if (k < u_num.size()) {  // VÃ©rification de l'indice
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
                if (k < u_num.size()) {  // VÃ©rification de l'indice
                    error += pow(u_num(k) - u_exact, 2);
                }
            }
        }
    }
    return sqrt(error * dx * dy);
}


// RÃ©solution du systÃ¨me AU = b(alpha)
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
    
    for(int j=1; j< Ny; ++j){
        for (int i = 1; i < 2* Nr; i++)
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
    
    for (int p = 0; p < n1+n2; p++)
    {   if (p<n1)
        {   VectorXd alpha_eps1 = alpha1;
            VectorXd alpha_eps2 = alpha2;
            //alpha_eps2[] = alpha_eps2[m] +eps ;
            alpha_eps1[p] = alpha_eps1[p] +eps ;
            VectorXd f1_eps = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, alpha_eps1, alpha_eps2);
            VectorXd f2_eps = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, alpha_eps1, alpha_eps2);
            
            VectorXd U1_eps = solve_system(solver1, f1_eps);
            VectorXd U2_eps = solve_system(solver2, f2_eps);
            VectorXd U_eps=Concatener(U1_eps,U2_eps);

            double I_plus = compute_I(U_eps,Nx,Nr,Ny);
            G(p) = (I_plus - I_current) / eps;
        }
        else
        {
            VectorXd alpha_eps1 = alpha1;
            VectorXd alpha_eps2 = alpha2;
            alpha_eps2[p-n1] = alpha_eps2[p-n1] +eps ;
            VectorXd f1_eps = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, alpha_eps1, alpha_eps2);
            VectorXd f2_eps = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, alpha_eps1, alpha_eps2);
            
            VectorXd U1_eps = solve_system(solver1, f1_eps);
            VectorXd U2_eps = solve_system(solver2, f2_eps);
            VectorXd U_eps=Concatener(U1_eps,U2_eps);

            double I_plus = compute_I(U_eps,Nx,Nr,Ny);
            G(p) = (I_plus - I_current) / eps;
        }
        
    }
    // cout<< "G " << G.norm() << endl ;
    return G;
}

// CritÃ¨re de convergence
bool convergence_criteria(const VectorXd& G, double tol = 0.0001) {
    return G.norm() < tol;
}

// RÃ©solution du problÃ¨me inverse
// RÃ©solution du problÃ¨me inverse avec suivi de I
tuple<VectorXd, VectorXd, VectorXd, VectorXd> inverse_problem_sensitivity(const SparseLU<SparseMatrix<double>>& solver1,
                                     const SparseLU<SparseMatrix<double>>& solver2,
                                     VectorXd f1,
                                     VectorXd f2,
                                     VectorXd alpha1,
                                     VectorXd alpha2,
                                     int Nx, int Ny, int Nr, double dx, double dy,
                                     int max_iter = 1000, double tol = 1.e-6, double sigma = 0.1) {
    int n1 = alpha1.size();
    int n2 = alpha2.size();
    VectorXd alpha_old , U1, U2 ,U;
    
    // Ouvrir le fichier pour Ã©crire les valeurs de I
    ofstream file_I("valeurs_I.txt");
    if (!file_I.is_open()) {
        cerr << "Erreur: Impossible d'ouvrir le fichier valeurs_I.txt" << endl;
        throw runtime_error("Erreur d'ouverture de fichier");
    }
    
    // Ã‰crire l'en-tÃªte du fichier
    file_I << "# Iteration\tValeur de I" << endl;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        U1 = solve_system(solver1, f1); //gauche
        U2 = solve_system(solver2, f2); //droite
        U = Concatener(U1, U2);
        
        // // Calculer I pour cette itÃ©ration
        double I_current = compute_I(U, Nx, Nr, Ny);
        
        // Ã‰crire dans le fichier
        file_I << iter << "\t" << I_current << endl;
        
        // VectorXd alpha = Concatener(alpha1, alpha2);
        VectorXd G = compute_G(solver1, solver2, alpha1, alpha2, Nx, Ny, Nr, dx, dy, U);

        if (convergence_criteria(G)) {
            cout << "Convergence atteinte aprÃ¨s " << iter + 1 << " itÃ©rations." << endl;
            
            // f1 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, alpha1, alpha2);
            // f2 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, alpha1, alpha2);
            
            // U1 = solve_system(solver1, f1);
            // U2 = solve_system(solver2, f2);

            // Ã‰crire la valeur finale de I
            file_I << "# Convergence atteinte" << endl;
            // file_I << iter + 1 << "\t" << I_current << endl;
            break;
        }
        else {
            alpha1 = alpha1 - sigma * G.head(n1);
            alpha2 = alpha2 - sigma * G.tail(n2);
            cout<< "alpha1 " << alpha1.norm() << endl ;
            f1 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, alpha1, alpha2);
            f2 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, alpha1, alpha2);

        }
    }
    
    // Fermer le fichier
    file_I.close();
    
    return make_tuple(alpha1, alpha2, U1, U2);
    }
int main() {
    vector<int> N_values = {40};
    // vector<int> N_values = {10};
    ofstream convergence1("convergence_domaine1.txt");
    ofstream convergence2("convergence_domaine2.txt");

    ofstream solution1("solution_domain1.txt");
    ofstream solution2("solution_domain2.txt");

    ofstream exact1("exact_domain1.txt");
    ofstream exact2("exact_domain2.txt");

    
    double previous_h = 0;
    double previous_error1 = 0;
    double previous_error2 = 0;

    for(int N : N_values) {
        int Nx = N, Ny = N;
        int Nr = 2;
        int *Ns = charge(Nx);
        int Nx1 = Ns[0];
        int Nx2 = Ns[1];
        int N1=Nx1, N2=Nx-Nx2;
        cout << "N1 " << Nx1 << " N2 "<< N2 << endl;
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

        //DÃ©composition LU
        SparseLU<SparseMatrix<double>> solver1, solver2;
        solver1.compute(A1);
        solver2.compute(A2);
        
        //initialisation de alpha1 et alpha2
        VectorXd alpha_vec1 = VectorXd::Random(Ny-1); //g1
        VectorXd alpha_vec2 = VectorXd::Random(Ny-1); //g2
        // VectorXd alpha_vec1(Ny-1), alpha_vec2(Ny-1);  
        // for (int j = 1; j < Ny; ++j) {
        //     double x1 = xmin + (Nx1+Nr-1)*dx;
        //     double y = ymin + j*dy;
        //     alpha_vec1(j-1) = solution_exacte(x1,y);
        //     // alpha_vec1(j-1) = 0.;
        //     double x2 = xmin + (Nx2-Nr+1)*dx;
        //     alpha_vec2(j-1) = solution_exacte(x2,y);
        //     // alpha_vec2(j-1) = 0.;
        // }
        // VectorXd U_exact1 = second_membre(N1, N1, (xmax - xmin) / n, (ymax - ymin) / n).array() / (-4.0);
        
        //initialisation des seconds membres
        VectorXd f1 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, alpha_vec1, alpha_vec2);
        VectorXd f2 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, alpha_vec1, alpha_vec2);
        // VectorXd f =Concatener(f1,f2);

        auto [alpha_solution1, alpha_solution2, U_solution1, U_solution2] =inverse_problem_sensitivity(solver1, solver2, f1, f2, alpha_vec1, alpha_vec2, Nx, Ny, Nr, dx, dy);
        double error1 = calcul_erreur_L2(U_solution1, Ny, 1, Nr, dx, dy, Nx); 
        double error2 = calcul_erreur_L2(U_solution2, Ny, 2, Nr, dx, dy, Nx);

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
        // Sauvegarde pour le tracÃ©
        convergence1 << log10(h) << " " << log10(error1) << endl;
        convergence2 << log10(h) << " " << log10(error2) << endl;
        
        previous_h = h;
        previous_error1 = error1;
        previous_error2 = error2;

        // Pour le domaine 2
        for(int j = 1; j < Ny; ++j) {
            for(int i = 1; i < Nx1+Nr; ++i) {
                double x = xmin + (i+Nx2-Nr)*dx;
                double y = ymin + (j)*dy;
                int k = (j-1) * (Nx1 + Nr-1) + i-1;
                solution2 << x << " " << y << " " << U_solution2[k] << endl;
            }
            solution2 << endl; // Pour gnuplot splot
        }

        // // Pour le domaine 1
        for(int j = 1; j < Ny; ++j) {
            for(int i = 1; i < N2+Nr ; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                int k = (j-1) * (Nx1 + Nr-1) + i-1;
                solution1 << x << " " << y << " " << U_solution1[k] << endl;
            }
            solution1 << endl;
        }
        
        // Solution exacte U
        for(int j = 1; j < Ny; ++j) {
            for(int i = 1; i < Nx1+Nr; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                int k = (j-1) * (Nx1 + Nr-1) + i-1;
                exact1 << x << " " << y << " " << solution_exacte(x,y) << endl;
            }
            exact1 << endl;
        }

        for(int j = 1; j < Ny; ++j) {
            for(int i = Nx2-Nr+1; i < Nx; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                int k = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-Nx2+Nr-1);
                exact2 << x << " " << y << " " << solution_exacte(x,y) << endl;
            }
            exact2 << endl;
        }
        
        exact1.close();
        exact2.close();
        solution1.close();
        solution2.close();
        
    }
    convergence1.close();
    convergence2.close();
    
    return 0;
}
