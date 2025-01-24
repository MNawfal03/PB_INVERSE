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


int* charge(int Nx) {
    static int N[2];
    N[0] = floor(Nx/2);
    N[1] = (Nx%2 == 0) ? N[0] + 1 : N[0] + 1;
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
VectorXd second_membre_schwarz(int Nx, int Ny, double dx, double dy, int Nr, int domain) {
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
            }
        }
        return B;
    }
}

VectorXd second_membre_schwarz_0(int Nx, int Ny, double dx, double dy, int Nr, int domain, const VectorXd& g1, const VectorXd& g2) {
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
                // Condition à l'interface gauche
                if (i == Nx2-Nr+1) {
                    B(k) += -g2(j-1)/(dx*dx);  // Correction de l'indice
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



SparseMatrix<double> MatriceB1(int N1, int N2, int nr, int Ny, double beta) {
    // Dimensions
    int taille_ligne = (N1 + nr- 1) * (Ny - 1);
    int taille_colonne = (Ny - 1);

    SparseMatrix<double> J(taille_ligne, taille_colonne); // Matrice creuse
    std::vector<Triplet<double>> coefficients; // Stocke les triplets non nuls

    // Remplissage pour les Ny-1 premières lignes
    
    for(int ligne=N1+nr-2; ligne<(Ny - 1)*(N1+nr-1); ligne+=N1+nr-1){
        int colonne = ligne/(N1+nr-1);               // Chaque bloc correspond à une colonne dans ce groupe
            
        coefficients.emplace_back(ligne, colonne, -beta);
        // std::cout << "Ajout 1er bloc : ligne = " << ligne << ", colonne = " << colonne << ", valeur = " << -beta << "\n";
            
        }
    


    // Construction de la matrice à partir des triplets
    J.setFromTriplets(coefficients.begin(), coefficients.end());
    return J;
}



SparseMatrix<double> MatriceB2(int N1, int N2, int nr, int Ny, double beta) {
    // Dimensions
    int taille_ligne = (N2 + nr- 1) * (Ny - 1);
    int taille_colonne = (Ny - 1);

    SparseMatrix<double> J(taille_ligne, taille_colonne); // Matrice creuse
    std::vector<Triplet<double>> coefficients; // Stocke les triplets non nuls

    
    // Remplissage du second bloc
     for(int ligne=0; ligne<(Ny - 1)*(N2+nr-1); ligne+=N2+nr-1){
        int colonne = (ligne/(N2+nr-1));  
        // printf("ligne %d, colonne %d\n",ligne ,colonne);                       // Décalage des colonnes pour le deuxième bloc
            
        coefficients.emplace_back(ligne, colonne, -beta);
        // std::cout << "Ajout 2ème bloc : ligne = " << ligne << ", colonne = " << colonne << ", valeur = " << -beta << "\n";

        }


    
     // Construction de la matrice à partir des triplets
    J.setFromTriplets(coefficients.begin(), coefficients.end());
    return J;
}





// Fonction pour créer une matrice résultante par concaténation horizontale
SparseMatrix<double> Matrice_globale_horizontale(const SparseMatrix<double>& A1, const SparseMatrix<double>& A2) {
    // Dimensions des matrices d'entrée
    int sizeA1_rows = A1.rows();
    int sizeA1_cols = A1.cols();
    int sizeA2_cols = A2.cols();

    // Dimension totale de la matrice résultante
    int totalRows = sizeA1_rows;
    int totalCols = sizeA1_cols + sizeA2_cols;

    // Matrice résultante
    SparseMatrix<double> bigMatrix(totalRows, totalCols);

    // Stockage des coefficients pour la matrice résultante
    std::vector<Triplet<double>> coefficients;

    // Ajouter les coefficients de A1
    for (int k = 0; k < A1.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A1, k); it; ++it) {
            coefficients.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Ajouter les coefficients de A2
    for (int k = 0; k < A2.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A2, k); it; ++it) {
            coefficients.emplace_back(it.row(), it.col() + sizeA1_cols, it.value());
        }
    }

    // Construire la matrice finale à partir des triplets
    bigMatrix.setFromTriplets(coefficients.begin(), coefficients.end());

    return bigMatrix;
}


void afficherSparseMatrix(const SparseMatrix<double>& matrice) {
    for (int k = 0; k < matrice.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(matrice, k); it; ++it) {
            // std::cout << "Element (" << it.row() << ", " << it.col() 
            //           << ") = " << it.value() << std::endl;
        }
    }
}

// Fonction de concaténation de deux VectorXd
VectorXd Concatener(const VectorXd& u1, const VectorXd& u2) {
    int N1 = u1.size();
    int N2 = u2.size();
    VectorXd u(N1 + N2); // Crée un VectorXd pour stocker le résultat

    // Copie les valeurs de u1 dans u
    u.head(N1) = u1;
    
    // Copie les valeurs de u2 dans u, à partir de l'index N1
    u.tail(N2) = u2;

    return u;
}

// Fonction pour crÃ©er une matrice diagonale par blocs
SparseMatrix<double> Matrice_globale(const SparseMatrix<double>& A1, const SparseMatrix<double>& A2) {
    // Dimensions des matrices d'entrée
    int rowsA1 = A1.rows();
    int colsA1 = A1.cols();
    int rowsA2 = A2.rows();
    int colsA2 = A2.cols();

    // Dimension totale de la matrice diagonale par blocs
    int totalRows = rowsA1 + rowsA2;
    int totalCols = colsA1 + colsA2;

    // Matrice diagonale par blocs
    SparseMatrix<double> bigMatrix(totalRows, totalCols);

    // Stockage des coefficients pour la matrice diagonale par blocs
    std::vector<Triplet<double>> coefficients;

    // Ajouter les coefficients de A1 dans le bloc supérieur gauche
    for (int k = 0; k < A1.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A1, k); it; ++it) {
            coefficients.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Ajouter les coefficients de A2 dans le bloc inférieur droit
    for (int k = 0; k < A2.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A2, k); it; ++it) {
            coefficients.emplace_back(it.row() + rowsA1, it.col() + colsA1, it.value());
        }
    }

    // Construire la matrice finale à partir des triplets
    bigMatrix.setFromTriplets(coefficients.begin(), coefficients.end());

    return bigMatrix;
}

SparseMatrix<double> MatriceIdentité(int Nx, int Ny) {
    int N = (Nx - 1) * (Ny - 1);
    SparseMatrix<double> A(N, N);
    std::vector<Triplet<double>> coefficients;

    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            int k = (j - 1) * (Nx - 1) + (i - 1);
            if (i == j) { // Assurez-vous de remplir la diagonale
                coefficients.emplace_back(k, k, 1.0); // Valeur de 1 sur la diagonale
            }
        }
    }

    A.setFromTriplets(coefficients.begin(), coefficients.end());
    return A;
}

SparseMatrix<double> Matricenulle(int Nx, int Ny) {
    int N = (Nx - 1) * (Ny - 1);
    SparseMatrix<double> A(N, N);
    std::vector<Triplet<double>> coefficients;

    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            int k = (j - 1) * (Nx - 1) + (i - 1);
            if (i == j) { // Assurez-vous de remplir la diagonale
                coefficients.emplace_back(k, k, 0.0); // Valeur de 1 sur la diagonale
            }
        }
    }

    A.setFromTriplets(coefficients.begin(), coefficients.end());
    return A;
}

SparseMatrix<double> Matricenulle_0(int Nx, int Ny) {
    SparseMatrix<double> A(Nx-1, Ny-1);

    return A;
}


// Fonction pour empiler deux matrices creuses l'une au-dessus de l'autre
SparseMatrix<double> empilerVerticalement(const SparseMatrix<double>& A, const SparseMatrix<double>& B) {
    int totalRows = A.rows() + B.rows();
    int totalCols = std::max(A.cols(), B.cols());

    SparseMatrix<double> C(totalRows, totalCols);
    std::vector<Triplet<double>> coefficients;

    // Ajouter les coefficients de A
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            coefficients.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Ajouter les coefficients de B
    for (int k = 0; k < B.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
            coefficients.emplace_back(it.row() + A.rows(), it.col(), it.value()); // Décaler les lignes de B
        }
    }

    // Construire la matrice résultante à partir des triplets
    C.setFromTriplets(coefficients.begin(), coefficients.end());
    return C;
}


int main() {
    // vector<int> N_values = {20};
    vector<int> N_values = {10,20,40};
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
    
    cout << "Analyse de convergence par sous-domaine" << endl;
    cout << "----------------------------------------" << endl;
    
    for(int N : N_values) {
        int Nx = N, Ny = N;
        int Nr = 4;
        // int Nr = std::max(3, N/8); // Scale with mesh size
        int *Ns = charge(Nx);
        int Nx1 = Ns[0];
        int Nx2 = Ns[1];

        int N1 = Nx1  ;
        int N2 = Nx - Nx2 ;
        
        double dx = (xmax - xmin) / (Nx);
        double dy = (ymax - ymin) / (Ny);
        double h = max(dx, dy);
        
        double alpha = -2.0 * (1.0/(dx*dx) + 1.0/(dy*dy));
        double beta = 1.0 / (dy * dy);
        double gamma = 1.0 / (dx * dx);

        std :: cout << Nx1 << " " << Nx2 << " " << N1 << " " << N2 << std :: endl ;
        
    
        SparseMatrix<double> B1 = MatriceB1( N1,  N2,  Nr,  Ny,  beta);
        SparseMatrix<double> B2 = MatriceB2( N1,  N2,  Nr,  Ny,  beta);
        SparseMatrix<double> B = Matrice_globale(-B1 , -B2) ;
        // afficherSparseMatrix(B2);

        SparseMatrix<double> M1 = Matrice(Nx1 + Nr, Ny, alpha, beta, gamma);
        SparseMatrix<double> M2 = Matrice(Nx-Nx2+ Nr , Ny, alpha, beta, gamma);
        SparseMatrix<double> M = Matrice_globale(M1 , M2) ;

        SparseMatrix<double> zero1 = Matricenulle(Nx1 + Nr, Ny);
        SparseMatrix<double> zero2 = Matricenulle(Nx-Nx2+ Nr , Ny);
        SparseMatrix<double> Id1 = MatriceIdentité(Nx1 + Nr, Ny);
        SparseMatrix<double> Id2 = MatriceIdentité(Nx-Nx2+ Nr , Ny);
        SparseMatrix<double> zero_1 = Matrice_globale(zero1 , zero2) ;
        SparseMatrix<double> zero_2 = Matrice_globale(zero1 , zero2) ;


        SparseMatrix<double> Mf_sup = Matrice_globale_horizontale(M, B );
        SparseMatrix<double> Mf = Mf_sup;

        SparseLU<SparseMatrix<double>> solver1, solver2;

        solver1.compute(M1);
        solver2.compute(M2);
     

        VectorXd g1(Ny-1), g2(Ny-1);  
        for (int j = 1; j < Ny; ++j) {
            double x1 = xmin + (Nx1+Nr-1)*dx;
            double y = ymin + j*dy;
            g1(j-1) = solution_exacte(x1,y);
            double x2 = xmin + (Nx2-Nr+1)*dx;
            g2(j-1) = solution_exacte(x2,y);
        }

        VectorXd f1_ex = second_membre_schwarz_0(Nx, Ny, dx, dy, Nr, 1, g1, g2);
        VectorXd f2_ex = second_membre_schwarz_0(Nx, Ny, dx, dy, Nr, 2, g1, g2);

        VectorXd u1 = solver1.solve(f1_ex);
        VectorXd u2 = solver2.solve(f2_ex);
        // std :: cout << " ici " << std :: endl ;

        VectorXd U = Concatener(u1, u2); 
        VectorXd g = Concatener(g1, g2); 
        VectorXd Uf = Concatener(U, g); 

        VectorXd Zero1 = VectorXd::Zero(Ny-1);
        VectorXd Zero2 = VectorXd::Zero(Ny-1);

        // printf("norm Zero1 = %f \n", Zero1.norm());

        // Construction des conditions aux limites pour Schwarz

        VectorXd f1 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1);
        VectorXd f2 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2);
 
        VectorXd C_sup = Concatener(f1, f2);

        VectorXd C = C_sup ; 

        // SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver;
        LeastSquaresConjugateGradient<SparseMatrix<double>> solver;
        solver.setMaxIterations(1000);  // Ajuster si nécessaire
        solver.setTolerance(1e-10);     // Ajuster si nécessaire


        solver.compute(Mf);

        if (solver.info() != Success) {
        std::cerr << "Erreur de décomposition QR !" << std::endl;
        return -1;
        }



        VectorXd X0 = solver.solve(C);
        // VectorXd X0 = Uf; 

        cout << "X0.norm() = : " << X0.norm() << endl;

        solver1.compute(M1);
        solver2.compute(M2);
        
        VectorXd u1_ex = solver1.solve(f1_ex);
        VectorXd u2_ex = solver2.solve(f2_ex);


        // Construction de la matrice A pour le calcul matriciel de gradJ
        SparseMatrix<double> A(X0.size(), X0.size());
        std::vector<Triplet<double>> coefficients;

        for (int j = 1; j < Ny; j++) {
            for (int i = 1; i < Nx1 + Nr; ++i) {
                if (i >= Nx2 - Nr + 1 && i < Nx1 + Nr) {
                    int k1 = (j-1) * (N1+Nr -1) + (i-(Nx2-Nr+1)) + (N1-Nr+1); 
                    int k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-(Nx2-Nr+1));
                    
                    // printf("k1 = %d, k2 = %d",k1,k2);
                    coefficients.emplace_back(k1, k1, 1.0);
                    coefficients.emplace_back(k2, k2, 1.0);
                }
            }

            
        }
        A.setFromTriplets(coefficients.begin(), coefficients.end());

        // Construction du vecteur Xi
        VectorXd Xi = VectorXd::Zero(X0.size());
        // Remplissage de Xi avec les termes de couplage
        for (int j = 1; j < Ny; j++) {
            for (int i = 1; i < Nx1 + Nr; ++i) {
                if (i >= Nx2 - Nr + 1 && i < Nx1 + Nr) {
                    int k1 = (j-1) * (N1+Nr -1) + (i-(Nx2-Nr+1)) + (N1-Nr+1); 
                    int k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-(Nx2-Nr+1));
                    Xi[k1] = 0.5 * X0[k2];  // Facteur 0.5 pour compenser le gradient
                    Xi[k2] = 0.5 * X0[k1];
                }
            }
        }

        // Paramètres de l'algorithme
        double epsilon = 1e-6;
        double NormeGp = 1.0;
        int iter = 0;
        double nu = 0.1; // Pas de descente
        // double nu = std::min(0.1, 1.0/(double)N); // Scales with mesh size
        if (N >= 40) {
            nu = 1.0;
        }

        SparseLU<SparseMatrix<double>> solver_lambda;
        solver_lambda.compute(Mf * Mf.transpose());
        while (NormeGp > epsilon)
        {
            //------------------------------------------------------
            // 1. Calcul du coût courant (optionnel pour suivi)
            //------------------------------------------------------
            double currentCost = 0.0;
            for (int j = 1; j < Ny; j++) {
                for (int i = 1; i< Nx1 + Nr; ++i) {
                    int k = (j-1)*(Nx1+Nr-1) + i-1;
                    if (i >= Nx2 - Nr+1  && i < Nx1+Nr) {
                        int k1 = (j-1) * (N1+Nr -1) + (i-(Nx2-Nr+1)) + (N1-Nr+1); 
                        int k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-(Nx2-Nr+1));
                        currentCost += (X0[k1] - X0[k2]) * (X0[k1] - X0[k2]);

                    }
                }
            }
            // printf("ICI \n");
            
            //------------------------------------------------------
            // 2. Calcul du gradient « non projeté » gradJ = AX - Xi
            //------------------------------------------------------
            
            // Mise à jour de Xi avec les valeurs courantes de X0
            for (int j = 1; j < Ny; j++) {
                for (int i = 1; i< Nx1 + Nr; ++i) {
                    int k = (j-1)*(Nx1+Nr-1) + i-1;
                    if (i >= Nx2 - Nr +1  && i < Nx1+Nr) {
                        int k1 = (j-1) * (N1+Nr -1) + (i-(Nx2-Nr+1)) + (N1-Nr+1); 
                        int k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-(Nx2-Nr+1));
                        // printf("k1 = %d , k2 = %d \n" ,k1,k2);
                        Xi[k1] = X0[k2];  // Le terme -1.0 * X0[k2]
                        Xi[k2] = X0[k1];  // Le terme -1.0 * X0[k1]                
                    }
                }
            }
        
            
            VectorXd gradJ = A*X0 - Xi ;


            {
                // On résout MfMfᵗ λ = Mf * gradJ
                VectorXd lambda = solver_lambda.solve(Mf * gradJ);

                // Gradient projeté (Gp dans le cours)
                gradJ = gradJ - Mf.transpose() * lambda; 
                
            }

            //------------------------------------------------------
            // 5) Norme du gradient projeté (pour le critère d'arrêt)
            //------------------------------------------------------
            NormeGp = gradJ.norm();
            
            //------------------------------------------------------
            // 6) Test d'arrêt
            //------------------------------------------------------
            if (NormeGp < epsilon) {
                std::cout << "Convergence atteinte à l’itération " << iter << std::endl;
                break;
            }

            //------------------------------------------------------
            // 7) Mise à jour de X^(n+1) = X^n - alpha * Gp
            //------------------------------------------------------
            X0 = X0 - nu * gradJ;

            //------------------------------------------------------
            // 8) Affichage / suivi
            //------------------------------------------------------

            if (iter % 50 == 0) {
                // std::cout << "Itération " << iter
                //         << " : Cost = " << currentCost
                //         << ", ||Gp|| = " << NormeGp << std::endl;
            }
            // NormeGp = 1e-8;
            
            iter++;
        }
        
        VectorXd Xf = X0.head(X0.size() - 2*(Ny-1));  // enlève g1 et g2 

        // After while loop:
        int size1 = (Nx1 + Nr - 1) * (Ny - 1);
        int size2 = (Nx - Nx2 + Nr - 1) * (Ny - 1);

        VectorXd C_hat = Mf*X0 ; 
        double Error_hat = 0;
        // std::cout << "Error hat = " << (C_hat - C).norm() << std::endl;


        double error1 = calcul_erreur_L2(Xf.head(size1), Ny, 1, Nr, dx, dy, Nx); 
        double error2 = calcul_erreur_L2(Xf.tail(size2), Ny, 2, Nr, dx, dy, Nx);

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

        // Pour le domaine 1
        for(int j = 1; j < Ny; ++j) {
            for(int i = 1; i < Nx1+Nr; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                int k = (j-1) * (Nx1 + Nr-1) + i-1;
                solution1 << x << " " << y << " " << X0[k] << endl;
            }
            solution1 << endl; // Pour gnuplot splot
        }

        // Pour le domaine 2
        for(int j = 1; j < Ny; ++j) {
            for(int i = Nx2-Nr+1; i < Nx; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                int k = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-Nx2+Nr-1);
                solution2 << x << " " << y << " " << X0[k] << endl;
            }
            solution2 << endl;
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
        // fichier.close();

    }