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
    N[0] = floor(Nx / 2);  // Partie gauche
    N[1] = Nx - N[0] + 1;      // Partie droite
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
                // if (i == Nx1 + Nr - 1) {
                //     B(k) += -g2(j-1)/(dx*dx);  // Correction de l'indice
                // }
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
                // if (i == Nx2-Nr+1) {
                //     B(k) += -g1(j-1)/(dx*dx);  // Correction de l'indice
                // }
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
            std::cout << "Element (" << it.row() << ", " << it.col() 
                      << ") = " << it.value() << std::endl;
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
    // Dimensions des matrices d'entrÃ©e
    int sizeA1 = A1.rows();
    int sizeA2 = A2.rows();

    // Dimension totale de la matrice diagonale par blocs
    int totalSize = sizeA1 + sizeA2;

    // Matrice diagonale par blocs
    SparseMatrix<double> bigMatrix(totalSize, totalSize);

    // Stockage des coefficients pour la matrice diagonale par blocs
    std::vector<Triplet<double>> coefficients;

    // Ajouter les coefficients de A1 dans le bloc supÃ©rieur gauche
    for (int k = 0; k < A1.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A1, k); it; ++it) {
            coefficients.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Ajouter les coefficients de A2 dans le bloc infÃ©rieur droit
    for (int k = 0; k < A2.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A2, k); it; ++it) {
            coefficients.emplace_back(it.row() + sizeA1, it.col() + sizeA1, it.value());
        }
    }

    // Construire la matrice finale Ã  partir des triplets
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
    // vector<int> N_values = {40};
    vector<int> N_values = {10};
    ofstream convergence1("convergence_domaine1.txt");
    ofstream convergence2("convergence_domaine2.txt");
    
    double previous_h = 0;
    double previous_error1 = 0;
    double previous_error2 = 0;
    
    cout << "Analyse de convergence par sous-domaine" << endl;
    cout << "----------------------------------------" << endl;
    
    for(int N : N_values) {
        int Nx = N, Ny = 6;
        int Nr = 2;
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

        // std::cout << "Dimensions de B (SparseMatrix): " << (B1).rows() << " x " << (B1).cols() << std::endl;

        SparseMatrix<double> M1 = Matrice(Nx1 + Nr, Ny, alpha, beta, gamma);
        SparseMatrix<double> M2 = Matrice(Nx-Nx2+ Nr , Ny, alpha, beta, gamma);
        SparseMatrix<double> M = Matrice_globale(M1 , M2) ;

        SparseMatrix<double> zero1 = Matricenulle(Nx1 + Nr, Ny);
        SparseMatrix<double> zero2 = Matricenulle(Nx-Nx2+ Nr , Ny);
        SparseMatrix<double> Id1 = MatriceIdentité(Nx1 + Nr, Ny);
        SparseMatrix<double> Id2 = MatriceIdentité(Nx-Nx2+ Nr , Ny);
        SparseMatrix<double> zero_1 = Matrice_globale(zero1 , zero2) ;
        SparseMatrix<double> zero_2 = Matrice_globale(zero1 , zero2) ;




        SparseMatrix<double> Mf_sup = Matrice_globale_horizontale(M, zero_1 );
        // SparseMatrix<double> Mf_inf = Matrice_globale_horizontale(zero_1, zero_2 );
        // SparseMatrix<double> Mf = empilerVerticalement(Mf_sup, Mf_inf );
        SparseMatrix<double> Mf = Mf_sup;

        SparseLU<SparseMatrix<double>> solver1, solver2;

        // std :: cout << " ici " << std :: endl ;
        solver1.compute(M1);
        solver2.compute(M2);
        // std :: cout << " ici " << std :: endl ;



        VectorXd g1(Ny-1), g2(Ny-1);  
        for (int j = 1; j < Ny; ++j) {
            double x1 = xmin + (Nx1+Nr-1)*dx;
            double y = ymin + j*dy;
            g1(j-1) = 1e4;
            
            double x2 = xmin + (Nx2-Nr+1)*dx;
            g2(j-1) = 2e4;
        }

        VectorXd f1 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, g1, g2);
        VectorXd f2 = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, g1, g2);

        VectorXd u1 = solver1.solve(f1);
        VectorXd u2 = solver2.solve(f2);
        // std :: cout << " ici " << std :: endl ;

        VectorXd U = Concatener(u1, u2); 
        VectorXd g = Concatener(g1, g2); 
        VectorXd Uf = Concatener(U, g); 

        VectorXd Zero2((Nx1 + Nr-1) * (Ny-1)), Zero1((Nx-Nx2+ Nr-1) * (Ny-1));

        // Construction des conditions aux limites pour Schwarz
        VectorXd g1_ex((Ny-1)), g2_ex((Ny-1)); 
        for (int j = 1; j < Ny; ++j) {
            double x1 = xmin + (Nx1+Nr-1)*dx;
            double y = ymin + j*dy;
            g1_ex(j-1) = solution_exacte(x1,y);
            // g1_ex(j-1) = 0.0;
            
            double x2 = xmin + (Nx2-Nr+1)*dx;
            g2_ex(j-1) = solution_exacte(x2,y);
            // g2_ex(j-1) = 0.0;
        }
        
        VectorXd f1_ex = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, g1_ex, g2_ex);
        VectorXd f2_ex = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, g1_ex, g2_ex);
        // VectorXd f1_ex = second_membre(N1+Nr, Ny, dx, dy);
        // VectorXd f2_ex = second_membre(N2+Nr, Ny, dx, dy);

        // VectorXd C_sup = Concatener(f1_ex, f2_ex);
        VectorXd C_sup = Concatener(f1, f2);
        // VectorXd C_inf = Concatener(Zero1, Zero2);
        // VectorXd C = Concatener(C_sup, C_inf);
         VectorXd C = C_sup ; 

        std::cout << "Dimensions de Mf (SparseMatrix): " << (Mf).rows() << " x " << (Mf).cols() << std::endl;
        std::cout << "Dimensions de Uf (SparseMatrix): " << (Uf).rows() << std::endl;
        std::cout << "Dimensions de C (SparseMatrix): " << (C).rows() << std::endl;



        // Mf.makeCompressed();

        SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver;
        solver.compute(Mf);

        if (solver.info() != Success) {
        std::cerr << "Erreur de décomposition QR !" << std::endl;
        return -1;
        }


    
        VectorXd X0 = solver.solve(C);
     
        solver1.compute(M1);
        solver2.compute(M2);
        
        VectorXd u1_ex = solver1.solve(f1_ex);
        VectorXd u2_ex = solver2.solve(f2_ex);


        VectorXd U_ex = Concatener(u1_ex, u2_ex);
        double cost;
        for ( int i = 1 ; i < 2*(Nr-1) +1; i++){
                for (int j = 1 ; j<Ny ; j++){
                    int k1 = (j-1) * (N1+Nr -1) + (i ) + (N1-Nr); 
                    int k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-1);
                    // printf("k1 = %d, k2 = %d \n", k1, k2);
                    cost += (X0[k1] - X0[k2]) * (X0[k1] - X0[k2]); 
                }
        }

        for ( int i = (N1+N2 +2*Nr -2 + Ny) ; i < (N1+N2 +2*Nr -2 + Ny) + ( Ny - 1) ; i++){
            int j = i - (N1+N2 +2*Nr -2 + Ny) ; 
            double y = ymin + (j+1)*dy;
            double x2 = xmin + (Nx2-Nr+1)*dx;
            g2_ex(j) = solution_exacte(x2,y);
            std :: cout << X0[i] << " " << g2_ex(j) << std ::endl ;
        }


        std::cout << "cost avant optimisation : " << cost<< std::endl;

        VectorXd Zero = Concatener(Zero1, Zero2); 

        VectorXd U_xi( (Ny-1)* (N1+N2 + 2*Nr - 2)) ;

        for ( int  i = 0 ; i<(Ny-1)* (N1+N2 + 2*Nr - 2) ; i++){
            if ( i > (Ny-1)*(N1 -Nr +1) && i <  (Ny-1)*(N1 -Nr +1) + (2*Nr * (Ny -1) )){
                U_xi[i] = U_ex[i] ;
            }

            else {
                 U_xi[i] = 0.;
            }

        } 



        // VectorXd Xi = Concatener(U_ex , Zero) ;
        VectorXd Xi = Concatener(U_xi , Zero) ;
        // VectorXd Xi = U_ex ; 

        std::cout << "Dimensions de Xi (SparseMatrix): " << (Xi).rows() << std::endl;

        

       
        SparseMatrix<double> Matrice_nulle0 = Matricenulle(Nx1 + Nr, Ny);
        SparseMatrix<double> Matrice_nulle1 = Matricenulle(Nx - Nx2  +Nr , Ny);
        SparseMatrix<double> Matrice_nulle = Matrice_globale(Matrice_nulle0 , Matrice_nulle1) ;



        SparseMatrix<double> I1 = MatriceIdentité(Nx1 + Nr, Ny);
        SparseMatrix<double> I2 = MatriceIdentité(Nx - Nx2  +Nr , Ny);
        SparseMatrix<double> I = Matrice_globale(I1 , I2) ;


        // Revoir la forme de A
        SparseMatrix<double> A = Matrice_globale(I , Matrice_nulle) ;
        //  std::cout << "Dimensions de A (SparseMatrix): " << (A).rows() << " x " << (A).cols() << std::endl;

       
        
        double Norme_grad = 1. ; 
        int iter = 0 ;

        std:: cout << "Début de la boucle de l'optimisation" << std :: endl ;
        std::ofstream fichier("resultats.dat");

        while ( Norme_grad > 1e-6){
            // SparseLU<SparseMatrix<double>> solver;

            double Cost = 0. ; 
            for ( int i = 1 ; i < 2*(Nr-1) +1; i++){
                for (int j = 1 ; j<Ny ; j++){
                    int k1 = (j-1) * (N1+Nr -1) + (i ) + (N1-Nr); 
                    int k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-1);
                    // printf("k1 = %d, k2 = %d \n", k1, k2);
                    Cost += (X0[k1] - X0[k2]) * (X0[k1] - X0[k2]); 
                    if ( i == 2 && j == 2) {
                        // std :: cout << X0[k1] << " X0(k1) " << X0[k2] << " X0(k2) " << std :: endl ;
                    }
                    

                }
            }

            // std :: cout << Norme_grad << " " << iter << " " << Cost << std :: endl ;

            SparseMatrix<double> Mf_T = Mf * Mf.transpose();
            solver.compute(Mf_T);


            //  // Affichage des dimensions
            // std::cout << "Dimensions de A (SparseMatrix): " << (A).rows() << " x " << (A).cols() << std::endl;
            // std::cout << "Dimensions de Mf (SparseMatrix): " << (Mf).rows() << " x " << (Mf).cols() << std::endl;

            // std::cout << "Dimensions de b (VectorXd): " << Mf.rows() << " x " << Mf.cols() << std::endl;
            // std::cout << "Dimensions de X0 (VectorXd): " << X0.rows() << std::endl;
            // std::cout << "Dimensions de Xi (VectorXd): " << Xi.rows() << std::endl;

            VectorXd SM = Mf * (A* X0 - Xi) ; 

            
            VectorXd lambda = solver.solve(SM);
    

            VectorXd Grad = A * X0 - Xi - Mf.transpose() * lambda ;
            
            
            // VectorXd Test = Mf * Grad ;
            // std ::cout << Test.norm() << " doit etre nulle " << std :: endl ;

            Norme_grad = Grad.norm() ;
            std :: cout << Norme_grad << " " << iter << " " << Cost << std :: endl ;
            if (Norme_grad > 1e-3){
                X0 = X0 - 1e-1 * Grad ; 
            }

            for ( int i = (N1+N2 +2*Nr -2 + Ny) ; i < (N1+N2 +2*Nr -2 + Ny) + ( Ny - 1) ; i++){
                int j = i - (N1+N2 +2*Nr -2 + Ny) ; 
                double y = ymin + (j+1)*dy;
                double x2 = xmin + (Nx2-Nr+1)*dx;
                g2_ex(j) = solution_exacte(x2,y);
                // std :: cout << X0[i] << " " << g2_ex(j) << std ::endl ;
            }

            
            fichier << iter << " " << Norme_grad << std::endl; 
            // Norme_grad = 1e-4 ;
            iter += 1 ;

            // Eigen::VectorXd U1 = x1.segment(0, (Nx1 + Nr-1) * (Ny-1)); 
            // (Nx-Nx2+ Nr-1) * (Ny-1)
            // Eigen::VectorXd U2 = x1.segment((Nx-Nx2+ Nr-1) * (Ny-1), ); 


            // VectorXd Zero = Concatener(Zero1, Zero2); 

            // Xi = Concatener(U , Zero) ;


        }

        fichier.close();


            

        // SparseMatrix<double> Mf = MatriceMf(A1 , A2, B1, B2);
        // afficherSparseMatrix(Mf);

        // Construction des conditions aux limites pour Schwarz
        // for (int j = 1; j < Ny; ++j) {
        //     // int k1 = ( j-1) * (Nx1 + Nr -1 ) + (2*Nr -1) + (Nx1 - Nr -1) ; 
        //     // int k1 = ( j-1) * (Nx+Nr) + (2*Nr -1) + (Nx1 - Nr ) ; 
        //     // int k2 = 2*Nr + (Nx-Nx2+ Nr)*(j-1) ;
        //     int k1 = (j-1) * (N1+Nr -1) + (2*Nr - 1) + (N1-Nr - 1); 
        //     int k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) ;
 

        //     std :: cout << k1 << " " << k2 << std :: endl ;
        //     // g1(j-1) = solution_exacte(x1,y);
            
        //     // double x2 = xmin + (Nx2-Nr+1)*dx;
        //     // g2(j-1) = solution_exacte(x2,y);
        // }
    }
    
    return 0;
}