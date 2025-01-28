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
    N[1] = N[0]+1;
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


//fonction à minimiser
double I(VectorXd U, int N1, int N2,int Nr, int Ny)
{
    double r;
    int k1,k2;
    for(int j=1; j<Ny; ++j){
        for (int i = 1; i < 2*Nr-1; i++)
        {
            k1 = (j-1) * (N1+Nr -1) + (i -1) + (N1-Nr+1);
            k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-1);  
            r+=(U[k1]-U[k2])*(U[k1]-U[k2]);  
        }        
    }
    return r;
}

//gradient I par rapport a u
VectorXd dI_dU(VectorXd  U,int N1, int N2, int Ny, int Nr){
    int taille_globale=U.size();
    VectorXd der(taille_globale);
    der.setConstant(0.0);
    int k1,k2;
    
    for (int j = 1; j < Ny; j++) {
            for (int i = 1; i < N1 + Nr; ++i) {
                if (i >= N1+1 - Nr + 1 && i < N1 + Nr) {
                    int k1 = (j-1) * (N1+Nr -1) + (i-(N1+1-Nr+1)) + (N1-Nr+1); 
                    int k2 = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-(N1+1-Nr+1));
                    
                    // printf("k1 = %d, k2 = %d, u k1 %lf , u k2 %lf\n",k1,k2, U[k1], U[k2]); 
                    der[k1]=2*(U[k1]-U[k2]);
                   der[k2]=-2*(U[k1]- U[k2]);
                }
            }
        }


    return der;
}

VectorXd Concatener(VectorXd u1, VectorXd  u2){
    int N1= u1.size(), N2=u2.size();
    int N=N1+N2;
    VectorXd u(N);
    for(int k=0; k<N; ++k){
        if (k<N1){
            u[k]=u1[k];
        }
        else{
            u[k]=u2[k-N1];
        }
    } 
    return u;  
}


// Fonction pour créer une matrice diagonale par blocs  de diag A1 et A2
SparseMatrix<double> Matrice_globale(const SparseMatrix<double>& A1, const SparseMatrix<double>& A2) {
    // Dimensions des matrices d'entrée
    int sizeA1 = A1.rows();
    int sizeA2 = A2.rows();

    // Dimension totale de la matrice diagonale par blocs
    int totalSize = sizeA1 + sizeA2;

    // Matrice diagonale par blocs
    SparseMatrix<double> bigMatrix(totalSize, totalSize);

    // Stockage des coefficients pour la matrice diagonale par blocs
    std::vector<Triplet<double>> coefficients;

    // Coefficients de A1 dans le bloc supérieur gauche
    for (int k = 0; k < A1.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A1, k); it; ++it) {
            coefficients.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Coefficients de A2 dans le bloc inférieur droit
    for (int k = 0; k < A2.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A2, k); it; ++it) {
            coefficients.emplace_back(it.row() + sizeA1, it.col() + sizeA1, it.value());
        }
    }

    // Construire la matrice finale à partir des triplets
    bigMatrix.setFromTriplets(coefficients.begin(), coefficients.end());

    return bigMatrix;
}

SparseMatrix<double> Jacob1(int N1, int nr, int Ny, double beta)
{
    int taille_ligne = (N1 + nr-1) * (Ny - 1);
    int taille_colonne = (Ny - 1);
    // printf("taille_ligne %d, taille_colone %d\n", taille_ligne, taille_colonne);
    SparseMatrix<double> J1(taille_ligne, taille_colonne); 
    std::vector<Triplet<double>> coefficients; 

    // Remplissage pour les Ny-1 premières lignes    
    for(int ligne=N1+nr-2; ligne<(Ny - 1)*(N1+nr-1); ligne+=N1+nr-1){
        int colonne = ligne/(N1+nr-1);              
        if (colonne < Ny - 1) { 
            coefficients.emplace_back(ligne, colonne, -beta);
            // std::cout << "Ajout 1ème bloc : ligne = " << ligne << ", colonne = " << colonne << ", valeur = " << -beta << "\n";
        }
    }
    J1.setFromTriplets(coefficients.begin(), coefficients.end());
    return J1;
}

SparseMatrix<double> Jacob2(int N2, int nr, int Ny, double beta)
{
    int taille_ligne = (N2 + nr-1) * (Ny - 1);
    int taille_colonne = (Ny - 1);
    // printf("taille_ligne %d, taille_colone %d\n", taille_ligne, taille_colonne);
    SparseMatrix<double> J(taille_ligne, taille_colonne); 
    std::vector<Triplet<double>> coefficients; 

    // Remplissage pour les Ny-1 premières lignes    
    for(int ligne=0; ligne<(Ny - 1)*(N2+nr-1); ligne+=N2+nr-1){
        int colonne = ligne/(N2+nr-1);              
        if (colonne < Ny - 1) { 
            coefficients.emplace_back(ligne, colonne, -beta);
            // std::cout << "Ajout 2ème bloc : ligne = " << ligne << ", colonne = " << colonne << ", valeur = " << -beta << "\n";
        }
    }
    J.setFromTriplets(coefficients.begin(), coefficients.end());
    return J;
}



SparseMatrix<double> construireJacobienne(int N1, int N2, int nr, int Ny, double beta) {
    // Construction des matrices J1 et J2
    SparseMatrix<double> J1 = Jacob1(N1, nr, Ny, beta);
    SparseMatrix<double> J2 = Jacob2(N2, nr, Ny, beta);
    
    // Dimensions des matrices J1 et J2
    int rows1 = J1.rows(), cols1 = J1.cols();
    int rows2 = J2.rows(), cols2 = J2.cols();

    // Vérification des dimensions
    if (cols1 != cols2 || cols1 != Ny - 1) {
        throw std::runtime_error("Le nombre de colonnes dans J1 et J2 est incohérent.");
    }

    // Dimensions de la matrice globale
    int totalRows = rows1 + rows2;
    int totalCols = cols1 + cols2; // 2 * (Ny - 1)

    // Création de la matrice globale
    SparseMatrix<double> BlockMatrix(totalRows, totalCols);

    // Vecteur pour stocker les triplets
    std::vector<Triplet<double>> coefficients;

    // Ajout des coefficients de J1
    for (int k = 0; k < J1.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(J1, k); it; ++it) {
            if (it.row() < rows1 && it.col() < cols1) {
                coefficients.emplace_back(it.row(), it.col(), it.value());
            } else {
                throw std::runtime_error("Indice de J1 hors limites !");
            }
        }
    }

    // Ajout des coefficients de J2
    for (int k = 0; k < J2.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(J2, k); it; ++it) {
            int newRow = it.row() + rows1; // Décalage des lignes
            int newCol = it.col() + cols1; // Décalage des colonnes
            if (newRow < totalRows && newCol < totalCols) {
                coefficients.emplace_back(newRow, newCol, it.value());
            } else {
                throw std::runtime_error("Indice de J2 hors limites !");
            }
        }
    }

    // Construction de la matrice globale
    BlockMatrix.setFromTriplets(coefficients.begin(), coefficients.end());

    return BlockMatrix;
}




VectorXd algo_adjoint(const SparseLU<SparseMatrix<double>>& solver,
                                     const SparseLU<SparseMatrix<double>>& solverAT,
                                     VectorXd& g1,
                                     VectorXd& g2,
                                     int Nx, int Ny, int Nr,int N1,int N2, double dx, double dy, double beta,
                                     double epsilon, double nu) {
    VectorXd f1, f2,f;
    VectorXd derive_I_U, lambda, grad;
    VectorXd u,g=Concatener(g1,g2);
    
    // Ouvrir le fichier pour écrire les valeurs de I
    ofstream file_I("valeurs_I.txt");
    if (!file_I.is_open()) {
        cerr << "Erreur: Impossible d'ouvrir le fichier valeurs_I.txt" << endl;
        throw runtime_error("Erreur d'ouverture de fichier");
    }
    
    // Écrire l'en-tête du fichier
    file_I << "# Iteration\tValeur de I" << endl;
   
    double loss;
    double loss_Old=20;
    int iter=0;
    printf("je suis pas dans la boucle while");
    while (iter<10000)
    {
        // printf("je suis dans la boucle while\n");
        f1 = second_membre_schwarz(Nx, Ny, dx, dy,Nr,1,g1,g2);
        f2 = second_membre_schwarz(Nx, Ny, dx, dy,Nr,2, g1,g2);
        f =Concatener(f1,f2);
        u = solver.solve(f);

        // Calculer I pour cette itération
        loss=I(u,N1,N2,Nr,Ny);
        // Écrire dans le fichier
        file_I << iter << "\t" << loss << endl;
       
        derive_I_U=dI_dU(u,N1,N2,Ny,Nr);
        lambda=solverAT.solve(derive_I_U);
        
        grad=lambda.transpose()*construireJacobienne(N1,N2,Nr,Ny,beta);
        if (grad.norm()<epsilon){
            cout << "Convergence atteinte après " << iter + 1 << " itérations." << endl;
            
            // Écrire la valeur finale de I
            file_I << "# Convergence atteinte" << endl;
            file_I << iter + 1 << "\t" << loss << endl;
            break;
        }
        else{
            g=g-nu*grad;
            g1=g.head(Ny-1);
            g2=g.tail(Ny-1);

            // Vérifier la divergence pour construire un pas adaptatif
            double tol = 1.e-3;
            if (std::abs(loss - loss_Old) < tol) {
                nu *= 1.1; 
                // cout << "On augmente le pas " << loss << " " << loss_Old << " " << nu <<endl ;
                if (nu > 0.7){
                    // cout << "On dépasse le pas max" << endl ;
                    nu = 0.2 * nu ;
                }
            } else {
                nu *= 0.8; 
                // cout << "On diminue le pas" << loss << " "<< loss_Old << " " << nu << endl ;
                    if (nu < 0.07){
                    // cout << "On dépasse le pas min" << endl ;
                    nu = 1.2 * nu ;
                    }
            }
        }
        iter+=1;
        loss_Old = loss;

        
    }
    
    cout << "Convergence atteinte après " << iter + 1 << " itérations." << endl;
    // Fermer le fichier
    file_I.close();
    g=Concatener(g1,g2);
    return g;
}


int main() {


    // vector<int> N_values = {20};
    vector<int> N_values = {10,20,40};
    // vector<int> N_values = {40};
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
        int Nr = 2;

    
        int *Ns = charge(Nx);
        int Nx1 = Ns[0];
        int Nx2 = Ns[1];
        int N1=Nx1, N2=Nx-Nx2;
        printf("Nx1 %d, Nx2 %d\n",Nx1,Nx2);
        printf("N1 %d, N2 %d\n",N1,N2);
        VectorXd u1((Ny-1)*(N1+Nr-1)),u2((Ny-1)*(N2+Nr-1));
        VectorXd u, derive_I_U,lambda, f1,f2,f;

        double dx = (xmax - xmin) / (Nx);
        double dy = (ymax - ymin) / (Ny);
        double h = max(dx, dy);
        
        double alpha = -2.0 * (1.0/(dx*dx) + 1.0/(dy*dy));
        double beta = 1.0 / (dy * dy);
        double gamma = 1.0 / (dx * dx);       
        int k1,k2,k;

        u=Concatener(u1,u2);

            // // Initialisation pour le domaine complet
        SparseMatrix<double> A1 = Matrice(N1+Nr, Ny, alpha, beta, gamma);
        SparseMatrix<double> A2 = Matrice(N2+Nr, Ny, alpha, beta, gamma);
        SparseMatrix<double> A=Matrice_globale(A1,A2);
        SparseLU<SparseMatrix<double>> solver, solver1, solver2;
        solver1.compute(A1);
        
        solver2.compute(A2);
        solver.compute(A);

        SparseMatrix<double> AT = A.transpose();
        SparseLU<SparseMatrix<double>> solverAT;
        solverAT.compute(AT);
        VectorXd g1=VectorXd::Random(Ny-1),g2=VectorXd::Random(Ny-1);
        VectorXd g;   
        double nu=0.1,epsilon=1e-6;

        g=algo_adjoint(solver,solverAT,g1,g2,Nx,Ny,Nr,N1,N2,dx,dy,beta,epsilon,nu);
        g1=g.head(Ny-1);
        g2=g.tail(Ny-1);
        f1 = second_membre_schwarz(Nx, Ny, dx, dy,Nr,1,g1,g2);
        f2 = second_membre_schwarz(Nx, Ny, dx, dy,Nr,2, g1,g2);
        f =Concatener(f1,f2);
        u = solver.solve(f);
        // VectorXd g1(Ny-1), g2(Ny-1);  
        // for (int j = 1; j < Ny; ++j) {
        //     double x1 = xmin + (Nx1+Nr-1)*dx;
        //     double y = ymin + j*dy;
        //     g1(j-1) = solution_exacte(x1,y);
        //     double x2 = xmin + (Nx2-Nr+1)*dx;
        //     g2(j-1) = solution_exacte(x2,y);
        // }

        // VectorXd f1_ex = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 1, g1, g2);
        // VectorXd f2_ex = second_membre_schwarz(Nx, Ny, dx, dy, Nr, 2, g1, g2);

        // u1 = solver1.solve(f1_ex);
        // u2 = solver2.solve(f2_ex);

        // u = Concatener(u1, u2); 

        ofstream solution1("solution_domain1.txt");
        ofstream solution2("solution_domain2.txt");

        ofstream exact1("exact_domain1.txt");
        ofstream exact2("exact_domain2.txt");
        // Pour le domaine 1
        for(int j = 1; j < Ny; ++j) {
            for(int i = 1; i < Nx1+Nr; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                int k = (j-1) * (Nx1 + Nr-1) + i-1;
                solution1 << x << " " << y << " " << u[k] << endl;
            }
            solution1 << endl; // Pour gnuplot splot
        }

        // Pour le domaine 2
        for(int j = 1; j < Ny; ++j) {
            for(int i = Nx2-Nr+1; i < Nx; ++i) {
                double x = xmin + i*dx;
                double y = ymin + j*dy;
                int k = (N1 + Nr -1) * (Ny -1) + (j-1) * (N2+Nr-1) + (i-Nx2+Nr-1);
                solution2 << x << " " << y << " " << u[k] << endl;
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
        double Error_hat = 0;

        int size1 = (N1 + Nr - 1) * (Ny - 1);
        int size2 = (N2 + Nr - 1) * (Ny - 1);
        double error1 = calcul_erreur_L2(u.head(size1), Ny, 1, Nr, dx, dy, Nx); 
        double error2 = calcul_erreur_L2(u.tail(size2), Ny, 2, Nr, dx, dy, Nx);

        // Affichage des résultats
        cout << "N = " << N << ", h = " << h << endl;
        // cout << "Domaine 1 - Erreur L2: " << error1;
        // cout << "Domaine 2 - Erreur L2: " << error2;
        // cout << endl;
        if(previous_h > 0) {
            double slope1 = log10(error1/previous_error1) / log10(h/previous_h);
            cout << "\tPente: " << slope1 ;
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


        exact1.close();
        exact2.close();
        solution1.close();
        solution2.close();
    }
    
    convergence1.close();
    convergence2.close();

    return 0;
}