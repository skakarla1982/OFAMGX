/**
 * @file setA.cpp
 */
 
# include "AmgXSolver.hpp"

/**
 * @brief A function convert PETSc Mat into AmgX matrix and bind it to solver
 *
 * This function will first extract the raw data from PETSc Mat and convert the
 * column index into 64bit integers. It also create a partition vector that is 
 * required by AmgX. Then, it upload the raw data to AmgX. Finally, it binds
 * the AmgX matrix to the AmgX solver.
 *
 * Be cautious! It lacks mechanism to check whether the PETSc Mat is AIJ format
 * and whether the PETSc Mat is using the same MPI communicator as the 
 * AmgXSolver instance.
 *
 * @param A A PETSc Mat. The coefficient matrix of a system of linear equations.
 * The matrix must be AIJ format and using the same MPI communicator as AmgX.
 *
 * @return Currently meaningless. May be error codes in the future.
 */
int AmgXSolver::setA(const int* row,
                     const int* col,
                     const double* data,
                     const int n)
{
    // upload matrix A to AmgX
      std::cout << "HEY 0\n";
    AMGX_matrix_upload_all(
            AmgXA, n, row[n], 1, 1, 
            row, col, data, NULL);
            
    std::cout << "HEY 1\n";
    // bind the matrix A to the solver
    AMGX_SAFE_CALL(AMGX_solver_setup(solver, AmgXA));
    std::cout << "HEY 2\n";
    // connect (bind) vectors to the matrix
    AMGX_vector_bind(AmgXP, AmgXA);
    AMGX_vector_bind(AmgXRHS, AmgXA);
    std::cout << "HEY 3\n";
    return 0;
}

