/**
 * @file AmgXSolver.hpp
 * @brief Declaration of the class AmgXSolver
 * @author Pi-Yueh Chuang (pychuang@gwu.edu)
 * @version alpha
 * @date 2015-09-01
 */

# pragma once

# include <iostream>
# include <string>
# include <cstring>
# include <functional>
# include <vector>
# include <cstdlib>
# include <cstddef>

# include <cuda_runtime.h>
# include <amgx_c.h>

# include <mpi.h>

# include "check.hpp"

/**
 * @brief A wrapper class for an interface between PETSc and AmgX
 *
 * This class is a wrapper of AmgX library. PETSc user only need to pass 
 * PETSc matrix and vectors into the AmgXSolver instance to solve their problem.
 *
 */
namespace OFAmgX
{

    class AmgXSolver
    {
        public:

            /// default constructor
            AmgXSolver() = default;

            /// initialization of instance
            int initialize(const std::string &_mode, const std::string &cfg_file);

            /// finalization
            int finalize();

            int setA(const int* row,
                     const int* col,
                     const double* data,
                     const int n);

            /// solve the problem, soultion vector will be updated in the end
            int solve(const double* rhs, double* unks, int size);

            /// Get the number of iterations of last solve phase
            int getIters();

            /// Get the residual at a specific iteration in last solve phase
            double getResidual(const int &iter);

            /// get the memory usage on device
            int getMemUsage();


        private:



            int                     nDevs,      /*< # of cuda devices*/
                                    devID;


            MPI_Comm                amgx_mpi_comm = MPI_COMM_WORLD;


            int                     rank = 0,
                                    lrank = 0,
                                    nranks = 0,
                                    gpu_count = 0;


            static int              count;      /*!< only one instance allowed*/
            int                     ring=2;       /*< a parameter used by AmgX*/


            AMGX_Mode               mode;               /*< AmgX mode*/
            AMGX_config_handle      cfg = NULL;      /*< AmgX config object*/
            AMGX_matrix_handle      AmgXA = NULL;    /*< AmgX coeff mat*/
            AMGX_vector_handle      AmgXP = NULL,    /*< AmgX unknowns vec*/
                                    AmgXRHS = NULL;  /*< AmgX RHS vec*/
            AMGX_solver_handle      solver = NULL;   /*< AmgX solver object*/
            static AMGX_resources_handle   rsrc;        /*< AmgX resource object*/


            bool                    isInitialized = false;  /*< as its name*/


            /// set up the mode of AmgX solver
            int setMode(const std::string &_mode);

            /// a printing function using stdout
            static void print_callback(const char *msg, int length);

            /// a printing function that prints nothing, used by AmgX
            static void print_none(const char *msg, int length);

            int initMPIcomms();
            
            int initAmgX(const std::string &_mode, const std::string &_cfg);
    };

}
