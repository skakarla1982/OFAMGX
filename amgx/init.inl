/**
 * @file init.inl
 */
# include "AmgXSolver.hpp"

/* CUDA error macro */
#define CUDA_SAFE_CALL(call) do {                                 \
  cudaError_t err = call;                                         \
  if(cudaSuccess != err) {                                        \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString( err) );       \
    exit(EXIT_FAILURE);                                           \
  } } while (0)

/**
 * @brief Initialization of AmgXSolver instance
 *
 * @param _mode The mode this solver will run in. Please refer to AmgX manual.
 * @param cfg_ A string containing the configurations of this solver
 *
 * @return Currently meaningless. May be error codes in the future.
 */
int AmgXSolver::initialize(const std::string &_mode, const std::string &cfg)
{
    // get the number of total cuda devices
    CUDA_SAFE_CALL(cudaGetDeviceCount(&nDevs));

    // Check whether there is at least one CUDA device on this node
    if (nDevs == 0) 
    {
        std::cerr << "There are no CUDA devices on the node " << std::endl;

        exit(EXIT_FAILURE);
    }

    // initialize other communicators
    initMPIcomms();

    if (rank == 0) initAmgX(_mode, cfg);

    return 0;
}


/**
 * @brief Initialize MPI communicator
 * 
 * @return Currently meaningless. May be error codes in the future.
 */
int AmgXSolver::initMPIcomms()
{
    MPI_Init(NULL, NULL);
    MPI_Comm_size(amgx_mpi_comm, &nranks);
    MPI_Comm_rank(amgx_mpi_comm, &rank);

    //CUDA GPUs
    lrank = rank % nDevs;
    CUDA_SAFE_CALL(cudaSetDevice(lrank));
    
    return 0;
}



/**
 * @brief Initialize the AmgX library
 *
 * This function initializes the current instance (solver). Based on the count, 
 * only the instance initialized first is in charge of initializing AmgX and the 
 * resource instance.
 *
 * @param _mode The mode this solver will run in. Please refer to AmgX manual.
 * @param _cfg A string containing the configurations of this solver
 *
 * @return Currently meaningless. May be error codes in the future.
 */
int AmgXSolver::initAmgX(const std::string &_mode, const std::string &_cfg)
{
    count += 1;

    // only the first instance (AmgX solver) is in charge of initializing AmgX
    if (count == 1)
    {
        // initialize AmgX
        AMGX_SAFE_CALL(AMGX_initialize());

        // intialize AmgX plugings
        AMGX_SAFE_CALL(AMGX_initialize_plugins());

        // use user-defined output mechanism. only the master process can output
        // something on the screen
        if (rank == 0) 
        { 
            AMGX_SAFE_CALL(
                AMGX_register_print_callback(&(AmgXSolver::print_callback))); 
        }
        else 
        { 
            AMGX_SAFE_CALL(
                AMGX_register_print_callback(&(AmgXSolver::print_none))); 
        }

        // let AmgX to handle errors returned
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    // create an AmgX configure object
    // TODO Use AmgX config string
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, ""));

    // AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, _cfg.c_str()));

    // let AmgX handle returned error codes internally
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    // create an AmgX resource object, only the first instance is in charge
    if (count == 1) AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &rank);
    
    // set mode
    setMode(_mode);

    // create AmgX vector object for unknowns and RHS
    AMGX_SAFE_CALL(AMGX_vector_create(&AmgXP, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&AmgXRHS, rsrc, mode));

    // create AmgX matrix object for unknowns and RHS
    AMGX_SAFE_CALL(AMGX_matrix_create(&AmgXA, rsrc, mode));

    // create an AmgX solver object
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, cfg));

    // obtain the default number of rings based on current configuration
    AMGX_SAFE_CALL(AMGX_config_get_default_number_of_rings(cfg, &ring));

    isInitialized = true;

    return 0;
}


/**
 * @brief Finalizing the instance.
 *
 * This function destroys AmgX data. The instance last destroyed also needs to 
 * destroy shared resource instance and finalizing AmgX.
 *
 * @return Currently meaningless. May be error codes in the future.
 */
int AmgXSolver::finalize()
{

    if (rank == 0)
    {
        // destroy solver instance
        AMGX_solver_destroy(solver);

        // destroy matrix instance
        AMGX_matrix_destroy(AmgXA);

        // destroy RHS and unknown vectors
        AMGX_vector_destroy(AmgXP);
        AMGX_vector_destroy(AmgXRHS);

        // only the last instance need to destroy resource and finalizing AmgX
        if (count == 1)
        {
            AMGX_resources_destroy(rsrc);
            AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

            AMGX_SAFE_CALL(AMGX_finalize_plugins());
            AMGX_SAFE_CALL(AMGX_finalize());
        }
        else
        {
            AMGX_config_destroy(cfg);
        }

        // change status
        isInitialized = false;

        count -= 1;
    }

    MPI_Finalize();
    CUDA_SAFE_CALL(cudaDeviceReset());
    
    return 0;
}

