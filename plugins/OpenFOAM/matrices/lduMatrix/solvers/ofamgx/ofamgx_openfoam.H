#pragma once

#include "lduMatrix.H"

#include <assert.h>

namespace OFAmgX {

enum _preconditioner_type {
  OFAmgX_None             = 0,
  OFAmgX_Jacobi           = 1,
  OFAmgX_MultiColoredGS   = 2,
  OFAmgX_MultiColoredSGS  = 3,
  OFAmgX_ILU              = 4,
  OFAmgX_MultiColoredILU  = 5,
  OFAmgX_MultiElimination = 6,
  OFAmgX_FSAI             = 7
};

enum _solver_type {
  OFAmgX_CG         = 0,
  OFAmgX_BiCGStab   = 1,
  OFAmgX_GMRES      = 2,
  OFAmgX_FixedPoint = 3,
  OFAmgX_FGMRES     = 4
};

template <typename DataType>
void allocate_host(const int size, DataType **ptr) {

//  LOG_DEBUG(0, "allocate_host()",
//            size);

  if (size > 0) {

    assert(*ptr == NULL);

    // *********************************************************
    // C++ style
    //    *ptr = new DataType[size];
    // *********************************************************



    // *********************************************************
    // C style
    // *ptr =  (DataType *) malloc(size*sizeof(DataType));
    // *********************************************************



    // *********************************************************
    // C style (zero-set)
    // *ptr = (DataType *) calloc(size, sizeof(DatatType));
    // *********************************************************



    // *********************************************************
    // Aligned allocation
    // total size = (size*datatype) + (alignment-1) + (void ptr)
    // void *non_aligned =  malloc(size*sizeof(DataType)+(MEM_ALIGNMENT-1)+sizeof(void*));
    // assert(non_aligned != NULL);

    // void *aligned = (void*)( ((LONG_PTR)(non_aligned)+MEM_ALIGNMENT+sizeof(void*) ) & ~(MEM_ALIGNMENT-1) );
    // *((void**)aligned-1) = non_aligned;

    // *ptr = (DataType*) aligned;

    // LOG_INFO("A " << *ptr << " " <<  aligned << " " << non_aligned << " "<<  sizeof(DataType) << " " << size);
    // *********************************************************



    // *********************************************************
    // C++ style and error handling

    *ptr = new (std::nothrow) DataType[size];

    if ((*ptr) == false) { // nullptr
        std::cout << "Cannot allocate memory" << std::endl;
//      LOG_INFO("Cannot allocate memory");
//      LOG_VERBOSE_INFO(2, "Size of the requested buffer = " << size*sizeof(DataType));
//      FATAL_ERROR(__FILE__, __LINE__);
    }
    // *********************************************************



    assert(*ptr != NULL);

//    LOG_DEBUG(0, "allocate_host()",
//              *ptr);

  }

}

template <typename DataType>
void free_host(DataType **ptr) {

//  LOG_DEBUG(0, "free_host()",
//            *ptr);

  assert(*ptr != NULL);

  // *********************************************************
  // C++ style
  delete[] *ptr;
  // *********************************************************


  // *********************************************************
  // C style
  // free(*ptr);
  // *********************************************************


  // *********************************************************
  // Aligned allocation
  //  free(*((void **)*ptr-1));
  // *********************************************************

  *ptr = NULL;
}

template <typename DataType>
void set_to_zero_host(const int size, DataType *ptr) {

//  LOG_DEBUG(0, "set_to_zero_host()",
//            "size =" << size <<
//            " ptr=" << ptr);

  if (size > 0) {

    assert(ptr != NULL);

    memset(ptr, 0, size*sizeof(DataType));

    // for (int i=0; i<size; ++i)
    //   ptr[i] = DataType(0);
  }

}


template void allocate_host<float                >(const int size, float                **ptr);
template void allocate_host<double               >(const int size, double               **ptr);
template void allocate_host<int                  >(const int size, int                  **ptr);
template void allocate_host<unsigned int         >(const int size, unsigned int         **ptr);
template void allocate_host<char                 >(const int size, char                 **ptr);

template void free_host<float                >(float                **ptr);
template void free_host<double               >(double               **ptr);
template void free_host<int                  >(int                  **ptr);
template void free_host<unsigned int         >(unsigned int         **ptr);
template void free_host<char                 >(char                 **ptr);

template void set_to_zero_host<float                >(const int size, float                *ptr);
template void set_to_zero_host<double               >(const int size, double               *ptr);
template void set_to_zero_host<int                  >(const int size, int                  *ptr);
template void set_to_zero_host<unsigned int         >(const int size, unsigned int         *ptr);
template void set_to_zero_host<char                 >(const int size, char                 *ptr);


void import_openfoam_matrix(const Foam::lduMatrix &foam_mat,
                            int* row_offset,
                            int* col,
                            double* val,
                            int *size)
{


//  assert(row_offset != NULL && col != NULL && val != NULL);

  // Matrix size
  int n = foam_mat.diag().size();
  *size = n;
  int nnz = foam_mat.diag().size() + foam_mat.lower().size() + foam_mat.upper().size();

//  // CSR values
  row_offset = NULL;
  col = NULL;
  val = NULL;
  
  allocate_host(n+1, &row_offset);
  allocate_host(nnz, &col);
  allocate_host(nnz, &val);
  
  row_offset[0] = 0;
  // initialize row_offsets with ones from OpenFOAM diagonal part
  for (int i=1; i<n+1; ++i)
    row_offset[i] = 1;

  Foam::UList<int>::const_iterator it_low = foam_mat.lduAddr().lowerAddr().begin();
  Foam::UList<int>::const_iterator it_up  = foam_mat.lduAddr().upperAddr().begin();

  // compute nnz per row in lower part
  for (int i=0; i<foam_mat.lower().size(); ++i) {
    row_offset[*it_low + 1]++;
    row_offset[*it_up  + 1]++;
    it_low++;
    it_up ++;
  }

  int sum = 0;
  for (int i=0; i<n; ++i) {
    int temp = row_offset[i];
    row_offset[i] = sum;
    sum += temp;
  }
  row_offset[n] = sum;

  Foam::UList<double>::const_iterator it_val = foam_mat.lower().begin();
  it_low = foam_mat.lduAddr().lowerAddr().begin();
  it_up  = foam_mat.lduAddr().upperAddr().begin();

  // fill col and val arrays for lower part
  for (int i=0; i<foam_mat.lower().size(); ++i) {
    // row index for lower = upper + 1
    int r_lower = *it_up + 1;
    int dest_lower = row_offset[r_lower];

    col[dest_lower] = *it_low;
    val[dest_lower] = *it_val;
    ++row_offset[r_lower];

    it_low++;
    it_up ++;
    it_val++;
  }

  it_val = foam_mat.diag().begin();

  // fill diagonal part
  for (int i=0; i<foam_mat.diag().size(); ++i) {
    int dest_diag = row_offset[i+1];

    val[dest_diag] = *it_val;
    col[dest_diag] = i;
    ++row_offset[i+1];

    it_val++;
  }

  it_low = foam_mat.lduAddr().lowerAddr().begin();
  it_up  = foam_mat.lduAddr().upperAddr().begin();
  it_val = foam_mat.upper().begin();

  // fill upper part
  for (int i=0; i<foam_mat.upper().size(); ++i) {
    // row index for upper part = lower + 1
    int r_upper = *it_low + 1;
    int dest_upper = row_offset[r_upper];

    col[dest_upper] = *it_up;
    val[dest_upper] = *it_val;
    ++row_offset[r_upper];

    it_low++;
    it_up ++;
    it_val++;
  }
}

void export_openfoam_vector(const double* val, int n, Foam::scalarField *foam_vec) {

  assert(n == int(foam_vec->size()));

  Foam::UList<double>::iterator it = foam_vec->begin();

  int size = n;

  for (int i = 0; i < size; ++i, ++it)
    *it = val[i];

  free_host(&val);
}

void import_openfoam_vector(const Foam::scalarField &foam_vec,
                            double *val = NULL) {

  Foam::UList<double>::const_iterator it     = foam_vec.begin();
  Foam::UList<double>::const_iterator it_end = foam_vec.end();

  int size = int(foam_vec.size());

  allocate_host(size, &val);

  for (int i=0; it != it_end; ++it, ++i)
    val[i] = *it;
}

void import_openfoam_restriction(const Foam::labelField &foam_vec,
                                 double *val = NULL) {

  Foam::UList<int>::const_iterator it     = foam_vec.begin();
  Foam::UList<int>::const_iterator it_end = foam_vec.end();

  int size = int(foam_vec.size());
  int i = 0;

  allocate_host(size, &val);

  for (; it != it_end; ++it) {
    val[i] = *it;
    ++i;
  }
}

} // OFAMGX

