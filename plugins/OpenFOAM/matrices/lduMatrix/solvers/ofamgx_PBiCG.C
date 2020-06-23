/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "ofamgx/ofamgx_openfoam.H"
#include "ofamgx_PBiCG.H"

#include <AmgXSolver.hpp>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(ofamgx_PBiCG, 0);

    lduMatrix::solver::addasymMatrixConstructorToTable<ofamgx_PBiCG>
        addofamgx_PBiCGAsymMatrixConstructorToTable_;

    lduMatrix::solver::addsymMatrixConstructorToTable<ofamgx_PBiCG>
        addofamgx_PBiCGSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ofamgx_PBiCG::ofamgx_PBiCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::ofamgx_PBiCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    word precond_name = lduMatrix::preconditioner::getName(controlDict_);
    double div   = controlDict_.lookupOrDefault<double>("div", 1e+08);
    int ILUp     = controlDict_.lookupOrDefault<int>("ILUp", 0);
    int ILUq     = controlDict_.lookupOrDefault<int>("ILUq", 1);
    int MEp      = controlDict_.lookupOrDefault<int>("MEp", 1);
    word LBPre   = controlDict_.lookupOrDefault<word>("LastBlockPrecond", "ofamgx_Jacobi");
    
    solverPerformance solverPerf(typeName + '(' + precond_name + ')', fieldName_);

    register label nCells = psi.size();

    scalarField pA(nCells);
    scalarField wA(nCells);

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    scalarField rA(source - wA);

    // --- Calculate normalisation factor
    scalar normFactor = this->normFactor(psi, source, wA, pA);

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() = gSumMag(rA)/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();


    Pout << "I am here --- 1" << endl;
    if (!solverPerf.checkConvergence(tolerance_, relTol_)) {
        int *row, *col;
        double *value;
        
        int size;
        
        OFAmgX::import_openfoam_matrix(matrix(), row, col, value, &size);
        

        OFAmgX::AmgXSolver solver;
        solver.initialize("dDDI", "");
        solver.setA(row, col, value, size);

        double *rhs, *unks;
        OFAmgX::import_openfoam_vector(source, rhs);
        OFAmgX::allocate_host(size, &unks);
        OFAmgX::set_to_zero_host(size, unks);
        
        solver.solve(rhs, unks, size);

        OFAmgX::export_openfoam_vector(unks, size, &psi);
        
        OFAmgX::free_host(&rhs);
        OFAmgX::free_host(&row);
        OFAmgX::free_host(&value);
        OFAmgX::free_host(&col);

        solverPerf.finalResidual()   = solver.getResidual(solver.getIters()) / normFactor; // divide by normFactor, see lduMatrixSolver.C
        solverPerf.nIterations()     = solver.getIters();
        solverPerf.checkConvergence(tolerance_, relTol_);
    }

    return solverPerf;
}


// ************************************************************************* //
