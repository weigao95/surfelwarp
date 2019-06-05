#include "pcg_solver/BinBlockCSR.h"
#include "pcg_solver/block6x6_pcg_weber.h"
#include "common/sanity_check.h"

#include <iostream>
#include <Eigen/Eigen>

void surfelwarp::checkBinBlock6x6SparseMV()
{
	//First load the data for 6x6 block
	std::vector<float> A_data, b, diag_blks;
	std::vector<int> A_colptr, A_rowptr;
	loadCheckData(A_data, A_rowptr, A_colptr, b, diag_blks);

	//Use random vector
	const auto matrix_size = b.size();
	VectorXf x; x.resize(matrix_size);
	x.setRandom();

	//Do a spase matrix vector product
	std::vector<float> spmv; 
	spmv.resize(matrix_size);
	for(auto i = 0; i < matrix_size; i++)
	{
		spmv[i] = BinBlockCSR<6>::SparseMV(A_data.data(), A_colptr.data(), A_rowptr.data(), x.data(), i);
	}

	//Check the result with Eigen version
	hostEigenSpMV(A_data, A_rowptr, A_colptr, matrix_size, x, b);

	//Check against b and spmv
	assert(b.size() == spmv.size());
	const auto relative_err = maxRelativeError(b, spmv);
	if(relative_err > 1e-4)
	{
		std::cout << "The relative error of sparse matrix vector product checking " << relative_err << std::endl;
	}
}
