//
// Created by wei on 2/13/18.
//
#include "common/Stream.h"
#include "common/Serializer.h"
#include "common/BinaryFileStream.h"
#include "pcg_solver/solver_configs.h"
#include "pcg_solver/block6x6_pcg_weber.h"
#include "pcg_solver/BinBlockCSR.h"
#include "pcg_solver/BlockPCG6x6.h"
#include "common/sanity_check.h"

#include <boost/filesystem.hpp>

//The check data path for pcg solver
#if defined(_WIN32)
const std::string pcg_data_path = "C:/Users/wei/Documents/Visual Studio 2015/Projects/surfelwarp/data/pcg_test/";
#else
const std::string pcg_data_path = "/home/wei/Documents/programs/surfelwarp/data/pcg_test";
#endif

void surfelwarp::loadCheckData(
        std::vector<float> &A_data,
        std::vector<int> &A_rowptr,
        std::vector<int> &A_colptr,
        std::vector<float> &b,
        std::vector<float> &diag_blks
) {
	std::string pcg_test_file = "pcg_test.dat";

    //Build the path
    using path = boost::filesystem::path;
    path data_dir(pcg_data_path);
	path pcg_test_path = data_dir / pcg_test_file;
	BinaryFileStream input_fstream(pcg_test_path.string().c_str(), BinaryFileStream::Mode::ReadOnly);
	input_fstream.SerializeRead(&A_data);
	input_fstream.SerializeRead(&A_rowptr);
	input_fstream.SerializeRead(&A_colptr);
	input_fstream.SerializeRead(&b);
	input_fstream.SerializeRead(&diag_blks);
}


void surfelwarp::loadCheckData(
        std::vector<float> &x,
        std::vector<float> &inv_diag_blks
) {
    //The name of loaded values
    std::string result_name = "pcg_result.dat";

    //Build the path
    using path = boost::filesystem::path;
    path data_dir(pcg_data_path);

    //Load them
    path pcg_result_path = data_dir / result_name;
	BinaryFileStream input_fstream(pcg_result_path.string().c_str(), BinaryFileStream::Mode::ReadOnly);
	input_fstream.SerializeRead(&x);
	input_fstream.SerializeRead(&inv_diag_blks);
}


void surfelwarp::check6x6DiagBlocksInverse(
        const std::vector<float> &diag_blks,
        const std::vector<float> &inv_diag_blks
) {
    //Upload the input to device
    DeviceArray<float> d_diag_blks, d_diag_inversed;
    d_diag_blks.upload(diag_blks);
    d_diag_inversed.create(d_diag_blks.size());
    const auto num_matrix = diag_blks.size() / 36;

    //Call the kernel
    block6x6_diag_inverse(d_diag_blks, d_diag_inversed, num_matrix);

    //Download and check
    std::vector<float> h_diag_inversed;
    d_diag_inversed.download(h_diag_inversed);

    //The checking code
    float max_relative_err = 0.0f;
    double avg_matrix_value = 0.0;
    for (auto i = 0; i < h_diag_inversed.size(); i++) {
        float h_diag_value = h_diag_inversed[i];
        float check_diag_value = inv_diag_blks[i];
        avg_matrix_value += h_diag_value;
        if(std::abs(check_diag_value) > 2e-3) {
            if (std::abs((check_diag_value - h_diag_value) / h_diag_value) > max_relative_err) {
                max_relative_err = std::abs((check_diag_value - h_diag_value) / h_diag_value);
	            LOG(INFO) << "The host value and checked value are " << h_diag_value << " and " << check_diag_value;
            }
        }
    }
    std::cout << "The maximun relative error is " << max_relative_err << std::endl;
    std::cout << "The average matrix entry value is " << (avg_matrix_value / h_diag_inversed.size()) << std::endl;
}


void surfelwarp::block6x6BuildTripletVector(
        const std::vector<float> &A_data,
        const std::vector<int> &A_rowptr,
        const std::vector<int> &A_colptr,
        const int matrix_size,
        std::vector<Eigen::Triplet<float>> &tripletVec
) {
	//Clear the output vector
    tripletVec.clear();

	//Loop over the bins
	int num_bins = divUp(matrix_size, bin_size);
	for (auto bin_idx = 0; bin_idx < num_bins; bin_idx++) {
		int first_row = bin_idx * bin_size;
		int bin_data_offset = A_rowptr[first_row];
		int bin_colptr_offset = bin_data_offset / 6;
		//Loop over the row in this bin
		for (auto j = 0; j < bin_size; j++) {
			int row_idx = first_row + j;
			int row_data_offset = bin_data_offset + j;
			int row_colptr_offset = bin_colptr_offset + j;
			int row_data_end = A_rowptr[row_idx + bin_size];
			while (row_data_offset < row_data_end) {
				for (auto k = 0; k < 6; k++) {
					float data = A_data[row_data_offset];
					int column_idx = A_colptr[row_colptr_offset] + k;
					if(column_idx >= 0 && std::abs(data) > 0.0f)
						tripletVec.push_back(Eigen::Triplet<float>(row_idx, column_idx, data));
					row_data_offset += bin_size;
				}
				row_colptr_offset += bin_size;
			}
		}
	}
}


void surfelwarp::hostEigenSpMV(
	const std::vector<float>& A_data, 
	const std::vector<int>& A_rowptr,
	const std::vector<int>& A_colptr, 
	const int matrix_size, 
	const std::vector<float>& x, 
	std::vector<float>& spmv
) {
	//Transfer to Eigen Vector
	Eigen::VectorXf eigen_x;
	eigen_x.resize(x.size());
	for(auto i = 0; i < x.size(); i++)
	{
		eigen_x(i) = x[i];
	}

	//Perform Sparse MV
	hostEigenSpMV(A_data, A_rowptr, A_colptr, matrix_size, eigen_x, spmv);
}


void surfelwarp::hostEigenSpMV(
	const std::vector<float>& A_data, 
	const std::vector<int>& A_rowptr, 
	const std::vector<int>& A_colptr, 
	const int matrix_size, 
	const Eigen::VectorXf & x, 
	std::vector<float>& spmv
) {
	//Build the triplet vector
	std::vector<Eigen::Triplet<float>> tripletVec;
	block6x6BuildTripletVector(A_data, A_rowptr, A_colptr, matrix_size, tripletVec);

	//Build the sparse matrix in Eigen
	Eigen::SparseMatrix<float> matrix;
	matrix.resize(matrix_size, matrix_size);
	matrix.setFromTriplets(tripletVec.begin(), tripletVec.end());

	//Do product and store the result
	Eigen::VectorXf product = matrix * x;
	spmv.resize(x.size());
	for (auto i = 0; i < x.size(); i++) {
		spmv[i] = product(i);
	}
}



void surfelwarp::checkBlock6x6PCGSolver(
		const std::vector<float> &diag_blks,
		const std::vector<float> &A_data,
		const std::vector<int> &A_colptr,
		const std::vector<int> &A_rowptr,
		const std::vector<float> &b,
		std::vector<float> &x
) {
	const auto matrix_size = b.size();
	//Prepare the data for device code
	DeviceArray<float> diag_blks_dev, A_data_dev, b_dev, x_dev;
	DeviceArray<int> A_colptr_dev, A_rowptr_dev;
	diag_blks_dev.upload(diag_blks);
	A_data_dev.upload(A_data);
	A_rowptr_dev.upload(A_rowptr);
	A_colptr_dev.upload(A_colptr);
	b_dev.upload(b);
	x_dev.create(matrix_size);

	//Prepare the aux storage
	DeviceArray<float> inv_diag_dev, p, q, r, s, t;
	inv_diag_dev.create(diag_blks_dev.size());
	p.create(matrix_size);
	q.create(matrix_size);
	r.create(matrix_size);
	s.create(matrix_size);
	t.create(matrix_size);

	//Invoke the solver
	DeviceArray<float> valid_x;
	block6x6_pcg_weber(
			diag_blks_dev,
			A_data_dev,
			A_colptr_dev,
			A_rowptr_dev,
			b_dev,
			x_dev,
			inv_diag_dev,
			p, q, r, s, t,
			valid_x
	);

	//Solve it with class version
#if defined(CHECK_CLASS_PCG6x6)
	BlockPCG6x6 solver;
	solver.AllocateBuffer(matrix_size);
	solver.SetSolverInput(diag_blks_dev, A_data_dev, A_colptr_dev, A_rowptr_dev, b_dev);
	//valid_x = solver.Solve();
	valid_x = solver.SolveTextured();
#endif

	//Check the solved x
	assert(valid_x.size() == x.size());
	valid_x.download(x);

	std::vector<float> spmv;
	spmv.resize(x.size());
	for(auto row = 0; row < x.size(); row++) {
		spmv[row] = BinBlockCSR<6>::SparseMV(A_data.data(), A_colptr.data(), A_rowptr.data(), x.data(), row);
	}
}