#include "common/sanity_check.h"
#include "pcg_solver/ApplySpMVBinBlockCSR.h"
#include "core/warp_solver/JtJMaterializer.h"

//An integrated test on the correctness of spmv
void surfelwarp::JtJMaterializer::TestSparseMV(
	DeviceArrayView<float> x,
	DeviceArrayView<float> jtj_x_result
) {
	LOG(INFO) << "Check materialized spmv given input groundtruth";
	//Construct the matrix applier
	ApplySpMVBinBlockCSR<6> spmv_handler;
	spmv_handler.SetInputs(
		m_binblock_csr_data.Ptr(),
		m_nodepair2term_map.binblock_csr_rowptr.RawPtr(),
		m_nodepair2term_map.binblock_csr_colptr,
		x.Size()
	);
	
	//Apply it
	DeviceArray<float> jtj_x; jtj_x.create(x.Size());
	spmv_handler.ApplySpMV(x, DeviceArraySlice<float>(jtj_x));
	
	//Check the error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
#endif
    //TODO: instead of leaving dead code, either change this function w/ optional argument/flag check or
    // use a separate set of functions / child class w/ overriding member functions for performance measurement
    //TODO: get rid of PCL usage in favor of std::chrono or the like
    /*{
        pcl::ScopeTime time("The performance of materialized spmv");
        for(auto i = 0; i < 10000; i++) {
            spmv_handler.ApplySpMV(x, DeviceArraySlice<float>(jtj_x));
        }
        cudaSafeCall(cudaDeviceSynchronize());
    }*/
	
	//Compare with ground truth
	std::vector<float> spmv_materialized, spmv_result;
	jtj_x.download(spmv_materialized);
	jtj_x_result.Download(spmv_result);
	
	//Compare it
	const auto relative_err = maxRelativeError(spmv_result, spmv_materialized);
	
	LOG(INFO) << "Check done, the relative error between materalized method and matrix free method is " << relative_err;
}