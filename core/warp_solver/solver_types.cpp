#include "common/sanity_check.h"
#include "core/warp_solver/solver_types.h"


//Compute mean, max/min, top K of the node error
void surfelwarp::NodeAlignmentError::errorStatistics(std::ostream& output) const {
	//First download the data
	std::vector<float> accumlate_error, accumlate_weight;
	node_accumlated_error.Download(accumlate_error);
	DeviceArrayView<float>(node_accumlate_weight, node_accumlated_error.Size()).Download(accumlate_weight);
	
	//Statistic about the raw error
	residualVectorStatistic(accumlate_error, 50, output);
	
	//Normalize the error using weight
	for(auto i = 0; i < accumlate_error.size(); i++) {
		if(accumlate_weight[i] > 1e-3f) {
			accumlate_error[i] /= accumlate_weight[i];
		}
	}
	
	//Statistic about the averaged error
	residualVectorStatistic(accumlate_error, 50, output);
}
