#pragma once
#include "common/common_types.h"
#include <string>

namespace surfelwarp {


	/**
	 * \brief The struct to maintained the 
	 *        host accessed constants.
	 */
	struct Constants
	{
		//The scale factor used in depth reprojection
		const static int kReprojectScaleFactor;

		//The sigma value used in bilateral filtering
		const static float kFilterSigma_S;
		const static float kFilterSigma_R;
		
		//The sigma value for foreground filtering
		const static float kForegroundSigma;
		
		//The size required by the renderer
		const static int kFusionMapScale;
		
		//The maximum number of surfels
		const static unsigned kMaxNumSurfels;
		
		//The maximum number of nodes and valid pairs
		const static unsigned kMaxNumNodes;
		const static unsigned kMaxNumNodePairs;
		const static unsigned kMaxNumSurfelCandidates;
		
		//The average and sampling distance of nodes
		const static float kNodeRadius; //[meter]
		const static float kNodeSamplingRadius; //[meter]
		
		//Select 1 nodes from kMaxSubsampleFrom candidates at most
		const static unsigned kMaxSubsampleFrom;

		//The number of node graph neigbours
		const static unsigned kNumNodeGraphNeigbours;
		
		//The recent time threshold for rendering solver maps
		const static int kRenderingRecentTimeThreshold;
		
		//The confidence threshold for stable surfel
		const static int kStableSurfelConfidenceThreshold;
		
		//The maximum number of sparse feature terms
		const static unsigned kMaxMatchedSparseFeature;
		
		//Use elastic penalty and the number of them
		const static bool kUseElasticPenalty;
		const static int kNumGlobalSolverItarations;
		const static int kNumGaussNewtonIterations;
		
		//The number of iterations used by segmenter
		const static int kMeanfieldSegmentIteration;
	};
}