#include "common/Constants.h"

const int surfelwarp::Constants::kReprojectScaleFactor = 2;

const float surfelwarp::Constants::kFilterSigma_S = 4.5f;
const float surfelwarp::Constants::kFilterSigma_R = 30.0f;

//The sigma value for foreground filtering
const float surfelwarp::Constants::kForegroundSigma = 5.0f;

//The maximum number of sparse feature terms
const unsigned surfelwarp::Constants::kMaxMatchedSparseFeature = 20000;

//The scale of fusion map
const int surfelwarp::Constants::kFusionMapScale = d_fusion_map_scale;

//The maximum number of surfels
const unsigned surfelwarp::Constants::kMaxNumSurfels = 500000;

//The maximum number of nodes
const unsigned surfelwarp::Constants::kMaxNumNodes = d_max_num_nodes;
const unsigned surfelwarp::Constants::kMaxNumNodePairs = 60 * surfelwarp::Constants::kMaxNumNodes;
const unsigned surfelwarp::Constants::kMaxNumSurfelCandidates = 50000;

//The average and sampling distance between nodes
const float surfelwarp::Constants::kNodeRadius = d_node_radius;
const float surfelwarp::Constants::kNodeSamplingRadius = 0.85f * surfelwarp::Constants::kNodeRadius;

//Select 1 nodes from kMaxSubsampleFrom candidates at most
const unsigned surfelwarp::Constants::kMaxSubsampleFrom = 5;

//The number of node graph neigbours
const unsigned surfelwarp::Constants::kNumNodeGraphNeigbours = 8;

//The recent time threshold for rendering solver maps
const int surfelwarp::Constants::kRenderingRecentTimeThreshold = 3;

//The confidence threshold for stable surfel
const int surfelwarp::Constants::kStableSurfelConfidenceThreshold = 10;

//Use elastic penalty or not
const bool surfelwarp::Constants::kUseElasticPenalty = true;
const int surfelwarp::Constants::kNumGlobalSolverItarations = 3;
const int surfelwarp::Constants::kNumGaussNewtonIterations = 6;

//The iteration used by segmenter
const int surfelwarp::Constants::kMeanfieldSegmentIteration = 3;