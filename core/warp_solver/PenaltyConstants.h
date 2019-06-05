//
// Created by wei on 4/26/18.
//

#pragma once

#include <vector_types.h>


namespace surfelwarp {
	
	//Forward declare
	class SolverIterationData;
	
	class PenaltyConstants {
	private:
		float m_lambda_smooth;
		float m_lambda_density;
		float m_lambda_foreground;
		float m_lambda_feature;
		
		//Only modifiable by warp solver
		friend class SolverIterationData;
		void setDefaultValue();
		void setGlobalIterationValue(bool use_foreground = false);
		void setLocalIterationValue(bool use_density = false);

	public:
		__host__ __device__ PenaltyConstants();
		
		//All access other than WarpSolver should be read-only
		__host__ __device__ __forceinline__ float DenseDepth() const { return 1.0f; }
		__host__ __device__ __forceinline__ float DenseDepthSquared() const { return 1.0f; }
		
		__host__ __device__ __forceinline__ float Smooth() const { return m_lambda_smooth; }
		__host__ __device__ __forceinline__ float SmoothSquared() const { return m_lambda_smooth * m_lambda_smooth; }
		
		__host__ __device__ __forceinline__ float Density() const { return m_lambda_density; }
		__host__ __device__ __forceinline__ float DensitySquared() const { return m_lambda_density * m_lambda_density; }
		
		__host__ __device__ __forceinline__ float Foreground() const { return m_lambda_foreground; }
		__host__ __device__ __forceinline__ float ForegroundSquared() const { return m_lambda_foreground * m_lambda_foreground; }
		
		__host__ __device__ __forceinline__ float SparseFeature() const { return m_lambda_feature; }
		__host__ __device__ __forceinline__ float SparseFeatureSquared() const { return m_lambda_feature * m_lambda_feature; }
	};
	
}
