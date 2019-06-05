#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/algorithm_types.h"
#include "imgproc/segmentation/crf_config.h"
#include "imgproc/segmentation/ForegroundSegmenter.h"
#include "imgproc/segmentation/permutohedral_common.h"
#include "imgproc/segmentation/foreground_permutohedral_deduplicate.h"

//The hash set for lattice coordinate
#include "hashing/TicketBoardSet.h"

#include <memory>

namespace surfelwarp {
	
	class ForegroundSegmenterPermutohedral : public ForegroundSegmenter {
	private:
		//The buffer for mean field probability, float1 texture
		CudaTextureSurface m_meanfield_foreground_collect_subsampled[2];
		int m_updated_meanfield_idx;

		//The output as a binary mask
		CudaTextureSurface m_segmented_mask_collect_subsampled;
		
		//The buffer for unary energy
		DeviceArray2D<float2> m_unary_energy_map_subsampled;

		//Small struct to collect the input
		struct InputTexture {
			cudaTextureObject_t clip_normalize_rgb_img;
			cudaTextureObject_t raw_depth_img;
			cudaTextureObject_t clip_depth_img;
		} m_input_texture;
		
		//Constants for apperance
		static constexpr float sigma_alpha_ = 10.0f;
		static constexpr float sigma_beta_  = 10.0f;

		//This value is almost emperical
		const unsigned kMaxUniqueLattices = 15000;

	public:
		//Update the pointer
		using Ptr = std::shared_ptr<ForegroundSegmenterPermutohedral>;

		//Defualt contstruct, no copy/assign/move
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(ForegroundSegmenterPermutohedral);
		SURFELWARP_NO_COPY_ASSIGN_MOVE(ForegroundSegmenterPermutohedral);

		//The caller ensure 1 allocate and 1 release
		void AllocateBuffer(
			unsigned clip_rows,
			unsigned clip_cols
		) override;
		void ReleaseBuffer() override;

		//The input output interface
		void SetInputImages(
			cudaTextureObject_t clip_normalized_rgb_img, 
			cudaTextureObject_t raw_depth_img, 
			cudaTextureObject_t clip_depth_img,
			int frame_idx,
			cudaTextureObject_t clip_background_rgb = 0
		) override;
		void Segment(cudaStream_t stream = 0) override;
		cudaTextureObject_t ForegroundMask() const override;
		cudaTextureObject_t FilterForegroundMask() const override;
		cudaTextureObject_t SubsampledForegroundMask() const;
		
		
		/**
		* \brief Methods that shared lots of similiarity
		*        with naive approach
		*/
	public:
		void initMeanfieldUnaryEnergy(cudaStream_t stream = 0);


		/**
		* \brief Methods to build the lattice coordinate index
		*/
	private:
		hashing::TicketBoardSet<LatticeCoordKey<5>> m_lattice_set;
		void allocateLatticeIndexBuffer();
		void releaseLatticeIndexBuffer();
	public:
		void buildLatticeIndex(cudaStream_t stream = 0);



		/**
		* \brief Methods to compute the value at lattice. For each
		*  lattice there is a vector, where vec.x and vec.y is the ENERGY
		*  for foreground and background. The actual size
		*  of this array should be the number of valid lattice.
		*/
	private:
		DeviceArray<float2> m_lattice_energy_array;

		//The size of vector are m_max_unique_lattices
		void allocateLatticeValueBuffer();
		void releaseLatticeValueBuffer();
	public:
		void splatEnergy(cudaStream_t stream = 0);
		void slice(cudaStream_t stream = 0);
		void writeSegmentationMask(cudaStream_t stream = 0);
		
		
		//Perform upsampling and filtering of the mask
	private:
		CudaTextureSurface m_foreground_mask_collect_upsampled;
		CudaTextureSurface m_filter_foreground_mask_collect_upsampled;
	public:
		void upsampleFilterForegroundMask(cudaStream_t stream = 0);
		
		
		//Debug method
		void saveMeanfieldApproximationMap(const unsigned iter = 0);
	};


}