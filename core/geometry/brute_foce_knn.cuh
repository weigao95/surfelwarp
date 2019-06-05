#pragma once
#include "common/common_utils.h"
#include <vector_types.h>

namespace surfelwarp { namespace device {
	
	struct KnnHeapDevice {
		float4& distance;
		ushort4& index;
		
		//The constructor just copy the pointer, the class will modifiy it
		__host__ __device__ KnnHeapDevice(float4& dist, ushort4& node_idx) : distance(dist), index(node_idx) {}
		
		//The update function
		__host__ __device__ __forceinline__ 
		void update(unsigned short idx, float dist) {
			if (dist < distance.x) {
				distance.x = dist;
				index.x = idx;
				
				if (distance.y < distance.z) {
					if (distance.x < distance.z) {
						swap(distance.x, distance.z);
						swap(index.x, index.z);
					}
				}
				else {
					if (distance.x < distance.y) {
						swap(distance.x, distance.y);
						swap(index.x, index.y);
						if (distance.y < distance.w) {
							swap(distance.y, distance.w);
							swap(index.y, index.w);
						}
					}
				}
			}
		}
	};



	__device__ __forceinline__ void bruteForceSearch4Padded(
		const float4& vertex, const float4* nodes, unsigned node_num,
		float4& distance,
		ushort4& node_idx
	) {
		//Construct the heap
		KnnHeapDevice heap(distance, node_idx);

		//The brute force search
		const auto padded_node_num = ((node_num + 3) / 4) * 4;
		for (int k = 0; k < padded_node_num; k += 4) {
			//Compute the distance to each nodes
			const float4& n0 = nodes[k + 0];
			const float tmp0 = vertex.x - n0.x;
			const float tmp1 = vertex.y - n0.y;
			const float tmp2 = vertex.z - n0.z;

			const float4& n1 = nodes[k + 1];
			const float tmp6 = vertex.x - n1.x;
			const float tmp7 = vertex.y - n1.y;
			const float tmp8 = vertex.z - n1.z;

			const float4& n2 = nodes[k + 2];
			const float tmp12 = vertex.x - n2.x;
			const float tmp13 = vertex.y - n2.y;
			const float tmp14 = vertex.z - n2.z;

			const float4& n3 = nodes[k + 3];
			const float tmp18 = vertex.x - n3.x;
			const float tmp19 = vertex.y - n3.y;
			const float tmp20 = vertex.z - n3.z;

			const float tmp3 = __fmul_rn(tmp0, tmp0);
			const float tmp9 = __fmul_rn(tmp6, tmp6);
			const float tmp15 = __fmul_rn(tmp12, tmp12);
			const float tmp21 = __fmul_rn(tmp18, tmp18);

			const float tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
			const float tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
			const float tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
			const float tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

			const float dist_0 = __fmaf_rn(tmp2, tmp2, tmp4);
			const float dist_1 = __fmaf_rn(tmp8, tmp8, tmp10);
			const float dist_2 = __fmaf_rn(tmp14, tmp14, tmp16);
			const float dist_3 = __fmaf_rn(tmp20, tmp20, tmp22);
			//End of distance computation

			//Update of distance index
			heap.update(k + 0, dist_0);
			heap.update(k + 1, dist_1);
			heap.update(k + 2, dist_2);
			heap.update(k + 3, dist_3);
		}//End of iteration over all nodes
	}
	
	//This method is deprecated and should not be used in later code
	__device__ __forceinline__ void bruteForceSearch4Padded(
		const float4& vertex, const float4* nodes, unsigned node_num,
		float& d0, float& d1, float& d2, float& d3,
		unsigned short& i0, unsigned short& i1, unsigned short& i2, unsigned short& i3
	) {
		//The brute force search
		const auto padded_node_num = ((node_num + 3) / 4) * 4;
		for (int k = 0; k < padded_node_num; k += 4) {
			//Compute the distance to each nodes
			const float4& n0 = nodes[k + 0];
			const float tmp0 = vertex.x - n0.x;
			const float tmp1 = vertex.y - n0.y;
			const float tmp2 = vertex.z - n0.z;

			const float4& n1 = nodes[k + 1];
			const float tmp6 = vertex.x - n1.x;
			const float tmp7 = vertex.y - n1.y;
			const float tmp8 = vertex.z - n1.z;

			const float4& n2 = nodes[k + 2];
			const float tmp12 = vertex.x - n2.x;
			const float tmp13 = vertex.y - n2.y;
			const float tmp14 = vertex.z - n2.z;

			const float4& n3 = nodes[k + 3];
			const float tmp18 = vertex.x - n3.x;
			const float tmp19 = vertex.y - n3.y;
			const float tmp20 = vertex.z - n3.z;

			const float tmp3 = __fmul_rn(tmp0, tmp0);
			const float tmp9 = __fmul_rn(tmp6, tmp6);
			const float tmp15 = __fmul_rn(tmp12, tmp12);
			const float tmp21 = __fmul_rn(tmp18, tmp18);

			const float tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
			const float tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
			const float tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
			const float tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

			const float dist_0 = __fmaf_rn(tmp2, tmp2, tmp4);
			const float dist_1 = __fmaf_rn(tmp8, tmp8, tmp10);
			const float dist_2 = __fmaf_rn(tmp14, tmp14, tmp16);
			const float dist_3 = __fmaf_rn(tmp20, tmp20, tmp22);
			//End of distance computation

			//Update of distance index
			if (dist_0 < d0) {
				d0 = dist_0;
				i0 = k;

				if (d1 < d2) {
					if (d0 < d2) {
						swap(d0, d2);
						swap(i0, i2);
					}
				}
				else {
					if (d0 < d1) {
						swap(d0, d1);
						swap(i0, i1);
						if (d1 < d3) {
							swap(d1, d3);
							swap(i1, i3);
						}
					}
				}
			}

			if (dist_1 < d0) {
				d0 = dist_1;
				i0 = k + 1;

				if (d1 < d2) {
					if (d0 < d2) {
						swap(d0, d2);
						swap(i0, i2);
					}
				}
				else {
					if (d0 < d1) {
						swap(d0, d1);
						swap(i0, i1);
						if (d1 < d3) {
							swap(d1, d3);
							swap(i1, i3);
						}
					}
				}
			}

			if (dist_2 < d0) {
				d0 = dist_2;
				i0 = k + 2;

				if (d1 < d2) {
					if (d0 < d2) {
						swap(d0, d2);
						swap(i0, i2);
					}
				}
				else {
					if (d0 < d1) {
						swap(d0, d1);
						swap(i0, i1);
						if (d1 < d3) {
							swap(d1, d3);
							swap(i1, i3);
						}
					}
				}
			}

			if (dist_3 < d0) {
				d0 = dist_3;
				i0 = k + 3;

				if (d1 < d2) {
					if (d0 < d2) {
						swap(d0, d2);
						swap(i0, i2);
					}
				}
				else {
					if (d0 < d1) {
						swap(d0, d1);
						swap(i0, i1);
						if (d1 < d3) {
							swap(d1, d3);
							swap(i1, i3);
						}
					}
				}
			}
		}//End of iteration over all nodes
	}


} // namespace device
} // namespace surfelwarp