//
// Created by wei on 2/20/18.
//

#include "common/data_transfer.h"
#include "visualization/Visualizer.h"
//#include <pcl/visualization/pcl_visualizer.h>


/* The depth image drawing methods
*/
void surfelwarp::Visualizer::DrawDepthImage(const cv::Mat & depth_img)
{
	double max_depth, min_depth;
	cv::minMaxIdx(depth_img, &min_depth, &max_depth);
	//Visualize depth-image in opencv
	cv::Mat depth_scale;
	cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
	cv::imshow("depth image", depth_scale);
	cv::waitKey(0);
}

void surfelwarp::Visualizer::SaveDepthImage(const cv::Mat &depth_img, const std::string &path) {
	double max_depth, min_depth;
	cv::minMaxIdx(depth_img, &min_depth, &max_depth);
	//Visualize depth-image in opencv
	cv::Mat depth_scale;
	cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
    cv::imwrite(path, depth_scale);
}

void surfelwarp::Visualizer::DrawDepthImage(const DeviceArray2D<unsigned short>& depth_img)
{
	const auto depth_cpu = downloadDepthImage(depth_img);
	DrawDepthImage(depth_cpu);
}

void surfelwarp::Visualizer::SaveDepthImage(const DeviceArray2D<unsigned short>& depth_img, const std::string& path)
{
	const auto depth_cpu = downloadDepthImage(depth_img);
	SaveDepthImage(depth_cpu, path);
}

void surfelwarp::Visualizer::DrawDepthImage(cudaTextureObject_t depth_img) {
	const auto depth_cpu = downloadDepthImage(depth_img);
	DrawDepthImage(depth_cpu);
}

void surfelwarp::Visualizer::SaveDepthImage(cudaTextureObject_t depth_img, const std::string & path) {
	const auto depth_cpu = downloadDepthImage(depth_img);
	SaveDepthImage(depth_cpu, path);
}


/* The color image drawing methods
*/
void surfelwarp::Visualizer::DrawRGBImage(const cv::Mat &rgb_img) {
	cv::imshow("color image", rgb_img);
	cv::waitKey(0);
}

void surfelwarp::Visualizer::SaveRGBImage(const cv::Mat &rgb_img, const std::string &path) {
    cv::imwrite(path, rgb_img);
}

void surfelwarp::Visualizer::DrawRGBImage(
	const DeviceArray<uchar3>& rgb_img,
	const int rows, const int cols
) {
	const auto rgb_cpu = downloadRGBImage(rgb_img, rows, cols);
	DrawRGBImage(rgb_cpu);
}


void surfelwarp::Visualizer::SaveRGBImage(
	const DeviceArray<uchar3>& rgb_img, 
	const int rows, const int cols, 
	const std::string & path
) {
	const auto rgb_cpu = downloadRGBImage(rgb_img, rows, cols);
	SaveRGBImage(rgb_cpu, path);
}



void surfelwarp::Visualizer::DrawNormalizeRGBImage(cudaTextureObject_t rgb_img)
{
	const auto rgb_cpu = downloadNormalizeRGBImage(rgb_img);
	DrawRGBImage(rgb_cpu);
}

void surfelwarp::Visualizer::SaveNormalizeRGBImage(cudaTextureObject_t rgb_img, const std::string & path)
{
	const auto rgb_cpu = downloadNormalizeRGBImage(rgb_img);
	cv::Mat rgb_cpu_8uc4;
	rgb_cpu.convertTo(rgb_cpu_8uc4, CV_8UC4, 255);
	SaveRGBImage(rgb_cpu_8uc4, path);
}


void surfelwarp::Visualizer::DrawColorTimeMap(cudaTextureObject_t color_time_map)
{
	const auto rgb_cpu = rgbImageFromColorTimeMap(color_time_map);
	DrawRGBImage(rgb_cpu);
}

void surfelwarp::Visualizer::DrawNormalMap(cudaTextureObject_t normal_map) {
	const auto rgb_cpu = normalMapForVisualize(normal_map);
	DrawRGBImage(rgb_cpu);
}

/* The gray scale image drawing methods
 */
void surfelwarp::Visualizer::DrawGrayScaleImage(const cv::Mat &gray_scale_img) {
	cv::imshow("gray scale image", gray_scale_img);
	cv::waitKey(0);
}

void surfelwarp::Visualizer::SaveGrayScaleImage(const cv::Mat &gray_scale_img, const std::string &path) {
	cv::imwrite(path, gray_scale_img);
}

void surfelwarp::Visualizer::DrawGrayScaleImage(cudaTextureObject_t gray_scale_img, float scale) {
	cv::Mat h_image;
	downloadGrayScaleImage(gray_scale_img, h_image, scale);
	DrawGrayScaleImage(h_image);
}

void surfelwarp::Visualizer::SaveGrayScaleImage(cudaTextureObject_t gray_scale_img, const std::string &path, float scale) {
	cv::Mat h_image;
	downloadGrayScaleImage(gray_scale_img, h_image, scale);
	SaveGrayScaleImage(h_image, path);
}

/* The segmentation mask drawing methods
*/
void surfelwarp::Visualizer::MarkSegmentationMask(
	const std::vector<unsigned char>& mask,
	cv::Mat & rgb_img,
	const unsigned sample_rate
) {
	const auto rgb_rows = rgb_img.rows;
	const auto rgb_cols = rgb_img.cols;
	const auto mask_cols = rgb_cols / sample_rate;
	for(auto row = 0; row < rgb_rows; row++) {
		for(auto col = 0; col < rgb_cols; col++) {
			const auto mask_r = row / sample_rate;
			const auto mask_c = col / sample_rate;
			const auto flatten_idx = mask_c + mask_r * mask_cols;
			const unsigned char mask_value = mask[flatten_idx];
			if(mask_value > 0) {
				rgb_img.at<unsigned char>(row, 4 * col + 0) = 255;
				rgb_img.at<unsigned char>(row, 4 * col + 1) = 255;
				//rgb_img.at<unsigned char>(row, 4 * col + 2) = 255;
			}
		}
	}
}

void surfelwarp::Visualizer::DrawSegmentMask(
	const std::vector<unsigned char>& mask,
	cv::Mat & rgb_img,
	const unsigned sample_rate
) {
	MarkSegmentationMask(mask, rgb_img, sample_rate);
	DrawRGBImage(rgb_img);
}

void surfelwarp::Visualizer::SaveSegmentMask(
	const std::vector<unsigned char>& mask, 
	cv::Mat & rgb_img,
	const std::string & path, 
	const unsigned sample_rate
) {
	MarkSegmentationMask(mask, rgb_img, sample_rate);
	SaveRGBImage(rgb_img, path);
}

void surfelwarp::Visualizer::DrawSegmentMask(
	cudaTextureObject_t mask, 
	cudaTextureObject_t normalized_rgb_img, 
	const unsigned sample_rate
) {
	//Download the rgb image
	const auto rgb_cpu = downloadNormalizeRGBImage(normalized_rgb_img);
	cv::Mat rgb_cpu_8uc4;
	rgb_cpu.convertTo(rgb_cpu_8uc4, CV_8UC4, 255);

	//Download the segmentation mask
	std::vector<unsigned char> h_mask;
	downloadSegmentationMask(mask, h_mask);

	//Call the drawing functions
	DrawSegmentMask(h_mask, rgb_cpu_8uc4, sample_rate);
}

void surfelwarp::Visualizer::SaveSegmentMask(
	cudaTextureObject_t mask, 
	cudaTextureObject_t normalized_rgb_img, 
	const std::string & path, 
	const unsigned sample_rate
) {
	//Download the rgb image
	const auto rgb_cpu = downloadNormalizeRGBImage(normalized_rgb_img);
	cv::Mat rgb_cpu_8uc4;
	rgb_cpu.convertTo(rgb_cpu_8uc4, CV_8UC4, 255);

	//Download the segmentation mask
	std::vector<unsigned char> h_mask;
	downloadSegmentationMask(mask, h_mask);

	//Call the saving methods
	SaveSegmentMask(h_mask, rgb_cpu_8uc4, path, sample_rate);
}

void surfelwarp::Visualizer::SaveRawSegmentMask(cudaTextureObject_t mask, const std::string &path) {
	//Download the segmentation mask
	cv::Mat raw_mask = downloadRawSegmentationMask(mask);
	
	//Save it to image
	cv::imwrite(path, raw_mask);
}

void surfelwarp::Visualizer::DrawRawSegmentMask(cudaTextureObject_t mask) {
	//Download the segmentation mask
	cv::Mat raw_mask = downloadRawSegmentationMask(mask);
	
	cv::Mat converted_mask;
	raw_mask.convertTo(converted_mask, CV_8UC1, 255);
	DrawRGBImage(converted_mask);
}

void surfelwarp::Visualizer::DrawBinaryMeanfield(cudaTextureObject_t meanfield_q)
{
	cv::Mat h_meanfield_uchar;
	downloadTransferBinaryMeanfield(meanfield_q, h_meanfield_uchar);
	DrawRGBImage(h_meanfield_uchar);
}

void surfelwarp::Visualizer::SaveBinaryMeanfield(cudaTextureObject_t meanfield_q, const std::string & path)
{
	cv::Mat h_meanfield_uchar;
	downloadTransferBinaryMeanfield(meanfield_q, h_meanfield_uchar);
	SaveRGBImage(h_meanfield_uchar, path);
}

/* The image pair correspondence method
 */
void surfelwarp::Visualizer::DrawImagePairCorrespondence(
	cudaTextureObject_t rgb_0,
	cudaTextureObject_t rgb_1,
	const surfelwarp::DeviceArray<ushort4> &corr_d
) {
	//Download the data
	std::vector<ushort4> corr;
	corr_d.download(corr);
	cv::Mat normalized_from = downloadNormalizeRGBImage(rgb_0);
	cv::Mat normalized_to = downloadNormalizeRGBImage(rgb_1);
	cv::Mat from, to;
	normalized_from.convertTo(from, CV_8UC4, 255);
	normalized_to.convertTo(to, CV_8UC4, 255);
	
	//cv programs
	int ind = 0;
	
	cv::Mat show0 = cv::Mat::zeros(from.rows, to.cols * 2, CV_8UC4);
	cv::Rect rect(0, 0, from.cols, from.rows);
	from.copyTo(show0(rect));
	rect.x = rect.x + from.cols;
	to.copyTo(show0(rect));
	
	cv::RNG rng(12345);
	srand(time(NULL));
	int step = corr.size() / 10 * 2;
	int goout = 1;
	cv::Mat show;
	
	cv::namedWindow("Correspondences", cv::WINDOW_AUTOSIZE);
	std::cout << "Only about 100 points are shown, press N to show others, press other keys to exit!" << std::endl;
	while (goout == 1) {
		show = show0.clone();
		while (ind < corr.size()) {
			cv::Point2f a; // = corr[ind].first;
			a.x = corr[ind].x;
			a.y = corr[ind].y;
			cv::Point2f b; // = corr[ind].second;
			b.x = corr[ind].z;
			b.y = corr[ind].w;
			b.x += from.cols;
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::line(show, a, b, color, 1);
			ind = ind + (int)rand() % step;
		}
		imshow("Correspondences", show);
		int c = cv::waitKey();
		switch (c) {
			case 'n':
			case 'N': goout = 1; break;
			default: goout = 0;
		}
		ind = 0;
	}
}







