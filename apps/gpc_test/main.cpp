//
// Created by wei on 3/7/18.
//

#include <iostream>
#include "common/ConfigParser.h"
#include "common/data_transfer.h"
#include "visualization/Visualizer.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "imgproc/ImageProcessor.h"
#include "imgproc/correspondence/ImagePairCorrespondence.h"
#include "imgproc/correspondence/PatchColliderRGBCorrespondence.h"

int main() {
	using namespace surfelwarp;
	std::cout << "Test for global patch collider" << std::endl;
	
	//Parpare the test data
	auto parser = ConfigParser::Instance();
	
	//First test fetching
	FileFetch::Ptr fetcher = std::make_shared<FileFetch>(parser.data_path());
	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(parser, fetcher);
	processor->FetchFrame(1);
	processor->UploadDepthImage();
	processor->ReprojectDepthToRGB();
	processor->ClipFilterDepthImage();
	
	//the implemented rbg methods
	processor->UploadRawRGBImage();
	processor->ClipNormalizeRGBImage();

	//the geometry methods
	processor->BuildVertexConfigMap();
	processor->BuildNormalRadiusMap();

	//the color time map
	processor->BuildColorTimeTexture(0);

	//the surfel array
	processor->CollectValidDepthSurfel();
	
	//segment it
	processor->SegmentForeground();
	
	//Contruct the segmenter
	cudaTextureObject_t rgb_0 = processor->ClipNormalizedRGBTexturePrev();
	cudaTextureObject_t rgb_1 = processor->ClipNormalizedRGBTexture();
	cudaTextureObject_t foreground_1 = processor->ForegroundMask();

	//Contruct the class
	ImagePairCorrespondence::Ptr correspondence = std::make_shared<PatchColliderRGBCorrespondence>();

	//The interface
	const auto rows = processor->clip_rows();
	const auto cols = processor->clip_cols();
	correspondence->AllocateBuffer(rows, cols);
	
	std::cout << "Finish of model loading and allocation" << std::endl;

	//Set the input image
	correspondence->SetInputImages(rgb_0, rgb_1, foreground_1);

	//Invoke the method
	correspondence->FindCorrespondence();
	
	//Take a look
	//Visualizer::DrawNormalizeRGBImage(rgb_1);
	const auto corr_d = correspondence->CorrespondedPixelPairs();
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
			cv::line(show, a, b, color, 1, cv::LINE_8, 0);
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
	
	
	correspondence->ReleaseBuffer();
	std::cout << "End of the test" << std::endl;
}