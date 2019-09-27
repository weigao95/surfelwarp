#include <iostream>
#include "common/ConfigParser.h"
#include "visualization/Visualizer.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "imgproc/ImageProcessor.h"
#include "imgproc/segmentation/ForegroundSegmenterNaiveCRF.h"

int main() {
	using namespace surfelwarp;
	std::cout << "Test for crf field" << std::endl;

	//Parpare the test data
	auto parser = ConfigParser::Instance();

	//First test fetching
	FileFetch::Ptr fetcher = std::make_shared<FileFetch>(parser.data_path());
	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(parser, fetcher);
	processor->FetchFrame(0);
	processor->UploadDepthImage();
	processor->ReprojectDepthToRGB();
	processor->ClipFilterDepthImage();

	//the implemented rbg methods
	processor->UploadRawRGBImage();
	processor->ClipNormalizeRGBImage();

	//Contruct the segmenter
	ForegroundSegmenterNaiveCRF::Ptr segmenter = std::make_shared<ForegroundSegmenterNaiveCRF>();
	const unsigned subsampled_rows = 220;
	const unsigned subsampled_cols = 300;
	segmenter->AllocateBuffer(subsampled_rows, subsampled_cols);

	//Set input
	segmenter->SetInputImages(processor->ClipNormalizedRGBTexture(), processor->RawDepthTexture(), processor->FilteredDepthTexture());
	segmenter->Segment();
	cudaTextureObject_t mask = segmenter->ForegroundMask();
	
	//Draw it
	//Visualizer::SaveNormalizeRGBImage(processor->ClipNormalizedRGBTexture(), "rgb.png");
	//Visualizer::SaveSegmentMask(mask, processor->ClipNormalizedRGBTexture(), "iter4_segment.png");
	Visualizer::DrawSegmentMask(mask, processor->ClipNormalizedRGBTexture());

	segmenter->ReleaseBuffer();
	
	
	std::cout << "End of the test" << std::endl;
}