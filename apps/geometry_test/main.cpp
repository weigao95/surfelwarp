//
// Created by wei on 5/19/18.
//

#include "core/SurfelWarpSerial.h"

int main() {
	using namespace surfelwarp;
	SurfelWarpSerial::Ptr fusion = std::make_shared<SurfelWarpSerial>();
	fusion->ProcessFirstFrame();
	fusion->TestGeometryProcessing();
}