#include "core/geometry/ReferenceNodeSkinner.h"


surfelwarp::ReferenceNodeSkinner::Ptr surfelwarp::ReferenceNodeSkinner::Instance() {
	static ReferenceNodeSkinner::Ptr instance = nullptr;
	if(instance == nullptr) {
		instance.reset(new ReferenceNodeSkinner());
	}
	return instance;
}

surfelwarp::ReferenceNodeSkinner::~ReferenceNodeSkinner() {
	m_invalid_nodes.release();
}