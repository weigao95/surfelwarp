//
// Created by wei on 3/5/18.
//

#pragma once

#include "hashing/hash_common.h"
#include "hashing/hash_config.h"


namespace hashing {


	class PrimeTableSizer
	{
	public:
		static unsigned GetPrimeTableSize(unsigned required_size);
	};


}
