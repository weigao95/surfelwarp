#include "hashing/PrimeTableSizer.h"
#include <algorithm>


unsigned hashing::PrimeTableSizer::GetPrimeTableSize(unsigned required_size)
{
	static unsigned table_size = 17;
	static unsigned prime_table[] = {
		3001, 5003, 7001, 
		10007, 30011, 60013,
		100003, 200003, 300023,
		400031, 500009, 600011, 
		700001, 800011, 900007,
		1000003, 1500007 //The million scale
	};
	return *std::lower_bound(prime_table, prime_table + table_size, required_size);
}