//
// Created by wei on 2/7/18.
//

#include <iostream>
#include <Eigen/Eigen>
#include "common/ConfigParser.h"
#include "common/sanity_check.h"

int main() {
    using namespace surfelwarp;
    checkPrefixSum();
    checkKeyValueSort();
    checkFlagSelection();
    checkUniqueSelection();
}