#include "math/ldlt6x6.h"

void surfelwarp::ldlt6x6::factor(const float *matrix, float *factored) {
    //The first itertion
    float D0 = factored[flatten(0, 0)] = matrix[flatten(0, 0)];
    for (auto i = 1; i < 6; i++) {
        factored[flatten(0, i)] = matrix[flatten(0, i)] / D0;
    }

    //Other iterations
    for (auto j = 1; j < 6; j++) {
        //First compute the diag factored value
        float Dj = matrix[flatten(j, j)];
        for(auto k = 0; k < j; k++) {
            Dj -= factored[flatten(j, k)] * factored[flatten(j, k)] * factored[flatten(k, k)];
        }
        factored[flatten(j, j)] = Dj;

        //Next compute other values
        for(auto i = j + 1; i < 6; i++) {
            float Lij = matrix[flatten(i, j)];
            for(auto k = 0; k < j; k++) {
                Lij -= factored[flatten(j, k)] * factored[flatten(i, k)] * factored[flatten(k, k)];
            }
            factored[flatten(i, j)] = Lij / Dj;
        }
    }
}