#pragma once

template<int N>
void surfelwarp::DenseLDLT<N>::Factor(float *matrix, float *factored) {
    //The first itertion
    float D0 = factored[toprightFlatten(0, 0)] = matrix[bottomleftFlatten(0, 0)];
    for (auto i = 1; i < N; i++) {
        factored[toprightFlatten(0, i)] = matrix[bottomleftFlatten(0, i)] / D0;
    }

    //Other iterations
    for (auto j = 1; j < N; j++) {
        //First compute the diag factored value
        float Dj = matrix[bottomleftFlatten(j, j)];
        for(auto k = 0; k < j; k++) {
            Dj -= factored[toprightFlatten(j, k)] * factored[toprightFlatten(j, k)] * factored[toprightFlatten(k, k)];
        }
        factored[toprightFlatten(j, j)] = Dj;

        //Next compute other values
        for(auto i = j + 1; i < N; i++) {
            float Lij = matrix[bottomleftFlatten(i, j)];
            for(auto k = 0; k < j; k++) {
                Lij -= factored[toprightFlatten(j, k)] * factored[toprightFlatten(i, k)] * factored[toprightFlatten(k, k)];
            }
            factored[toprightFlatten(i, j)] = Lij / Dj;
        }
    }
}

template<int N>
void surfelwarp::DenseLDLT<N>::Factor(float *matrix) {
    DenseLDLT<N>::Factor(matrix, matrix);
}


template<int N>
void surfelwarp::DenseLDLT<N>::Solve(const float * factored, float * b, float * auxiliary)
{
	//First solve L x_0 = b
	float* x_0 = auxiliary;
	x_0[0] = b[0];
	for (auto i = 1; i < N; i++) {
		float x_0i = b[i];
		for (auto j = 0; j < i; j++) {
			x_0i -= x_0[j] * factored[toprightFlatten(j, i)];
		}
		x_0[i] = x_0i;
	}

	//Next solve diag(D) x_1 = x_0
	for (auto i = 0; i < N; i++) {
		x_0[i] = x_0[i] / (factored[toprightFlatten(i, i)]);
	}

	//Finally solve L^{T} x_2 = x_1, x_2 should be stored to b
	float* x_2 = b;
	float* x_1 = x_0;
	x_2[N - 1] = x_1[N - 1];
	for (auto i = N - 2; i >= 0; i--) {
		float x_2i = x_1[i];
		float minus = 0.0f;
		for (auto j = N - 1; j > i; j--) {
			minus += x_2[j] * factored[toprightFlatten(i, j)];
		}
		x_2[i] = x_2i - minus;
	}
}

template<int N>
void surfelwarp::DenseLDLT<N>::Solve(const float * factored, float * b)
{
	float auxiliary[N];
	Solve(factored, b, auxiliary);
}