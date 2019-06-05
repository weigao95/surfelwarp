#include "pcg_solver/BlockPCG.h"

void surfelwarp::hostBlockPCGZeroInit(
        const std::vector<float> &b,
        const std::vector<float> &inv_diags,
        std::vector<float> &h_r,
        std::vector<float> &h_s,
        float& dot_r_d
) {
    //Allocate host data
    h_r.resize(b.size());
    h_s.resize(b.size());

    //r <- b
    assert(h_r.size() == b.size());
    for(auto i = 0; i < b.size(); i++) {
        h_r[i] = b[i];
    }

    //Check s = inv_diag * b
    for(auto row = 0; row < b.size(); row++) {
        int blk_idx = row / 6;
        int inblk_offset = row % 6;
        int diag_offset = 36 * blk_idx;
        int diag_start_idx = diag_offset + 6 * inblk_offset;
        float s_row = 0.0f;
        for(auto j = 0; j < 6; j++) {
            s_row += inv_diags[diag_start_idx + j] * b[6 * blk_idx + j];
        }
        h_s[row] = s_row;
    }

    //Compute the dot product
    float dot_rs = 0.0f;
    for(auto i = 0; i < b.size(); i++) {
        dot_rs += h_r[i] * h_s[i];
    }
    dot_r_d = dot_rs;
}