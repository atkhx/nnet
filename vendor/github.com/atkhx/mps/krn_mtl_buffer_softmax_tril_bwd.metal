#include <metal_stdlib>

using namespace metal;

struct Parameters {
    uint width;
};

kernel void softmaxTrilBwd(
    device float *oGrad [[ buffer(0) ]],
    device float *iGrad [[ buffer(1) ]],
    device float *softmax [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    // constant struct Parameters &params [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    // uint startIdx = gid.y * params.width;
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float g = 0;
    float s = 0;

    for (uint i = startIdx; i < endIdx; ++i) {
        g = softmax[i] * oGrad[i];
        s += g;
        iGrad[i] += g;
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        iGrad[i] -= softmax[i] * s;
    }
}
