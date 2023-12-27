#include <metal_stdlib>

using namespace metal;

kernel void rmsByRows(
    device float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    float val = 0.0;
    for (uint i = gid.y*width; i < (gid.y+1)*width; ++i) {
        val += input[i] * input[i];
    }
    output[gid.y] = sqrt(1e-5 + (val / float(width)));
}

kernel void divRows(
    device float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    device float *divider [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    output[gid.y*width + gid.x] = input[gid.y*width + gid.x] / divider[gid.y];
}

kernel void divRowsDividerGrads(
    device float *aggData [[ buffer(0) ]],
    device float *aggGrad [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    device float *outputGrad [[ buffer(3) ]],
    constant uint& chunkSize [[ buffer(4) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float ssGrad = 0.0;
    for (uint i = id * chunkSize; i < (id+1)*chunkSize; ++i) {
        ssGrad -= outputGrad[i] * outputData[i];
    }
    aggGrad[id] = ssGrad / (aggData[id] * aggData[id] * float(chunkSize));
}

kernel void divRowsInputGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    device float *aggData [[ buffer(3) ]],
    device float *aggGrad [[ buffer(4) ]],
    constant uint& chunkSize [[ buffer(5) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    uint row = id / chunkSize;

    inputGrad[id] += outputGrad[id]/aggData[row] + (aggGrad[row] * inputData[id]);
}

