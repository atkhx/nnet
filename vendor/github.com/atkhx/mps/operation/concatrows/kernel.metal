#include <metal_stdlib>

using namespace metal;


kernel void concatRows(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& inputWidth [[ buffer(2) ]],
    constant uint& outputWidth [[ buffer(3) ]],
    constant uint& outputOffset [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    outputData[(gid.y*outputWidth) + outputOffset + gid.x] = inputData[(gid.y*inputWidth) + gid.x];
}

kernel void concatRowsBwd(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& inputWidth [[ buffer(2) ]],
    constant uint& outputWidth [[ buffer(3) ]],
    constant uint& outputOffset [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    inputGrad[(gid.y*inputWidth) + gid.x] += outputGrad[(gid.y*outputWidth) + outputOffset + gid.x];
}


