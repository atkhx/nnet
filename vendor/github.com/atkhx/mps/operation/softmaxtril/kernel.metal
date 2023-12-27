#include <metal_stdlib>

using namespace metal;

kernel void softmaxTril(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& colsCount [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * colsCount;
    uint endIdx = startIdx + gid.y + 1;

    float max = inputData[startIdx];
    for (uint i = startIdx+1; i < endIdx; ++i) {
        if (max < inputData[i]) {
            max = inputData[i];
        }
    }

    float sumExp = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        outputData[i] = exp(inputData[i]-max);
        sumExp += outputData[i];
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        outputData[i] /= sumExp;
    }
}

kernel void softmaxBufferTrilBwd(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    constant uint& colsCount [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * colsCount;
    uint endIdx = startIdx +  gid.y+1;

    float g = 0;
    float s = 0;

    for (uint i = startIdx; i < endIdx; ++i) {
        g = outputData[i] * outputGrad[i];
        s += g;
        inputGrad[i] += g;
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        inputGrad[i] -= outputData[i] * s;
    }
}
