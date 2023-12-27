#include <metal_stdlib>

using namespace metal;

kernel void ropeRows(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& featuresCount [[ buffer(2) ]],
    constant uint& headSize [[ buffer(3) ]],
    constant uint& contextLength [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i1 = gid.z*featuresCount*contextLength + gid.y*featuresCount + gid.x*2;
    uint i2 = i1 + 1;

    float freq = 1.0 / pow(10000.0, float((gid.x*2) % headSize)/float(headSize));
    float val = float(gid.y) * freq;

    float fcr = cos(val);
    float fci = sin(val);

    outputData[i1] = inputData[i1]*fcr - inputData[i2]*fci;
    outputData[i2] = inputData[i1]*fci + inputData[i2]*fcr;
}

kernel void ropeRowsGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& featuresCount [[ buffer(2) ]],
    constant uint& headSize [[ buffer(3) ]],
    constant uint& contextLength [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i1 = gid.z*featuresCount*contextLength + (gid.y)*featuresCount + gid.x*2;
    uint i2 = gid.z*featuresCount*contextLength + (gid.y)*featuresCount + gid.x*2  + 1;

    float freq = 1.0 / pow(10000.0, float((gid.x*2) % headSize)/float(headSize));
    float val = float(gid.y) * freq;

    float fcr = cos(val);
    float fci = sin(val);

    float outputGrad0 = outputGrad[i1];
    float outputGrad1 = outputGrad[i2];

    inputGrad[i1] +=  outputGrad0*fcr + outputGrad1*fci;
    inputGrad[i2] += -outputGrad0*fci + outputGrad1*fcr;
}
