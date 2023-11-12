#include <metal_stdlib>

using namespace metal;

struct Parameters {
    uint width;
};

kernel void copy(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = srcBuffer[id];
}

kernel void copy2(
    device float* dstBuffer [[ buffer(0) ]],
    device float* srcBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = srcBuffer[id];
    // memcpy(dstBuffer, srcBuffer, 4);
}

kernel void fill(
    device float *dstBuffer [[ buffer(0) ]],
    const uint id [[ thread_position_in_grid ]],
    constant float& value [[ buffer(1) ]])
{
    dstBuffer[id] = value;
}

// todo implement Axpy ( y = A*x + y ) or ( y = A*x + B*y )

kernel void add(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] += srcBuffer[id];
}

kernel void addTo(
    device float *dstBuffer [[ buffer(0) ]],
    device float *aBuffer [[ buffer(1) ]],
    device float *bBuffer [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = aBuffer[id] + bBuffer[id];
}

kernel void addScalar(
    device float *dstBuffer [[ buffer(0) ]],
    constant float& value [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] += value;
}

kernel void mul(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] *= srcBuffer[id];
}

kernel void maxByRow(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx + width;

    float maxValue = srcBuffer[startIdx];
    for (uint i = startIdx+1; i < endIdx; ++i) {
        if (maxValue < srcBuffer[i]) {
            maxValue = srcBuffer[i];
        }
    }

    dstBuffer[gid.y] = maxValue;
}

kernel void subMaxByRow(
    device float *dstBuffer [[ buffer(0) ]],
    device float *maxBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    dstBuffer[gid.y * width+gid.x] -= maxBuffer[gid.y];
}

kernel void exp(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    destinationBuffer[id] = exp(sourceBuffer[id]);
}

kernel void sumByRow(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx + width;

    float sumValue = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        sumValue += srcBuffer[i];
    }

    dstBuffer[gid.y] = sumValue;
    //dstBuffer[gid.y] += srcBuffer[gid.y*width + gid.x];
}

kernel void nllByPos(
    device float *dstBuffer [[ buffer(0) ]],
    device float *smxBuffer [[ buffer(1) ]],
    device float *tgtBuffer [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = -log(smxBuffer[id * width + int(tgtBuffer[id])]);
}

kernel void divOnSum(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    device float *sumBuffer [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    dstBuffer[gid.y * width+gid.x] = srcBuffer[gid.y * width+gid.x] / sumBuffer[gid.y];
}

kernel void relu(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (srcBuffer[id] < 0) {
        dstBuffer[id] = 0;
    } else {
        dstBuffer[id] = srcBuffer[id];
    }
}

kernel void reluBwd(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *mskBuffer [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (mskBuffer[id] > 0) {
        dstBuffer[id] += srcBuffer[id];
    }
}

kernel void dropout(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *mskBuffer [[ buffer(2) ]],
    constant float& probability [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (mskBuffer[id] > probability) {
        dstBuffer[id] = srcBuffer[id];
    } else {
        dstBuffer[id] = 0.0;
    }
}

kernel void dropoutBwd(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *mskBuffer [[ buffer(2) ]],
    constant float& probability [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (mskBuffer[id] > probability) {
        dstBuffer[id] += srcBuffer[id];
    }
}

kernel void updateWithAdam(
    device float *dataBuffer [[ buffer(0) ]],
    device float *gradBuffer [[ buffer(1) ]],
    device float *mBuffer [[ buffer(2) ]],
    device float *vBuffer [[ buffer(3) ]],
    constant float& beta1 [[ buffer(4) ]],
    constant float& beta2 [[ buffer(5) ]],
    constant float& beta1powIterationLR [[ buffer(6) ]],
    constant float& beta2powIteration [[ buffer(7) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    mBuffer[id] = beta1*mBuffer[id] + (1 - beta1)*gradBuffer[id];
    vBuffer[id] = beta2*vBuffer[id] + (1 - beta2)*gradBuffer[id]*gradBuffer[id];

    dataBuffer[id] -= mBuffer[id] * beta1powIterationLR / (sqrt(vBuffer[id] * beta2powIteration) + 0.000000001);
}

kernel void softmaxTril(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float max = srcBuffer[startIdx];
    for (uint i = startIdx+1; i < endIdx; ++i) {
        if (max < srcBuffer[i]) {
            max = srcBuffer[i];
        }
    }

    float sumExp = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] = exp(srcBuffer[i]-max);
        sumExp += dstBuffer[i];
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] /= sumExp;
    }
}

kernel void softmaxBufferTrilBwd(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *smxBuffer [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float g = 0;
    float s = 0;

    for (uint i = startIdx; i < endIdx; ++i) {
        g = smxBuffer[i] * srcBuffer[i];
        s += g;
        dstBuffer[i] += g;
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] -= smxBuffer[i] * s;
    }
}

kernel void crossEntropyPosBwd(
    device float *oGrad [[ buffer(0) ]],
    device float *aGrad [[ buffer(1) ]],
    device float *tgtBuffer [[ buffer(2) ]],
    device float *smxBuffer [[ buffer(3) ]],
    constant uint& chunkSize [[ buffer(4) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    int rowIdx = id / chunkSize;
    float softmaxI = smxBuffer[id];
    int tgtIndex = tgtBuffer[rowIdx];

    if (id == tgtIndex + (rowIdx * chunkSize)) {
        aGrad[id] += oGrad[rowIdx] * (softmaxI - 1);
    } else {
        aGrad[id] += oGrad[rowIdx] * softmaxI;
    }
}

kernel void sqsByRow(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    float val = 0.0;
    for (uint i = gid.y*width; i < (gid.y+1)*width; ++i) {
        val += srcBuffer[i] * srcBuffer[i];
    }
    dstBuffer[gid.y] = sqrt(1e-5 + (val/float(width)));
}
