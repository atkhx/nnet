#ifndef MPSCustomKernel_h
#define MPSCustomKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MPSCustomKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) copy:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length;

- (void) fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length;

- (void) add:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length;

// todo add offset/length
- (void) addTo:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer;

- (void) addScalar:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value;

// todo add mulTo
- (void) mul:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length;

- (void) relu:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer;

- (void)reluBwd:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer;

- (void) exp:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

- (void) sumByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

// todo naming (here we div each value in row on their sum
- (void) divOnSum:(id<MTLCommandBuffer>)commandBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount;

- (void) softmax:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

- (void) dropout:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer
        probability:(float)probability;

- (void) dropoutBwd:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer
        probability:(float)probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;

- (void) updateWithAdam:(id<MTLCommandBuffer>)commandBuffer
        dataBuffer:(id<MTLBuffer>)dataBuffer
        gradBuffer:(id<MTLBuffer>)gradBuffer
        mBuffer:(id<MTLBuffer>)mBuffer
        vBuffer:(id<MTLBuffer>)vBuffer
        beta1:(float)beta1
        beta2:(float)beta2
        beta1powIterationLR:(float)beta1powIterationLR
        beta2powIteration:(float)beta2powIteration;

- (void) softmaxTril:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

- (void) softmaxTrilBwd:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;


- (void) crossEntropyPos:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        chunkSize:(uint)chunkSize;

- (void) crossEntropyPosBwd:(id<MTLCommandBuffer>)commandBuffer
        oGrad:(id<MTLBuffer>)oGrad
        aGrad:(id<MTLBuffer>)aGrad
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        chunkSize:(uint)chunkSize;

- (void) rmsNorm:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        chunkSize:(uint)chunkSize;

@end


@interface MPSCustomKernelImpl : NSObject <MPSCustomKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MPSCustomKernel_h */
