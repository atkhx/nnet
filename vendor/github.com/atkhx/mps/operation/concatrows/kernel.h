#ifndef ConcatRowsKernel_h
#define ConcatRowsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol ConcatRowsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset;

@end


@interface ConcatRowsKernelImpl : NSObject <ConcatRowsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* ConcatRowsKernel_h */
