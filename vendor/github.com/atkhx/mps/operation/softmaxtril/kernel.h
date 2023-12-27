#ifndef SoftmaxtrilKernel_h
#define SoftmaxtrilKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol SoftmaxtrilKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        outputData:(id<MTLBuffer>)outputData
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

@end


@interface SoftmaxtrilKernelImpl : NSObject <SoftmaxtrilKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* SoftmaxtrilKernel_h */
