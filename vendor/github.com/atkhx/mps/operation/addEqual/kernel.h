#ifndef AddEqualKernel_h
#define AddEqualKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol AddEqualKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad;

@end


@interface AddEqualKernelImpl : NSObject <AddEqualKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* AddEqualKernel_h */
