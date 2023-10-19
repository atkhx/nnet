#ifndef KernelMTLBufferMul_h
#define KernelMTLBufferMul_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferMul <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void) mul:(id<MTLBuffer>)destinationBuffer
        multiplierBuffer:(id<MTLBuffer>)multiplierBuffer
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface KernelMTLBufferMulImpl : NSObject <KernelMTLBufferMul>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferMul_h */
