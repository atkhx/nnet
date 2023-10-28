#ifndef KernelMTLBufferCopy_h
#define KernelMTLBufferCopy_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferCopy <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void) copy:(id<MTLBuffer>)dstBuffer
    srcBuffer:(id<MTLBuffer>)srcBuffer
    dstOffset:(uint)dstOffset
    srcOffset:(uint)srcOffset
    length:(uint)length
    withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface KernelMTLBufferCopyImpl : NSObject <KernelMTLBufferCopy>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferCopy_h */
