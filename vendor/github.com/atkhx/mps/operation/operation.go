package operation

import "github.com/atkhx/mps"

type Operation interface {
	Forward(buffer *mps.MTLCommandBuffer)
	Backward(buffer *mps.MTLCommandBuffer)
}
