package metal

import (
	"context"
)

var device = &Native{}

type Native struct{}

type Operation func(context.Context)
