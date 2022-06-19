//go:generate mockgen -package=$GOPACKAGE -source=$GOFILE -destination=interfaces_mock.go
package activation

type Activation interface {
	Forward(v float64) float64
	Backward(v float64) float64
}
