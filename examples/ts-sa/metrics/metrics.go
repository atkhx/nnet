package metrics

import "github.com/prometheus/client_golang/prometheus"

var Loss = prometheus.NewGauge(prometheus.GaugeOpts{
	Name: "loss",
	Help: "Loss",
})

var LossMean = prometheus.NewGauge(prometheus.GaugeOpts{
	Name: "loss_mean",
	Help: "Loss mean",
})

var AvgLossMean = prometheus.NewGauge(prometheus.GaugeOpts{
	Name: "loss_mean_avg",
	Help: "Loss mean AVG be training chunk",
})

var TrainDuration = prometheus.NewCounter(prometheus.CounterOpts{
	Name: "train_duration",
	Help: "Train duration",
})

func init() {
	prometheus.MustRegister(Loss)
	prometheus.MustRegister(LossMean)
	prometheus.MustRegister(AvgLossMean)
	prometheus.MustRegister(TrainDuration)
}
