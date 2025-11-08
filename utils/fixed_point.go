package utils

import "math"

const ScalingFactor = 1 << 32

func FloatToFixed(f float64) int64 {
	return int64(f * float64(ScalingFactor))
}

func FixedToFloat(i int64) float64 {
	return float64(i) / float64(ScalingFactor)
}

func ComputeZ(w, b, x float64) float64 {
	wFixed := FloatToFixed(w)
	bFixed := FloatToFixed(b)
	xFixed := FloatToFixed(x)

	zScaled := (wFixed * xFixed) / int64(ScalingFactor)
	z := zScaled + bFixed

	return FixedToFloat(z)
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func Predict(w, b, x float64) int {
	z := ComputeZ(w, b, x)
	sig := Sigmoid(z)
	if sig >= 0.5 {
		return 0 // Pass
	}
	return 1
}