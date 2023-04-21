package num

func CalcConvOutputSize(
	iw, ih int,
	fw, fh int,
	padding int,
	stride int,
) (int, int) {
	return (iw-fw+2*padding)/stride + 1, (ih-fh+2*padding)/stride + 1
}
