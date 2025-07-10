package kmeans

// NormalizePoints applies min-max normalization to a set of points.
// Each coordinate of the point is scaled to the range [0, 1] using the formula:
//
//	normalized = (x - min) / (max - min)
//
// If max == min (no spread on a given axis), the result is set to 0.
// This indicates all points have the same value in that dimension.
func NormalizePoints(points []Point) []Point {
	normalized := make([]Point, len(points)) // output slice of the same length
	minValue := findMin(points)              // minimum values per dimension
	maxValue := findMax(points)              // maximum values per dimension

	for i, point := range points {
		n := make(Point, len(point)) // new normalized point
		for j, val := range point {
			denominator := maxValue[j] - minValue[j] // value range in dimension j
			if denominator == 0 {
				n[j] = 0 // no variance – set to 0, i.e., start of the scale
			} else {
				n[j] = (val - minValue[j]) / denominator
			}
		}
		normalized[i] = n
	}

	return normalized
}
