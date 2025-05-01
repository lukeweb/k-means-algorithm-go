package kmeans

import "math"

// distance calculates the Euclidean distance between two n-dimensional points p and q.
// Formula: sqrt(Σ(p_i - q_i)^2 for i=1..n)
func distance(p, q Point) float64 {
	var sum float64 = 0

	for i := range p {
		diff := p[i] - q[i]
		sum += math.Pow(diff, 2)
	}

	return math.Sqrt(sum)
}

// mean computes the centroid (average point) of a group of points.
// Each coordinate of the centroid is the average of the corresponding coordinates of the points.
func mean(points []Point) Point {
	n := len(points[0]) // Number of dimensions
	mean := make(Point, n)

	// Sum all points dimension-wise
	for _, p := range points {
		for i := 0; i < n; i++ {
			mean[i] += p[i]
		}
	}

	// Divide by number of points to get the average
	for i := 0; i < n; i++ {
		mean[i] /= float64(len(points))
	}

	return mean
}
