package kmeans

import (
	"math"
)

// Point represents a point in n-dimensional space.
// Example: [2.5, 3.1, 0.8] is a 3-dimensional point.
type Point []float64
type InitializeCentroidsFunction func(points []Point, k int) []Point

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

// KMeans performs k-means clustering on the given dataset.
//
// Parameters:
// - points: a slice of n-dimensional data points to cluster.
// - k: the number of clusters to form.
// - iterations: the maximum number of iterations to run.
// - initializeCentroids: a function that initializes the initial cluster centroids.
//
// Returns:
// - centroids: the final positions of the cluster centroids.
// - assignments: a slice mapping each point to its assigned cluster index.
func KMeans(points []Point, k, iterations int, initializeCentroids InitializeCentroidsFunction) ([]Point, []int) {
	centroids := initializeCentroids(points, k)
	assignments := make([]int, len(points))

	for iter := 0; iter < iterations; iter++ {
		// Create empty clusters
		clusters := make([][]Point, k)

		// Assign each point to the nearest centroid
		for i, point := range points {
			minDist := math.MaxFloat64
			closestIndex := -1

			for index, centroid := range centroids {
				distance := distance(point, centroid)
				if distance < minDist {
					minDist = distance
					closestIndex = index
				}
			}
			clusters[closestIndex] = append(clusters[closestIndex], point)
			assignments[i] = closestIndex
		}

		// Update centroids by calculating the mean of assigned points
		for j := range centroids {
			if len(clusters[j]) > 0 {
				centroids[j] = mean(clusters[j])
			}
		}
	}

	return centroids, assignments
}
