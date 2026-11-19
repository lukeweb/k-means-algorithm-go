package kmeans

import (
	"math"
)

// Point represents a point in n-dimensional space.
// Example: [2.5, 3.1, 0.8] is a 3-dimensional point.
type Point []float64
type InitializeCentroidsFunction func(points []Point, k int) []Point

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
			closestIndex := findClosestIndex(point, centroids)
			clusters[closestIndex] = append(clusters[closestIndex], point)
			assignments[i] = closestIndex
		}

		// Update centroids by calculating the mean of assigned points
		for j := range centroids {
			if len(clusters[j]) > 0 {
				centroids[j] = mean(clusters[j])
			} else {
				handleEmptyCentroid(j, centroids, points)
			}
		}
	}

	return centroids, assignments
}

// findClosestIndex returns the index of the centroid closest to the given point.
//
// Parameters:
// - point: the n-dimensional data point to assign.
// - centroids: a slice of current centroids.
//
// Returns:
// - the index of the nearest centroid. Returns -1 if centroids slice is empty.
func findClosestIndex(point Point, centroids []Point) int {
	if len(centroids) == 0 {
		return -1 // no centroids to compare
	}

	minDist := math.MaxFloat64
	closestIndex := -1

	for index, centroid := range centroids {
		distance := distanceSquared(point, centroid)
		if distance < minDist {
			minDist = distance
			closestIndex = index
		}
	}

	return closestIndex
}

// handleEmptyCentroid reinitializes an empty cluster centroid.
//
// Parameters:
// - centroidIdx: the index of the empty centroid to reinitialize.
// - centroids: the slice of current centroid positions.
// - points: the dataset used for clustering.
//
// The centroid is relocated to the data point farthest from it in order
// to reduce clustering error and allow the algorithm to continue.
func handleEmptyCentroid(
	controlIdx int,
	centroids []Point,
	points []Point,
) {
	var (
		furthestPointIdx = -1
		maxDist          = -1.0
	)

	for idx, point := range points {
		d := distanceSquared(point, centroids[controlIdx])
		if d > maxDist {
			maxDist = d
			furthestPointIdx = idx
		}
	}

	if furthestPointIdx >= 0 {
		copy(centroids[controlIdx], points[furthestPointIdx])
	}
}
