package kmeans

import (
	"math"
	"math/rand"
)

// randomCentroids selects k random points from the dataset to serve as initial centroids.
// It randomly permutes the indices and picks the first k points.
func RandomCentroids(points []Point, k int) []Point {
	centroids := make([]Point, k)
	// #nosec G404 -- Random permutation of indices
	perm := rand.Perm(len(points))

	for i := 0; i < k; i++ {
		centroids[i] = points[perm[i]] // Randomly chosen points
	}

	return centroids
}

// smartCentroids initializes centroids using the k-means++ method.
// It selects centroids that are spread out across the data to improve clustering performance.
func SmartCentroids(points []Point, k int) []Point {
	nPoints := len(points)

	// Initialize centroids slice
	centroids := make([]Point, 0, k)

	// #nosec G404 -- Step 1: Randomly pick the first centroid
	firstIndex := rand.Intn(nPoints)
	centroids = append(centroids, points[firstIndex])

	// Step 2: Select the remaining k-1 centroids
	for len(centroids) < k {
		distances := make([]float64, nPoints)
		total := 0.0

		// For each point, calculate the squared distance to the nearest existing centroid
		for i, point := range points {
			minDistance := math.MaxFloat64
			for _, centroid := range centroids {
				d := distance(point, centroid)
				if d < minDistance {
					minDistance = d
				}
			}
			dSquared := math.Pow(minDistance, 2)
			distances[i] = dSquared
			total += dSquared // total must be the sum of squared distances!
		}

		// #nosec G404 -- Pick a new point with probability proportional to distance squared
		randomPoint := rand.Float64() * total
		cumulative := 0.0

		for i, d := range distances {
			cumulative += d
			if cumulative >= randomPoint {
				centroids = append(centroids, points[i])
				break
			}
		}
	}

	return centroids
}
