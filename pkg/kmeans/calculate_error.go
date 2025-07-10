package kmeans

import "math"

// CalculateSSE calculates the total within-cluster sum of squared errors (SSE).
//
// Arguments:
//   - points: dataset of n-dimensional points
//   - centroids: final centroids after k-means clustering
//   - assignments: index of the centroid assigned to each point
//
// Returns:
//   - SSE: float64, the total error measuring compactness of clusters
//
// Formula:
//
//	SSE = Σ ||x_i - c_{a_i}||^2
func CalculateSSE(points, centroids []Point, assignments []int) float64 {
	sse := 0.0

	// Iterate over all points
	for i, point := range points {
		centroid := centroids[assignments[i]] // Get the assigned centroid
		dist := distance(point, centroid)     // Euclidean distance between point and centroid
		sse += math.Pow(dist, 2)              // Add the squared distance to the total error
	}

	return sse
}

// CalculateMSE calculates the Mean Squared Error (MSE) for the clustering result.
// MSE is computed by averaging the squared Euclidean distances between each point
// and its assigned cluster centroid.
//
// Mathematics:
// SSE = Σ ||x_i - μ_c_i||²   (Sum of Squared Errors for all points)
// MSE = SSE / n              (Mean Squared Error, where n is the number of points)
//
// Arguments:
//   - points: the original dataset (slice of points, each a float64 vector)
//   - assignments: a slice where each index corresponds to the cluster ID assigned to the point
//   - centroids: current centroid positions for each cluster
//
// Returns:
//   - Mean Squared Error (float64)
func CalculateMSE(points, centroids []Point, assignments []int) float64 {
	return CalculateSSE(points, centroids, assignments) / float64(len(points)) // Return Mean Squared Error
}
