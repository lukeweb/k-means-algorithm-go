# K-Means Algorithm in Go

A small, dependency-free (runtime) implementation of the [k-means](https://en.wikipedia.org/wiki/K-means_clustering)
clustering algorithm in Go. It groups n-dimensional points into `k` clusters using squared Euclidean distance.

## Features

- Clustering of 1D, 2D, 3D … n-dimensional points (`kmeans.Point` is a `[]float64`)
- Two centroid initializers:
  - `RandomCentroids` – picks `k` distinct random points from the dataset
  - `SmartCentroids` – [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) seeding (spreads initial centroids)
- Pluggable initialization via `InitializeCentroidsFunction`
- Empty-cluster handling (an empty centroid is moved to the farthest point)
- Min-max normalization (`NormalizePoints`) to put all dimensions on the `[0, 1]` scale
- Quality metrics: Sum of Squared Errors (`CalculateSSE`) and Mean Squared Error (`CalculateMSE`)
- Input validation (`ValidatePoints`) with sentinel errors usable with `errors.Is`

## Requirements

- Go **1.24** or newer

## Installation

```bash
go get github.com/lukeweb/k-means-algorithm-go
```

```go
import "github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
```

## Quick start

Recommended flow: **validate → (optionally) normalize → cluster → evaluate**.

```go
package main

import (
	"fmt"
	"log"

	"github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
)

func main() {
	points := []kmeans.Point{
		{1.0, 2.0},
		{1.5, 1.8},
		{5.0, 8.0},
		{8.0, 8.0},
		{1.0, 0.6},
		{9.0, 11.0},
	}

	k := 2
	maxIterations := 100

	// KMeans does NOT validate its input – always validate first.
	if err := kmeans.ValidatePoints(points, k); err != nil {
		log.Fatal(err)
	}

	// Optional: scale every dimension to [0, 1] so no feature dominates the distance.
	normalized := kmeans.NormalizePoints(points)

	centroids, assignments := kmeans.KMeans(normalized, k, maxIterations, kmeans.SmartCentroids)

	fmt.Println("Centroids (normalized space):")
	for _, c := range centroids {
		fmt.Println(c)
	}

	fmt.Println("\nPoint assignments:")
	for i, cluster := range assignments {
		fmt.Printf("point %d %v -> cluster %d\n", i, points[i], cluster)
	}

	fmt.Printf("\nSSE: %.4f\n", kmeans.CalculateSSE(normalized, centroids, assignments))
	fmt.Printf("MSE: %.4f\n", kmeans.CalculateMSE(normalized, centroids, assignments))
}
```

## API reference

All symbols live in package `kmeans` (`pkg/kmeans`).

| Symbol | Description |
|---|---|
| `type Point []float64` | A point in n-dimensional space. |
| `type InitializeCentroidsFunction func(points []Point, k int) []Point` | Strategy used by `KMeans` to choose the starting centroids. |
| `KMeans(points []Point, k, iterations int, init InitializeCentroidsFunction) ([]Point, []int)` | Runs k-means for exactly `iterations` iterations. Returns the centroids and, for every input point, the index of its cluster. |
| `RandomCentroids(points []Point, k int) []Point` | Chooses `k` distinct random points as centroids. |
| `SmartCentroids(points []Point, k int) []Point` | k-means++ initialization. |
| `NormalizePoints(points []Point) []Point` | Returns a new slice with min-max normalized coordinates (`(x-min)/(max-min)`; `0` when a dimension has no spread). |
| `ValidatePoints(points []Point, k int) error` | Checks the input before clustering (see errors below). |
| `CalculateSSE(points, centroids []Point, assignments []int) float64` | Σ‖xᵢ − c₍ₐᵢ₎‖² – total within-cluster error. |
| `CalculateMSE(points, centroids []Point, assignments []int) float64` | `SSE / len(points)`. |

### Validation errors

`ValidatePoints` returns the first problem it finds, as one of these sentinel errors:

| Error | Condition |
|---|---|
| `ErrNoPoints` | `points` is empty |
| `ErrNegativeNumberOfClusters` | `k <= 0` |
| `ErrNotEnoughPoints` | `len(points) < k` |
| `ErrInvalidNumberOfDimensions` | the first point has 0 dimensions |
| `ErrInconsistentDimensions` | points have different numbers of dimensions |
| `ErrInvalidNumericValue` | a coordinate is `NaN` or `±Inf` |

```go
if err := kmeans.ValidatePoints(points, k); errors.Is(err, kmeans.ErrNotEnoughPoints) {
	// handle
}
```

### Custom initializer

Any function matching `InitializeCentroidsFunction` can be used, e.g. for deterministic results in tests:

```go
firstK := func(points []kmeans.Point, k int) []kmeans.Point {
	centroids := make([]kmeans.Point, k)
	for i := range centroids {
		centroids[i] = append(kmeans.Point(nil), points[i]...) // copy, do not alias input
	}
	return centroids
}

centroids, assignments := kmeans.KMeans(points, 2, 100, firstK)
```

## Limitations (current version)

Be aware of the following when using the library in other projects:

- **No input validation inside `KMeans`** – invalid input (e.g. `k > len(points)`, empty slice) causes a panic.
  Call `ValidatePoints` first.
- **Fixed number of iterations** – there is no convergence tolerance / early stopping.
- **Input may be modified** – the built-in initializers return centroids that share memory with the input
  points; when a cluster becomes empty the centroid is overwritten in place, which also changes the
  corresponding input point. Pass a copy of your data if you need it unchanged.
- **`SmartCentroids` may return fewer than `k` centroids** when the dataset has fewer than `k` distinct points;
  `KMeans` then returns fewer centroids as well.
- **Non-reproducible results** – initializers use the global `math/rand` source; there is no way to pass a seed.
  Use a custom initializer if you need determinism.
- `CalculateMSE` returns `NaN` for an empty dataset.
- `NormalizePoints` does not return the min/max used, so new points cannot be projected into the same scale and
  centroids cannot be converted back to original units.

See [REVIEW.md](REVIEW.md) for the full code review and improvement roadmap.

## Project structure

```
k-means-algorithm-go/
├── .github/workflows/kmeans-sci.yaml  # CI: lint + tests
├── .golangci.yaml                     # golangci-lint configuration (v1 format)
├── pkg/
│   └── kmeans/
│       ├── kmeans.go            # KMeans, closest-centroid search, empty-cluster handling
│       ├── initializers.go      # RandomCentroids, SmartCentroids (k-means++)
│       ├── normalizers.go       # NormalizePoints (min-max)
│       ├── calculate_error.go   # CalculateSSE, CalculateMSE
│       ├── validator.go         # ValidatePoints + sentinel errors
│       ├── math_utils.go        # distanceSquared, mean, findMin, findMax
│       └── *_test.go            # unit tests (testify)
├── go.mod
└── README.md
```

## Development

```bash
# run tests
go test ./...

# run tests with race detector and coverage report
go test -race -coverprofile=coverage.out ./...
go tool cover -func=coverage.out

# lint (config is in golangci-lint v1 format; CI uses v1.64.8)
golangci-lint run ./...
```

CI (`.github/workflows/kmeans-sci.yaml`) runs golangci-lint and the tests on every push and pull request to `main`.

## License

No license file is included yet. Until one is added, the code is not licensed for reuse by others.
