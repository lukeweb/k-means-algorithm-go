# K-Means Algorithm in Go

This repository provides a basic implementation of the K-Means clustering algorithm written in Go. It allows you to group a set of n-dimensional data points into k clusters based on similarity (Euclidean distance).

## Features

- Supports clustering of 2D, 3D, and n-dimensional points
- Min-Max normalization for consistent scaling
- Error calculation for cluster stability
- Input validation with detailed error handling
- Unit-tested core functions

## Installation

```bash
go get github.com/lukeweb/k-means-algorithm-go
```

## Project Structure
```
k-means-algorithm-go/
├── cmd/
├── pkg/
│   └── kmeans/
│       ├── kmeans.go            # Main algorithm logic
│       ├── calculate_error.go   # Sum of squared error calculation
│       ├── initializers.go      # Centroid initialization logic
│       ├── normalizers.go       # Points normalization logic
│       ├── validator.go         # Input validation
│       └── math_utils.go        # Centroid calculation
└── README.md
```

## Example

```go
package main

import (
    "fmt"

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
    i := 2

    centroids, assignments := kmeans.KMeans(points, k, i, kmeans.RandomCentroids)

    	fmt.Println("Centroids:")
      for _, c := range centroids {
        fmt.Println(c)
      }

      fmt.Println("\nPoints assignments:")
      for i, cluster := range assignments {
        fmt.Printf("point %d -> group %d\n", i+1, cluster+1)
      }

      see := kmeans.SumSquaredError(points, centroids, assignments)

      fmt.Printf("SEE: %d\n", see)
}
```