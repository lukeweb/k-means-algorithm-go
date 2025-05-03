package kmeans

import (
	"errors"
	"math"
)

var (
	ErrNoPoints                  = errors.New("no points provided")
	ErrNegativeNumberOfClusters  = errors.New("number of clusters must be positive")
	ErrNotEnoughPoints           = errors.New("not enough points for clusters")
	ErrInvalidNumberOfDimensions = errors.New("points must have at least one dimension")
	ErrInconsistentDimensions    = errors.New("points must all have the same number of dimensions")
	ErrInvalidNumericValue       = errors.New("points contain invalid numeric values (NaN or Inf)")
)

func ValidatePoints(points []Point, k int) error {
	if len(points) == 0 {
		return ErrNoPoints
	}

	if k <= 0 {
		return ErrNegativeNumberOfClusters
	}

	if len(points) < k {
		return ErrNotEnoughPoints
	}

	dim := len(points[0])
	if dim == 0 {
		return ErrInvalidNumberOfDimensions
	}

	for _, p := range points {
		if len(p) != dim {
			return ErrInconsistentDimensions
		}
		for _, val := range p {
			if math.IsNaN(val) || math.IsInf(val, 0) {
				return ErrInvalidNumericValue
			}
		}
	}

	return nil
}
