package kmeans_test

import (
	"math"
	"testing"

	"github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
	"github.com/stretchr/testify/suite"
)

type ValidatorSuite struct {
	suite.Suite
}

func TestValidatorSuite(t *testing.T) {
	suite.Run(t, new(ValidatorSuite))
}

func (s *ValidatorSuite) TestValidPoints2D() {
	points := []kmeans.Point{{1, 2}, {3, 4}, {5, 6}}
	err := kmeans.ValidatePoints(points, 2)
	s.NoError(err)
}

func (s *ValidatorSuite) TestValidPoints3D() {
	points := []kmeans.Point{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	err := kmeans.ValidatePoints(points, 3)
	s.NoError(err)
}

func (s *ValidatorSuite) TestValidPoints5D() {
	points := []kmeans.Point{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}
	err := kmeans.ValidatePoints(points, 2)
	s.NoError(err)
}

func (s *ValidatorSuite) TestEmptySliceReturnsError() {
	err := kmeans.ValidatePoints([]kmeans.Point{}, 2)
	s.ErrorIs(err, kmeans.ErrNoPoints)
}

func (s *ValidatorSuite) TestZeroClustersReturnsError() {
	points := []kmeans.Point{{1, 2}, {3, 4}}
	err := kmeans.ValidatePoints(points, 0)
	s.ErrorIs(err, kmeans.ErrNegativeNumberOfClusters)
}

func (s *ValidatorSuite) TestNegativeClustersReturnsError() {
	points := []kmeans.Point{{1, 2}, {3, 4}}
	err := kmeans.ValidatePoints(points, -1)
	s.ErrorIs(err, kmeans.ErrNegativeNumberOfClusters)
}

func (s *ValidatorSuite) TestNotEnoughPointsReturnsError() {
	points := []kmeans.Point{{1, 2}}
	err := kmeans.ValidatePoints(points, 2)
	s.ErrorIs(err, kmeans.ErrNotEnoughPoints)
}

func (s *ValidatorSuite) TestZeroDimensionsReturnsError() {
	points := []kmeans.Point{{}}
	err := kmeans.ValidatePoints(points, 1)
	s.ErrorIs(err, kmeans.ErrInvalidNumberOfDimensions)
}

func (s *ValidatorSuite) TestInconsistentDimensionsReturnsError() {
	points := []kmeans.Point{{1, 2}, {3, 4, 5}}
	err := kmeans.ValidatePoints(points, 2)
	s.ErrorIs(err, kmeans.ErrInconsistentDimensions)
}

func (s *ValidatorSuite) TestNaNValueReturnsError() {
	points := []kmeans.Point{{1, 2}, {math.NaN(), 4}}
	err := kmeans.ValidatePoints(points, 2)
	s.ErrorIs(err, kmeans.ErrInvalidNumericValue)
}

func (s *ValidatorSuite) TestInfValueReturnsError() {
	points := []kmeans.Point{{1, 2}, {math.Inf(1), 4}}
	err := kmeans.ValidatePoints(points, 2)
	s.ErrorIs(err, kmeans.ErrInvalidNumericValue)
}
