package kmeans

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type KmeansSuite struct {
	suite.Suite
}

func NewKmeansSuite(t *testing.T) {
	suite.Run(t, new(KmeansSuite))
}

func (s *KmeansSuite) TestFindClosestIndexEmptyCentroids(t *testing.T) {
	point := Point{1, 2}
	centroids := []Point{}

	result := findClosestIndex(point, centroids)

	s.Equal(-1, result)
}

func (s *KmeansSuite) TestFindClosestIndexSingleCentroid(t *testing.T) {
	point := Point{1, 1}
	centroids := []Point{{0, 0}}

	result := findClosestIndex(point, centroids)

	s.Equal(0, result)
}

func (s *KmeansSuite) TestFindClosestIndexCloserToFirst(t *testing.T) {
	point := Point{1, 1}
	centroids := []Point{
		{0, 0},
		{5, 5},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(0, result)
}

func (s *KmeansSuite) TestFindClosestIndexCloserToSecond(t *testing.T) {
	point := Point{4, 4}
	centroids := []Point{
		{0, 0},
		{5, 5},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(1, result)
}

func (s *KmeansSuite) TestFindClosestIndexExactMatch(t *testing.T) {
	point := Point{3, 3}
	centroids := []Point{
		{0, 0},
		{3, 3},
		{6, 6},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(1, result)
}

func (s *KmeansSuite) TestFindClosestIndexTieReturnsFirst(t *testing.T) {
	point := Point{1, 1}
	centroids := []Point{
		{0, 2},
		{2, 0},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(0, result)
}

func (s *KmeansSuite) TestFindClosestIndexHigherDimensions(t *testing.T) {
	point := Point{1, 2, 3}
	centroids := []Point{
		{10, 10, 10},
		{1, 2, 4},
		{0, 0, 0},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(1, result)
}
