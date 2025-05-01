package kmeans_test

import (
	"reflect"
	"testing"

	"github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
	"github.com/stretchr/testify/suite"
)

type InitializersSuite struct {
	suite.Suite
}

func NewInitializersSuite(t *testing.T) {
	suite.Run(t, new(InitializersSuite))
}

func (s *InitializersSuite) TestRandomCentroids2D() {
	points := []kmeans.Point{
		{3.2, 7.1},
		{8.5, 2.4},
		{1.0, 9.9},
		{6.6, 3.3},
		{0.5, 4.7},
	}
	centroids := kmeans.RandomCentroids(points, 3)
	s.Len(centroids, 3)

	for _, c := range centroids {
		s.True(pointInSlice(c, points), "centroid %+v not in dataset", c)
	}
}

func (s *InitializersSuite) TestRandomCentroids3D() {
	points := []kmeans.Point{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
		{13, 14, 15},
	}
	centroids := kmeans.RandomCentroids(points, 3)
	s.Len(centroids, 3)

	for _, c := range centroids {
		s.True(pointInSlice(c, points), "centroid %+v not in dataset", c)
	}
}

func (s *InitializersSuite) TestRandomCentroids5D() {
	points := []kmeans.Point{
		{1.1, 9.2, 3.3, 4.4, 5.5},
		{6.6, 7.7, 8.8, 0.1, 2.2},
		{3.3, 4.4, 5.5, 6.6, 7.7},
		{8.8, 9.9, 1.1, 2.2, 3.3},
		{0.4, 1.5, 2.6, 3.7, 4.8},
	}
	centroids := kmeans.RandomCentroids(points, 3)
	s.Len(centroids, 3)

	for _, c := range centroids {
		s.True(pointInSlice(c, points), "centroid %+v not in dataset", c)
	}
}

func (s *InitializersSuite) TestSmartCentroids2D() {
	points := []kmeans.Point{
		{3.2, 7.1},
		{8.5, 2.4},
		{1.0, 9.9},
		{6.6, 3.3},
		{0.5, 4.7},
	}
	centroids := kmeans.SmartCentroids(points, 3)
	s.Len(centroids, 3)

	for _, c := range centroids {
		s.True(pointInSlice(c, points), "centroid %+v not in dataset", c)
	}
}

func (s *InitializersSuite) TestSmartCentroids3D() {
	points := []kmeans.Point{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
		{13, 14, 15},
	}
	centroids := kmeans.SmartCentroids(points, 3)
	s.Len(centroids, 3)

	for _, c := range centroids {
		s.True(pointInSlice(c, points), "centroid %+v not in dataset", c)
	}
}

func (s *InitializersSuite) TestSmartCentroids5D() {
	points := []kmeans.Point{
		{1.1, 9.2, 3.3, 4.4, 5.5},
		{6.6, 7.7, 8.8, 0.1, 2.2},
		{3.3, 4.4, 5.5, 6.6, 7.7},
		{8.8, 9.9, 1.1, 2.2, 3.3},
		{0.4, 1.5, 2.6, 3.7, 4.8},
	}
	centroids := kmeans.SmartCentroids(points, 3)
	s.Len(centroids, 3)

	for _, c := range centroids {
		s.True(pointInSlice(c, points), "centroid %+v not in dataset", c)
	}
}

func pointInSlice(p kmeans.Point, list []kmeans.Point) bool {
	for _, el := range list {
		if reflect.DeepEqual(p, el) {
			return true
		}
	}
	return false
}
