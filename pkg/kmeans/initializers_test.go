package kmeans_test

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
)

type InitializersSuite struct {
	suite.Suite
}

func TestInitializersSuite(t *testing.T) {
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

func (s *InitializersSuite) TestRandomCentroidsSingleCluster() {
	points := []kmeans.Point{{1, 1}, {2, 2}, {3, 3}}

	centroids := kmeans.RandomCentroids(points, 1)

	s.Require().Len(centroids, 1)
	s.True(pointInSlice(centroids[0], points))
}

func (s *InitializersSuite) TestRandomCentroidsKEqualsNumberOfPointsReturnsEachPointOnce() {
	points := []kmeans.Point{{1, 1}, {2, 2}, {3, 3}, {4, 4}}

	centroids := kmeans.RandomCentroids(points, len(points))

	s.Require().Len(centroids, len(points))

	for _, p := range points {
		s.Equal(1, countPoint(p, centroids), "point %v must appear exactly once", p)
	}
}

func (s *InitializersSuite) TestRandomCentroidsAreDistinct() {
	points := []kmeans.Point{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}

	for run := 0; run < 100; run++ {
		centroids := kmeans.RandomCentroids(points, 3)

		for _, c := range centroids {
			s.Equal(1, countPoint(c, centroids), "run %d: duplicate centroid %v", run, c)
		}
	}
}

func (s *InitializersSuite) TestSmartCentroidsSingleCluster() {
	points := []kmeans.Point{{1, 1}, {2, 2}, {3, 3}}

	centroids := kmeans.SmartCentroids(points, 1)

	s.Require().Len(centroids, 1)
	s.True(pointInSlice(centroids[0], points))
}

func (s *InitializersSuite) TestSmartCentroidsKEqualsNumberOfDistinctPointsReturnsEachPointOnce() {
	points := []kmeans.Point{{1, 1}, {2, 2}, {3, 3}, {4, 4}}

	for run := 0; run < 50; run++ {
		centroids := kmeans.SmartCentroids(points, len(points))

		s.Require().Len(centroids, len(points), "run %d", run)

		for _, p := range points {
			s.Equal(1, countPoint(p, centroids), "run %d: point %v must appear exactly once", run, p)
		}
	}
}

// Documents current behavior (see REVIEW.md, H1): with fewer than k distinct
// points k-means++ stops early and returns fewer than k centroids.
func (s *InitializersSuite) TestSmartCentroidsIdenticalPointsReturnsFewerThanK() {
	points := []kmeans.Point{{1, 1}, {1, 1}, {1, 1}}

	centroids := kmeans.SmartCentroids(points, 3)

	s.Len(centroids, 1)
}

func (s *InitializersSuite) TestSmartCentroidsPicksOneCentroidFromEachFarApartGroup() {
	const far = 1e6

	groupA := []kmeans.Point{{0, 0}, {0, 1}}
	groupB := []kmeans.Point{{far, far}, {far, far + 1}}
	points := append(append([]kmeans.Point{}, groupA...), groupB...)

	// Probability of picking both centroids from one group is ~1e-12 per run.
	for run := 0; run < 100; run++ {
		centroids := kmeans.SmartCentroids(points, 2)

		s.Require().Len(centroids, 2, "run %d", run)

		fromA := pointInSlice(centroids[0], groupA) != pointInSlice(centroids[1], groupA)
		fromB := pointInSlice(centroids[0], groupB) != pointInSlice(centroids[1], groupB)
		s.True(fromA && fromB, "run %d: centroids %v are not from different groups", run, centroids)
	}
}

func countPoint(p kmeans.Point, list []kmeans.Point) int {
	count := 0

	for _, el := range list {
		if reflect.DeepEqual(p, el) {
			count++
		}
	}

	return count
}
