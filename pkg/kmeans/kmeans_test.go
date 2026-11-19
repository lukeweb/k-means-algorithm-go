package kmeans

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type KmeansSuite struct {
	suite.Suite
}

func TestKmeansSuite(t *testing.T) {
	suite.Run(t, new(KmeansSuite))
}

func (s *KmeansSuite) TestFindClosestIndexEmptyCentroids() {
	point := Point{1, 2}
	centroids := []Point{}

	result := findClosestIndex(point, centroids)

	s.Equal(-1, result)
}

func (s *KmeansSuite) TestFindClosestIndexSingleCentroid() {
	point := Point{1, 1}
	centroids := []Point{{0, 0}}

	result := findClosestIndex(point, centroids)

	s.Equal(0, result)
}

func (s *KmeansSuite) TestFindClosestIndexCloserToFirst() {
	point := Point{1, 1}
	centroids := []Point{
		{0, 0},
		{5, 5},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(0, result)
}

func (s *KmeansSuite) TestFindClosestIndexCloserToSecond() {
	point := Point{4, 4}
	centroids := []Point{
		{0, 0},
		{5, 5},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(1, result)
}

func (s *KmeansSuite) TestFindClosestIndexExactMatch() {
	point := Point{3, 3}
	centroids := []Point{
		{0, 0},
		{3, 3},
		{6, 6},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(1, result)
}

func (s *KmeansSuite) TestFindClosestIndexTieReturnsFirst() {
	point := Point{1, 1}
	centroids := []Point{
		{0, 2},
		{2, 0},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(0, result)
}

func (s *KmeansSuite) TestFindClosestIndexHigherDimensions() {
	point := Point{1, 2, 3}
	centroids := []Point{
		{10, 10, 10},
		{1, 2, 4},
		{0, 0, 0},
	}

	result := findClosestIndex(point, centroids)

	s.Equal(1, result)
}

// fixedCentroids returns an initializer that ignores the dataset and always
// returns deep copies of the given centroids, which makes KMeans deterministic
// and keeps the centroids from sharing memory with the input points.
func fixedCentroids(centroids ...Point) InitializeCentroidsFunction {
	return func(_ []Point, _ int) []Point {
		result := make([]Point, len(centroids))
		for i, c := range centroids {
			result[i] = append(Point(nil), c...)
		}

		return result
	}
}

func twoBlobs() []Point {
	return []Point{
		{0, 0}, {0, 1}, {1, 0},
		{10, 10}, {10, 11}, {11, 10},
	}
}

func (s *KmeansSuite) TestKMeansTwoSeparatedBlobs() {
	points := twoBlobs()

	centroids, assignments := KMeans(points, 2, 10, fixedCentroids(Point{0, 0}, Point{10, 10}))

	s.Equal([]int{0, 0, 0, 1, 1, 1}, assignments)
	s.Require().Len(centroids, 2)
	s.InDeltaSlice(Point{1.0 / 3, 1.0 / 3}, centroids[0], 1e-9)
	s.InDeltaSlice(Point{31.0 / 3, 31.0 / 3}, centroids[1], 1e-9)
}

func (s *KmeansSuite) TestKMeansSingleClusterIsMeanOfAllPoints() {
	points := []Point{{0, 0}, {2, 4}, {4, 8}}

	centroids, assignments := KMeans(points, 1, 5, fixedCentroids(Point{0, 0}))

	s.Equal([]int{0, 0, 0}, assignments)
	s.Require().Len(centroids, 1)
	s.InDeltaSlice(Point{2, 4}, centroids[0], 1e-9)
}

func (s *KmeansSuite) TestKMeansKEqualsNumberOfPointsGivesZeroError() {
	points := []Point{{0, 0}, {5, 5}, {9, 9}}

	centroids, assignments := KMeans(points, 3, 3, fixedCentroids(points[0], points[1], points[2]))

	s.Equal([]int{0, 1, 2}, assignments)
	s.Require().Len(centroids, 3)

	for i, c := range centroids {
		s.InDeltaSlice(points[i], c, 1e-9)
	}
}

func (s *KmeansSuite) TestKMeansZeroIterationsReturnsInitialCentroids() {
	points := twoBlobs()

	centroids, assignments := KMeans(points, 2, 0, fixedCentroids(Point{0, 0}, Point{10, 10}))

	s.Equal([]Point{{0, 0}, {10, 10}}, centroids)
	s.Equal(make([]int, len(points)), assignments, "no assignment step runs, all indexes stay 0")
}

func (s *KmeansSuite) TestKMeansAssignmentsHaveValidLengthAndRange() {
	points := twoBlobs()
	k := 2

	_, assignments := KMeans(points, k, 10, fixedCentroids(Point{0, 0}, Point{10, 10}))

	s.Len(assignments, len(points))

	for i, a := range assignments {
		s.GreaterOrEqual(a, 0, "assignment %d", i)
		s.Less(a, k, "assignment %d", i)
	}
}

func (s *KmeansSuite) TestKMeansConvergedResultIsStableWithMoreIterations() {
	points := twoBlobs()
	initializer := fixedCentroids(Point{0, 0}, Point{10, 10})

	fewCentroids, fewAssignments := KMeans(points, 2, 5, initializer)
	manyCentroids, manyAssignments := KMeans(points, 2, 50, initializer)

	s.Equal(fewCentroids, manyCentroids)
	s.Equal(fewAssignments, manyAssignments)
}

func (s *KmeansSuite) TestKMeansEmptyClusterIsRelocatedToFarthestPoint() {
	// Every point is closest to the first centroid, so the second cluster is empty.
	points := []Point{{0, 0}, {1, 1}, {2, 2}}

	centroids, _ := KMeans(points, 2, 1, fixedCentroids(Point{0, 0}, Point{50, 50}))

	s.Require().Len(centroids, 2)
	s.InDeltaSlice(Point{1, 1}, centroids[0], 1e-9, "mean of all points")
	s.InDeltaSlice(Point{0, 0}, centroids[1], 1e-9, "moved to the point farthest from {50,50}")
}

func (s *KmeansSuite) TestHandleEmptyCentroidMovesToFarthestPoint() {
	centroids := []Point{{0, 0}, {1, 1}}
	points := []Point{{0, 0}, {1, 1}, {5, 5}}

	handleEmptyCentroid(1, centroids, points)

	s.Equal(Point{5, 5}, centroids[1])
	s.Equal(Point{0, 0}, centroids[0], "other centroids are untouched")
}

func (s *KmeansSuite) TestHandleEmptyCentroidWithoutPointsIsNoop() {
	centroids := []Point{{3, 3}}

	handleEmptyCentroid(0, centroids, nil)

	s.Equal([]Point{{3, 3}}, centroids)
}
