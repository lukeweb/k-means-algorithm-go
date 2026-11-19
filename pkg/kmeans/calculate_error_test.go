package kmeans_test

import (
	"math"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
)

type CalculateErrorSuite struct {
	suite.Suite
}

func TestCalculateErrorSuite(t *testing.T) {
	suite.Run(t, new(CalculateErrorSuite))
}

func (s *CalculateErrorSuite) TestCalculatingSSE_2DPerfectClustering() {
	points := []kmeans.Point{{1, 1}, {1, 1}}
	centroids := []kmeans.Point{{1, 1}}
	assignments := []int{0, 0}
	expected := 0.0

	actual := kmeans.CalculateSSE(points, centroids, assignments)
	s.InDelta(expected, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingSSE_2DSimpleError() {
	points := []kmeans.Point{{0, 0}, {2, 2}}
	centroids := []kmeans.Point{{1, 1}}
	assignments := []int{0, 0}
	expected := 4.0

	actual := kmeans.CalculateSSE(points, centroids, assignments)
	s.InDelta(expected, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingSSE_3DTwoClusters() {
	points := []kmeans.Point{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	centroids := []kmeans.Point{{2, 3, 4}, {6, 7, 8}}
	assignments := []int{0, 0, 1}
	expected := 18.0 // 3 + 12 + 3

	actual := kmeans.CalculateSSE(points, centroids, assignments)
	s.InDelta(expected, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingMSE_2DPerfectClustering() {
	points := []kmeans.Point{{1, 1}, {1, 1}}
	centroids := []kmeans.Point{{1, 1}}
	assignments := []int{0, 0}
	expected := 0.0

	actual := kmeans.CalculateMSE(points, centroids, assignments)
	s.InDelta(expected, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingMSE_2DUniformError() {
	points := []kmeans.Point{{0, 0}, {2, 2}}
	centroids := []kmeans.Point{{1, 1}}
	assignments := []int{0, 0}
	expected := 2.0

	actual := kmeans.CalculateMSE(points, centroids, assignments)
	s.InDelta(expected, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingMSE_3DTwoClusters() {
	points := []kmeans.Point{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	centroids := []kmeans.Point{{2, 3, 4}, {6, 7, 8}}
	assignments := []int{0, 0, 1}
	expected := 6.0 // SSE 18 / 3 points

	actual := kmeans.CalculateMSE(points, centroids, assignments)
	s.InDelta(expected, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingSSE_EmptyInputIsZero() {
	actual := kmeans.CalculateSSE(nil, nil, nil)

	s.InDelta(0.0, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingMSE_EmptyInputIsNaN() {
	actual := kmeans.CalculateMSE(nil, nil, nil)

	s.True(math.IsNaN(actual), "expected NaN, got %v", actual)
}

func (s *CalculateErrorSuite) TestCalculatingSSE_2DMultipleClusters() {
	points := []kmeans.Point{{0, 0}, {0, 2}, {10, 10}, {10, 14}}
	centroids := []kmeans.Point{{0, 1}, {10, 12}}
	assignments := []int{0, 0, 1, 1}

	actual := kmeans.CalculateSSE(points, centroids, assignments)

	s.InDelta(10.0, actual, 1e-9) // 1 + 1 + 4 + 4
}

func (s *CalculateErrorSuite) TestCalculatingMSE_2DMultipleClusters() {
	points := []kmeans.Point{{0, 0}, {0, 2}, {10, 10}, {10, 14}}
	centroids := []kmeans.Point{{0, 1}, {10, 12}}
	assignments := []int{0, 0, 1, 1}

	actual := kmeans.CalculateMSE(points, centroids, assignments)

	s.InDelta(2.5, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingSSE_1D() {
	points := []kmeans.Point{{1}, {3}, {8}}
	centroids := []kmeans.Point{{2}, {8}}
	assignments := []int{0, 0, 1}

	actual := kmeans.CalculateSSE(points, centroids, assignments)

	s.InDelta(2.0, actual, 1e-9)
}

func (s *CalculateErrorSuite) TestCalculatingMSE_1D() {
	points := []kmeans.Point{{1}, {3}, {8}}
	centroids := []kmeans.Point{{2}, {8}}
	assignments := []int{0, 0, 1}

	actual := kmeans.CalculateMSE(points, centroids, assignments)

	s.InDelta(2.0/3.0, actual, 1e-9)
}
