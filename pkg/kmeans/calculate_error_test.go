package kmeans_test

import (
	"testing"

	"github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
	"github.com/stretchr/testify/suite"
)

type CalculateErrorSuite struct {
	suite.Suite
}

func NewCalculateErrorSuite(t *testing.T) {
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
	expected := 6.0

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
	expected := 2.0

	actual := kmeans.CalculateMSE(points, centroids, assignments)
	s.InDelta(expected, actual, 1e-9)
}
