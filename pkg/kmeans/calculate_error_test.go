package kmeans_test

import (
	"testing"

	"github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
	"github.com/stretchr/testify/suite"
)

var sseTestCases = []struct {
	name        string
	points      []kmeans.Point
	centroids   []kmeans.Point
	assignments []int
	expected    float64
}{
	{
		name:        "2D - perfect clustering",
		points:      []kmeans.Point{{1, 1}, {1, 1}},
		centroids:   []kmeans.Point{{1, 1}},
		assignments: []int{0, 0},
		expected:    0.0,
	},
	{
		name:        "2D - simple error",
		points:      []kmeans.Point{{0, 0}, {2, 2}},
		centroids:   []kmeans.Point{{1, 1}},
		assignments: []int{0, 0},
		expected:    4.0,
	},
	{
		name:        "3D - two clusters",
		points:      []kmeans.Point{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		centroids:   []kmeans.Point{{2, 3, 4}, {6, 7, 8}},
		assignments: []int{0, 0, 1},
		expected:    6.0,
	},
}

var mseTestCases = []struct {
	name        string
	points      []kmeans.Point
	centroids   []kmeans.Point
	assignments []int
	expected    float64
}{
	{
		name:        "2D - perfect clustering",
		points:      []kmeans.Point{{1, 1}, {1, 1}},
		centroids:   []kmeans.Point{{1, 1}},
		assignments: []int{0, 0},
		expected:    0.0,
	},
	{
		name:        "2D - uniform error",
		points:      []kmeans.Point{{0, 0}, {2, 2}},
		centroids:   []kmeans.Point{{1, 1}},
		assignments: []int{0, 0},
		expected:    2.0,
	},
	{
		name:        "3D - two clusters",
		points:      []kmeans.Point{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		centroids:   []kmeans.Point{{2, 3, 4}, {6, 7, 8}},
		assignments: []int{0, 0, 1},
		expected:    2.0,
	},
}

type CalculateErrorSuite struct {
	suite.Suite
}

func NewCalculateErrorSuite(t *testing.T) {
	suite.Run(t, new(CalculateErrorSuite))
}

func (s *CalculateErrorSuite) TestCalculatingSSE() {
	for _, testCase := range sseTestCases {
		s.Run(testCase.name, func() {
			actual := kmeans.CalculateSSE(testCase.points, testCase.centroids, testCase.assignments)

			s.InDelta(testCase.expected, actual, 1e-9)
		})
	}
}

func (s *CalculateErrorSuite) TestCalculatingMSE() {
	for _, testCase := range mseTestCases {
		s.Run(testCase.name, func() {
			actual := kmeans.CalculateMSE(testCase.points, testCase.centroids, testCase.assignments)

			s.InDelta(testCase.expected, actual, 1e-9)
		})
	}
}
