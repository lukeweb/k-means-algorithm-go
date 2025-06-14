package kmeans

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var distanceTestsCases = []struct {
	name     string
	p, q     Point
	expected float64
}{
	{"2D: basic", Point{0, 0}, Point{3, 4}, 5.0},
	{"2D: same point", Point{1, 2}, Point{1, 2}, 0.0},
	{"2D: negative", Point{-1, -1}, Point{1, 1}, 2.8284271247461903},

	{"3D: basic", Point{0, 0, 0}, Point{1, 2, 2}, 3.0},
	{"3D: arbitrary", Point{1, 1, 1}, Point{4, 5, 6}, 7.0710678118654755},
	{"3D: same", Point{5, 5, 5}, Point{5, 5, 5}, 0.0},

	{"5D: mirrored", Point{1, 2, 3, 4, 5}, Point{5, 4, 3, 2, 1}, 6.324555320336759},
	{"5D: ones", Point{0, 0, 0, 0, 0}, Point{1, 1, 1, 1, 1}, 2.23606797749979},
	{"5D: same", Point{1, 2, 3, 4, 5}, Point{1, 2, 3, 4, 5}, 0.0},
}

var meanTestsCases = []struct {
	name     string
	points   []Point
	expected Point
}{
	{"2D: avg center", []Point{{0, 0}, {2, 2}}, Point{1, 1}},
	{"2D: avg multiple", []Point{{1, 2}, {3, 4}, {5, 6}}, Point{3, 4}},
	{"2D: opposite", []Point{{-1, -1}, {1, 1}}, Point{0, 0}},

	{"3D: avg origin", []Point{{0, 0, 0}, {6, 6, 6}}, Point{3, 3, 3}},
	{"3D: step", []Point{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, Point{4, 5, 6}},
	{"3D: zero", []Point{{-2, -2, -2}, {2, 2, 2}}, Point{0, 0, 0}},

	{"5D: mirrored", []Point{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}, Point{3, 3, 3, 3, 3}},
	{"5D: range", []Point{{0, 0, 0, 0, 0}, {10, 10, 10, 10, 10}}, Point{5, 5, 5, 5, 5}},
	{"5D: arithmetic", []Point{{2, 4, 6, 8, 10}, {4, 6, 8, 10, 12}, {6, 8, 10, 12, 14}}, Point{4, 6, 8, 10, 12}},
}

var minTestCases = []struct {
	name     string
	points   []Point
	expected Point
}{
	{"2D: min left corner", []Point{{0, 0}, {2, 2}}, Point{0, 0}},
	{"2D: various", []Point{{1, 2}, {3, 1}, {5, 6}}, Point{1, 1}},
	{"2D: negatives", []Point{{-1, -3}, {1, -2}}, Point{-1, -3}},

	{"3D: origin min", []Point{{0, 0, 0}, {6, 6, 6}}, Point{0, 0, 0}},
	{"3D: scattered", []Point{{4, 5, 6}, {1, 2, 3}, {7, 8, 9}}, Point{1, 2, 3}},
	{"3D: negatives", []Point{{-2, -5, -1}, {-1, -2, -3}}, Point{-2, -5, -3}},

	{"5D: mirrored", []Point{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}, Point{1, 2, 3, 2, 1}},
	{"5D: zeros and tens", []Point{{0, 0, 0, 0, 0}, {10, 10, 10, 10, 10}}, Point{0, 0, 0, 0, 0}},
	{"5D: mixed", []Point{{4, 6, 8, 10, 12}, {2, 4, 6, 8, 10}, {6, 8, 10, 12, 14}}, Point{2, 4, 6, 8, 10}},
}

var maxTestCases = []struct {
	name     string
	points   []Point
	expected Point
}{
	{"2D: max right corner", []Point{{0, 0}, {2, 2}}, Point{2, 2}},
	{"2D: various", []Point{{1, 2}, {3, 1}, {5, 6}}, Point{5, 6}},
	{"2D: negatives", []Point{{-1, -3}, {-2, -2}}, Point{-1, -2}},

	{"3D: origin max", []Point{{0, 0, 0}, {6, 6, 6}}, Point{6, 6, 6}},
	{"3D: scattered", []Point{{4, 5, 6}, {1, 2, 3}, {7, 8, 9}}, Point{7, 8, 9}},
	{"3D: negatives", []Point{{-2, -5, -1}, {-1, -2, -3}}, Point{-1, -2, -1}},

	{"5D: mirrored", []Point{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}, Point{5, 4, 3, 4, 5}},
	{"5D: zeros and tens", []Point{{0, 0, 0, 0, 0}, {10, 10, 10, 10, 10}}, Point{10, 10, 10, 10, 10}},
	{"5D: mixed", []Point{{4, 6, 8, 10, 12}, {2, 4, 6, 8, 10}, {6, 8, 10, 12, 14}}, Point{6, 8, 10, 12, 14}},
}

func TestDistanceCalculation(t *testing.T) {
	for _, testCase := range distanceTestsCases {
		t.Run(testCase.name, func(t *testing.T) {
			actual := distance(testCase.p, testCase.q)

			assert.InDelta(t, testCase.expected, actual, 1e-9)
		})
	}
}

func TestMeanCalculation(t *testing.T) {
	for _, testCase := range meanTestsCases {
		t.Run(testCase.name, func(t *testing.T) {
			actual := mean(testCase.points)
			assert.Equal(t, len(testCase.expected), len(actual))

			for i := range testCase.expected {
				assert.InDelta(t, testCase.expected[i], actual[i], 1e-9)
			}
		})
	}
}

func TestFindMinPointsValue(t *testing.T) {
	for _, testCase := range minTestCases {
		t.Run(testCase.name, func(t *testing.T) {
			actual := findMin(testCase.points)
			assert.Equal(t, len(testCase.expected), len(actual))

			for i := range testCase.expected {
				assert.InDelta(t, testCase.expected[i], actual[i], 1e-9, "Mismatch at dimension %d", i)
			}
		})
	}
}

func TestFindMaxPointsValue(t *testing.T) {
	for _, testCase := range maxTestCases {
		t.Run(testCase.name, func(t *testing.T) {
			actual := findMax(testCase.points)
			assert.Equal(t, len(testCase.expected), len(actual))

			for i := range testCase.expected {
				assert.InDelta(t, testCase.expected[i], actual[i], 1e-9, "Mismatch at dimension %d", i)
			}
		})
	}
}
