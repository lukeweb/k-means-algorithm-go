package kmeans_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/lukeweb/k-means-algorithm-go/pkg/kmeans"
)

type NormalizersSuite struct {
	suite.Suite
}

func TestNormalizersSuite(t *testing.T) {
	suite.Run(t, new(NormalizersSuite))
}

func (s *NormalizersSuite) TestNormalize2D_LinearSpread() {
	points := []kmeans.Point{
		{2, 4},
		{6, 8},
		{10, 12},
	}

	expected := []kmeans.Point{
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
	}

	actual := kmeans.NormalizePoints(points)

	s.Equal(expected, actual)
}

func (s *NormalizersSuite) TestNormalize2D_UnevenSpread() {
	points := []kmeans.Point{
		{1, 2},
		{5, 10},
		{9, 6},
	}

	expected := []kmeans.Point{
		{0.0, 0.0},
		{0.5, 1.0},
		{1.0, 0.5},
	}

	actual := kmeans.NormalizePoints(points)
	s.Equal(expected, actual)
}

func (s *NormalizersSuite) TestNormalize3D_LinearSpread() {
	points := []kmeans.Point{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}

	expected := []kmeans.Point{
		{0.0, 0.0, 0.0},
		{0.5, 0.5, 0.5},
		{1.0, 1.0, 1.0},
	}

	actual := kmeans.NormalizePoints(points)
	s.Equal(expected, actual)
}

func (s *NormalizersSuite) TestNormalize3D_MixedSpread() {
	points := []kmeans.Point{
		{1, 100, 3},
		{4, 200, 6},
		{7, 150, 9},
	}

	expected := []kmeans.Point{
		{0.0, 0.0, 0.0},
		{0.5, 1.0, 0.5},
		{1.0, 0.5, 1.0},
	}

	actual := kmeans.NormalizePoints(points)
	s.Equal(expected, actual)
}

func (s *NormalizersSuite) TestNormalize5D_VariedSpread() {
	points := []kmeans.Point{
		{1, 2, 3, 4, 5},
		{6, 7, 8, 9, 10},
		{11, 12, 13, 14, 15},
	}

	expected := []kmeans.Point{
		{0.0, 0.0, 0.0, 0.0, 0.0},
		{0.5, 0.5, 0.5, 0.5, 0.5},
		{1.0, 1.0, 1.0, 1.0, 1.0},
	}

	actual := kmeans.NormalizePoints(points)
	s.Equal(expected, actual)
}

func (s *NormalizersSuite) TestNormalize2D_ZeroVariance() {
	points := []kmeans.Point{
		{5, 5},
		{5, 5},
		{5, 5},
	}

	expected := []kmeans.Point{
		{0.0, 0.0},
		{0.0, 0.0},
		{0.0, 0.0},
	}

	actual := kmeans.NormalizePoints(points)
	s.Equal(expected, actual)
}

func (s *NormalizersSuite) TestNormalizeSinglePointIsAllZeros() {
	actual := kmeans.NormalizePoints([]kmeans.Point{{3, -7, 42}})

	s.Equal([]kmeans.Point{{0, 0, 0}}, actual)
}

func (s *NormalizersSuite) TestNormalize1D() {
	actual := kmeans.NormalizePoints([]kmeans.Point{{2}, {4}, {6}})

	s.Equal([]kmeans.Point{{0}, {0.5}, {1}}, actual)
}

func (s *NormalizersSuite) TestNormalizeNegativeValues() {
	points := []kmeans.Point{{-10, -1}, {0, -3}, {10, -2}}

	expected := []kmeans.Point{
		{0.0, 1.0},
		{0.5, 0.0},
		{1.0, 0.5},
	}

	s.Equal(expected, kmeans.NormalizePoints(points))
}

func (s *NormalizersSuite) TestNormalizeMixedZeroVarianceAndSpreadDimensions() {
	points := []kmeans.Point{{5, 1}, {5, 3}}

	expected := []kmeans.Point{
		{0.0, 0.0},
		{0.0, 1.0},
	}

	s.Equal(expected, kmeans.NormalizePoints(points))
}

func (s *NormalizersSuite) TestNormalizeDoesNotModifyOrAliasInput() {
	points := []kmeans.Point{{1, 2}, {3, 4}}

	actual := kmeans.NormalizePoints(points)
	actual[0][0] = 99

	s.Equal([]kmeans.Point{{1, 2}, {3, 4}}, points)
}
