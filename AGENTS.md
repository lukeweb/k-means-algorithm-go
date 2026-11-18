# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Project overview

`github.com/lukeweb/k-means-algorithm-go` is a small Go library implementing k-means clustering for
n-dimensional points. The goal of the project is to become a **reusable library** for other Go projects, so
public API stability, correctness, documentation and tests matter more than features.

- Language: Go 1.24 (`go.mod`)
- Single package: `pkg/kmeans` (package name `kmeans`)
- Runtime dependencies: none (standard library only)
- Test dependency: `github.com/stretchr/testify` (`suite`, `assert`)

## Layout

| File | Contents |
|---|---|
| `pkg/kmeans/kmeans.go` | `Point`, `InitializeCentroidsFunction`, `KMeans`, `findClosestIndex`, `handleEmptyCentroid` |
| `pkg/kmeans/initializers.go` | `RandomCentroids`, `SmartCentroids` (k-means++) |
| `pkg/kmeans/normalizers.go` | `NormalizePoints` (min-max to `[0,1]`) |
| `pkg/kmeans/calculate_error.go` | `CalculateSSE`, `CalculateMSE` |
| `pkg/kmeans/validator.go` | `ValidatePoints` + sentinel `Err*` variables |
| `pkg/kmeans/math_utils.go` | `distanceSquared`, `mean`, `findMin`, `findMax` (unexported helpers) |
| `.golangci.yaml` | golangci-lint config (**v1 format**, CI uses golangci-lint v1.64.8) |
| `.github/workflows/kmeans-sci.yaml` | CI: `go mod tidy`, golangci-lint, `go test -v ./pkg/kmeans` |
| `REVIEW.md` | Code review report with known bugs, missing tests and roadmap |

## Algorithm flow

1. `initializeCentroids(points, k)` → initial centroids.
2. Repeat exactly `iterations` times (no convergence check):
   - assign each point to the nearest centroid (`findClosestIndex`, squared Euclidean distance, ties → lowest index);
   - recompute each centroid as the mean of its points; an empty cluster's centroid is moved to the point
     farthest from it (`handleEmptyCentroid`).
3. Return `(centroids, assignments)`.

`KMeans` does **not** call `ValidatePoints`; callers must validate first.

## Commands

```bash
go build ./...
go vet ./...
go test ./...
go test -race -coverprofile=coverage.out ./... && go tool cover -func=coverage.out
golangci-lint run ./...   # requires golangci-lint v1.x (config is v1 format; v2 fails to load it)
```

## Testing conventions

- Exported API is tested black-box from package `kmeans_test`; unexported helpers are tested from package
  `kmeans` (`math_utils_test.go`, `kmeans_test.go`).
- testify suites: the entry point **must** be named `TestXxx(t *testing.T)` and call `suite.Run`.
  Suite methods must have **no parameters** (`func (s *XSuite) TestSomething()`), otherwise testify panics.
  > Currently `NewCalculateErrorSuite`, `NewInitializersSuite`, `NewNormalizersSuite`, `NewKmeansSuite` are
  > *not* run by `go test` and some of their tests are wrong — see `REVIEW.md`.
- Unexported helpers use table-driven tests with `t.Run`.
- Compare floats with `InDelta(..., 1e-9)`, never `Equal`, unless values are exactly representable.
- Random initializers: assert properties (length, membership in dataset), not exact values; use a deterministic
  custom `InitializeCentroidsFunction` when testing `KMeans` outcomes.
- After changing tests, run `go test -v ./...` and check that the new tests actually appear in the output and
  that coverage went up.

## Known pitfalls (do not rely on / be careful when changing)

- Centroids returned by `RandomCentroids` / `SmartCentroids` alias the input `Point` slices;
  `handleEmptyCentroid` uses `copy` into them → input data can be mutated.
- `distanceSquared` returns `0` for points of different dimensions instead of failing.
- `SmartCentroids` may return fewer than `k` centroids for datasets with < k distinct points.
- `RandomCentroids` panics if `k > len(points)`; `mean`, `findMin`, `findMax` panic on empty input.
- Initializers use the global `math/rand` source (not seedable by callers).

## Code style

- Run `gofmt`/`goimports` (local prefix `github.com/lukeweb/k-means-algorithm-go`).
- Lines ≤ 120 chars; functions ≤ 80 lines / 40 statements; cyclomatic complexity ≤ 15.
- Every exported symbol needs a doc comment starting with its exact name.
- Wrap or return sentinel errors so callers can use `errors.Is`.
- Mark intentional non-crypto randomness with `// #nosec G404`.

## Git

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`.
- Work on a branch, open PRs into `main`; CI must be green.
- Treat any change to exported signatures as a breaking change for library users (semver).
