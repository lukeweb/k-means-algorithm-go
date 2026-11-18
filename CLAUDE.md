# CLAUDE.md

@AGENTS.md

## Claude-specific notes

- Do not change Go source, tests, CI or lint config unless the user explicitly asks for it; review/doc tasks
  should only touch documentation files.
- Before claiming tests pass, run `go test -v -count=1 ./...` and confirm the relevant suites appear in the
  output (several suites were silently not running — see `REVIEW.md`).
- Report coverage with `go test -coverprofile=/tmp/cover.out ./... && go tool cover -func=/tmp/cover.out`
  (write the profile outside the repo).
- For experiments that require editing code, use a throwaway copy outside the repo (e.g. `/tmp`).
- Keep `README.md`, `AGENTS.md` and `REVIEW.md` in sync with the public API when it changes.
