The previous turn was interrupted again when the machine ran low on memory. This was not a problem on the worker's side. The parent has confirmed that the old processes are gone and has restored the same session.

The worktree is clean, and two commits have not been pushed yet: `9f636229f fix(agents): recover interrupted publication revisions` and `63d6e777e Merge origin/main`. The PR is still at 238bf9ba3. Check the current state with `git log` and `git status`, then continue.
1. For pr-1876-2, confirm with the target tests that the regression test fails before the fix and passes after it.
2. Finish pr-1876-1 as the earlier instructions said: tests showing the gate rejects such PRs, workflow filter cases, and the boundary explanation in the 3-language README.
3. Push, and report once the CI `Python` result for the target HEAD is available.

Memory is tight, so run pytest with `-n 1` on the target test files only, and do not start several heavy processes at the same time. Push the commits that are already done as early as possible so they are not lost. The other constraints and the reporting conditions are the same as in the earlier instructions. Do not write local absolute paths in instructions or reports, not even as examples.
