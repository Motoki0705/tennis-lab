<!--
PR body template for `gh pr create --body-file`.

Suggested workflow:
1. Copy this file to a tmp file.
   BODY_FILE="$(mktemp /tmp/pr-body-XXXXXX.md)"
   cp .github/pull_request_template.md "$BODY_FILE"
2. Edit the tmp file and remove guidance comments if they are no longer needed.
3. Create the PR with:
   gh pr create --body-file "$BODY_FILE"
-->

## Summary

<!-- What this PR changes in 2-5 lines. -->

-

## Changes

<!-- List concrete implementation changes. -->

-
-
-

## Validation

<!-- List commands you ran or checks you performed. If untested, say why. -->

-

## Related Issues

<!-- Use GitHub closing/reference keywords as needed. -->

- Closes #
- References #

## Notes

<!-- Optional: reviewers should know this before merging. Delete if not needed. -->

-
