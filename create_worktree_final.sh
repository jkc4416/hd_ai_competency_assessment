#!/usr/bin/env bash

#-----------------------------------------------------------------------
# Worktree helper: enforce branch name as `issue-<NUMBER>`,
# reuse existing worktree if present, and cd into it.
#
# Usage (MUST source):
#   source ./gwt.sh <issue-id-or-name-ending-with-number> [<base-ref>]
#   e.g.) source ./gwt.sh 72
#         source ./gwt.sh feature-x-72 main
#-----------------------------------------------------------------------

# Ensure bash
[ -n "$BASH_VERSION" ] || { echo "Please run under bash."; return 1 2>/dev/null || exit 1; }

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"

# Must be inside a git repo
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "❌ Not inside a git repository."
  return 1 2>/dev/null || exit 1
}

# Args
if [ $# -lt 1 ]; then
  echo "Error: No issue identifier provided."
  echo "Usage: source $SCRIPT_NAME <issue-id-or-name-ending-with-number> [<base-ref>]"
  return 1 2>/dev/null || exit 1
fi

ARGUMENT="$1"
BASE_REF="${2:-HEAD}"

# Extract issue number (pure digits or last hyphen token)
if [[ "$ARGUMENT" =~ ^[0-9]+$ ]]; then
  ISSUE_NUMBER="$ARGUMENT"
else
  ISSUE_NUMBER="$(echo "$ARGUMENT" | awk -F'-' '{print $NF}')"
fi

if ! [[ "$ISSUE_NUMBER" =~ ^[0-9]+$ ]]; then
  echo "❌ Could not extract a numeric issue id from '$ARGUMENT'."
  echo "   Provide a number (e.g., '72') or a name ending with a number (e.g., 'feature-auth-72')."
  return 1 2>/dev/null || exit 1
fi

BRANCH_NAME="issue-$ISSUE_NUMBER"

# Resolve paths relative to repo root (avoid CWD surprises)
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKTREE_DIR_NAME="$BRANCH_NAME"
WORKTREE_PATH="$REPO_ROOT/../worktree/$WORKTREE_DIR_NAME"

# Ensure parent directory exists
mkdir -p "$(dirname "$WORKTREE_PATH")" || {
  echo "❌ Failed to create parent directory for $WORKTREE_PATH"
  return 1 2>/dev/null || exit 1
}

# Create branch if missing (from BASE_REF)
if ! git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
  echo "ℹ️  Creating branch '$BRANCH_NAME' from '$BASE_REF'..."
  git branch "$BRANCH_NAME" "$BASE_REF" || {
    echo "❌ Failed to create branch '$BRANCH_NAME' from '$BASE_REF'"
    return 1 2>/dev/null || exit 1
  }
fi

# Helper: is WORKTREE_PATH already registered as a worktree?
is_registered_worktree() {
  # Use porcelain for stable parsing
  git worktree list --porcelain | awk '
    $1=="worktree" { path=$2 }
    $1=="branch"   { branch=$2 }
    $1==""         { if (path!="") { print path; path=""; branch="" } }
    END            { if (path!="") print path }
  ' | while IFS= read -r p; do
      # compare by absolute path
      [ "$(readlink -f "$p")" = "$(readlink -f "$WORKTREE_PATH")" ] && return 0
    done
  return 1
}

# Decide action
if is_registered_worktree; then
  echo "ℹ️  Worktree already registered at: $WORKTREE_PATH (reusing)"
elif [ -d "$WORKTREE_PATH" ]; then
  # Directory exists but not registered: try to detect if it's a git worktree dir
  if [ -f "$WORKTREE_PATH/.git" ] || [ -d "$WORKTREE_PATH/.git" ]; then
    echo "ℹ️  Directory exists and seems to be a git worktree: $WORKTREE_PATH (reusing)"
  else
    # Non-empty unrelated dir? refuse unless empty
    if [ "$(ls -A "$WORKTREE_PATH" 2>/dev/null | wc -l)" -gt 0 ]; then
      echo "❌ Directory exists and is not an empty/git worktree: $WORKTREE_PATH"
      echo "   Aborting to avoid clobbering unrelated files. Remove or choose another issue id."
      return 1 2>/dev/null || exit 1
    fi
    # Empty directory: safe to add
    if git worktree add "$WORKTREE_PATH" "$BRANCH_NAME"; then
      echo "✅ Successfully created worktree at: $WORKTREE_PATH"
    else
      echo "❌ Failed to create worktree at $WORKTREE_PATH"
      return 1 2>/dev/null || exit 1
    fi
  fi
else
  # No directory: normal add
  if git worktree add "$WORKTREE_PATH" "$BRANCH_NAME"; then
    echo "✅ Successfully created worktree at: $WORKTREE_PATH"
  else
    echo "❌ Failed to create worktree at $WORKTREE_PATH"
    echo "   Tip: if a stale entry exists, run: git worktree prune"
    return 1 2>/dev/null || exit 1
  fi
fi

# cd into worktree
cd "$WORKTREE_PATH" || { echo "❌ Failed to cd into $WORKTREE_PATH"; return 1 2>/dev/null || exit 1; }

# Ensure we are on the intended branch (fix if user switched it)
CURRENT_BRANCH="$(git symbolic-ref --short -q HEAD || true)"
if [ "$CURRENT_BRANCH" != "$BRANCH_NAME" ]; then
  echo "ℹ️  Checking out branch '$BRANCH_NAME' (was '${CURRENT_BRANCH:-detached}')"
  git checkout "$BRANCH_NAME" || { echo "❌ Failed to checkout '$BRANCH_NAME'"; return 1 2>/dev/null || exit 1; }
fi

echo "➡️  Ready at $(pwd)  (branch: $BRANCH_NAME)"

# Kick off Claude for this issue (interactive progress is up to the CLI config)
echo "🧠 Running claude command for issue: #$ISSUE_NUMBER"
# claude -p "/user:resolve-issue $ISSUE_NUMBER"
# claude
claude --dangerously-skip-permissions
# claude -p "/resolve-issue $ISSUE_NUMBER"