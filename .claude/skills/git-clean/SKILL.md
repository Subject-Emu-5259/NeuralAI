---
name: git-clean
description: "Clean up git history and branches: prune merged branches, identify stale work, and organize the repository."
source: community
allowed-tools: "*"
user-invocable: true
---

# Git Cleanup

Safely clean up a git repository: prune merged branches, identify stale work, and improve repository hygiene.

## STEP 1: ASSESS CURRENT STATE

Analyze the repository:

- Total number of local and remote branches
- Branches already merged into main/master
- Branches with no recent commits (stale)
- Branches with unmerged work
- Tags and their age

## STEP 2: IDENTIFY CLEANUP CANDIDATES

Categorize branches:

### Safe to Delete
- Branches fully merged into the main branch
- Branches with no commits beyond the main branch
- Remote tracking branches for deleted remotes

### Needs Review
- Branches with unmerged commits but no recent activity (>30 days)
- Branches from former team members
- Branches with unclear purpose (no descriptive name)

### Keep
- Main/master and develop branches
- Active feature branches with recent commits
- Release and hotfix branches

## STEP 3: PRESENT PLAN

Before deleting anything, present:

```
Safe to delete (X branches):
- feature/old-feature (merged 45 days ago)
- bugfix/typo-fix (merged 12 days ago)

Needs review (Y branches):
- experiment/ml-model (last commit 60 days ago, 3 unmerged commits)
- wip/dashboard-v2 (last commit 90 days ago, 12 unmerged commits)

Keeping (Z branches):
- main
- feature/current-work (active)
```

## STEP 4: EXECUTE (with permission)

After user confirmation:
- Delete merged local branches
- Prune remote tracking branches
- Optionally delete confirmed stale remote branches
- Run `git gc` to optimize the repository

## STEP 5: REPORT

Summary:
- Branches deleted (local and remote)
- Branches kept
- Repository size before/after (if significant)
- Remaining cleanup recommendations
