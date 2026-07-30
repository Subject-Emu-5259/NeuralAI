---
name: ci-cd
description: "Set up or improve a CI/CD pipeline: build, test, lint, security scan, and deploy stages with proper caching and parallelism."
user-invocable: true
disable-model-invocation: false
model: sonnet
source: pandaos
allowed-tools: Read, Write, Bash, Glob
---

# CI/CD Pipeline Setup

Scaffolds or improves a continuous integration and deployment pipeline for the project.

## STEP 1: DETECT ENVIRONMENT

Parse $ARGUMENTS for:
- CI platform: GitHub Actions (default), GitLab CI, CircleCI, Bitbucket Pipelines
- Target environment: production, staging, preview
- Deployment target: Vercel, AWS, Fly.io, Docker, custom

If not specified, detect from existing CI config files in the repo.

## STEP 2: AUDIT EXISTING PIPELINE

Read any existing CI configuration files:
- `.github/workflows/`
- `.gitlab-ci.yml`
- `circle.yml`

Identify:
- What stages already exist
- What is missing or broken
- Performance bottlenecks (no caching, sequential jobs that should be parallel)

## STEP 3: DESIGN THE PIPELINE

A production-grade pipeline has these stages in order:

```
1. Install dependencies (with caching)
2. Lint (fail fast — no point running tests on code that doesn't meet standards)
3. Type check
4. Unit tests
5. Integration tests (if applicable)
6. Build
7. Security scan (npm audit, Snyk)
8. Deploy to staging (on main branch push)
9. Deploy to production (on release tag)
```

Parallelize independent stages (lint + type-check can run simultaneously).

## STEP 4: WRITE THE PIPELINE CONFIG

For GitHub Actions:
```yaml
name: CI/CD
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run lint
      - run: npm run type-check
      - run: npm test

  build:
    needs: quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run build
```

## STEP 5: CACHING STRATEGY

- Cache `node_modules` by `package-lock.json` hash — saves 30-60 seconds per run
- Cache build artifacts between stages when possible
- Never cache secrets or environment-specific files

## STEP 6: SECRETS MANAGEMENT

- All secrets stored in CI environment variables, never in code
- Production secrets different from staging secrets
- Principle of least privilege: CI service account has only the permissions it needs

## STEP 7: VERIFY

Trigger the pipeline and confirm:
- All stages pass
- Cache hits on subsequent runs (check timing)
- Deployment completes successfully

## SHIFT LEFT PRINCIPLE

Move quality checks as early as possible. The later a bug is found, the more expensive it is to fix.

| Stage | What Runs | Feedback Time |
|-------|----------|---------------|
| **Pre-commit** | Lint, format, type-check (via lint-staged) | < 5 seconds |
| **PR opened** | Full test suite, security scan | < 5 minutes |
| **PR merged** | Build, deploy to staging | < 10 minutes |
| **Release tagged** | Deploy to production | < 15 minutes |

## FASTER IS SAFER

Speed is a safety feature. Slow pipelines cause developers to batch changes, skip CI, or merge without waiting for results. Target these times:

| Metric | Target | Why |
|--------|--------|-----|
| Lint + type-check | < 60s | Developers won't wait longer |
| Unit tests | < 3 min | Must complete before attention shifts |
| Full pipeline | < 10 min | Longer pipelines get ignored |
| Deploy to staging | < 5 min after merge | Fast feedback on real environment |

## QUALITY GATE PIPELINE

Every merge to main passes through these gates in order. If any gate fails, the pipeline stops.

```
Gate 1: Lint + Format (fail fast on style)
Gate 2: Type Check (catch type errors before tests)
Gate 3: Unit Tests (catch logic errors)
Gate 4: Build (catch compilation errors)
Gate 5: Integration Tests (catch boundary errors)
Gate 6: Security Scan (catch vulnerabilities)
Gate 7: Deploy Preview/Staging (catch environment errors)
```

## BUILD CACHING STRATEGIES

| What to Cache | Cache Key | Invalidation |
|--------------|-----------|--------------|
| `node_modules` | `package-lock.json` hash | On dependency change |
| Build output | Source file hash + deps hash | On code or dep change |
| Docker layers | Dockerfile + context hash | On Dockerfile change |
| Test results | Test file hash + source hash | On test or source change |

## FAILURE FEEDBACK LOOPS

When CI fails, the developer must know within 60 seconds:
- GitHub status checks on the PR
- Slack/email notification with failure summary
- Direct link to the failing step (not just "CI failed")
- Clear error message (not just a log dump)

## ANTI-PATTERNS

- Running tests without caching dependencies (wastes minutes per run)
- Deploying to production on every commit to main without a staging gate
- Secrets stored in environment variables visible in logs (use `::add-mask::`)
- No test stage before deploy
- Manual deployment steps that could be automated

## ANTI-RATIONALIZATION TABLE

| Shortcut | Why It Fails | Do This Instead |
|----------|-------------|-----------------|
| "CI is too slow, I'll merge without waiting" | Broken code reaches main, blocks everyone | Speed up CI (caching, parallelism) instead of skipping it |
| "We'll add tests to CI later" | Code without CI tests accumulates tech debt exponentially | Add the test stage on day one |
| "Manual deploy is fine for now" | Manual deploys are error-prone and unreproducible | Automate from the first deployment |
| "Let's run everything sequentially to keep it simple" | Sequential pipelines are 3-5x slower than parallel | Parallelize independent stages from the start |
