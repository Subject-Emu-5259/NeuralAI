---
name: legacy-modernize
description: "Modernize legacy code: assess current state, plan incremental migration, and execute without breaking production."
source: community
allowed-tools: "*"
user-invocable: true
---

# Legacy Modernization

Plan and execute incremental modernization of legacy code without breaking production.

## STEP 1: ASSESS CURRENT STATE

Parse $ARGUMENTS for the legacy code to modernize. Analyze:

- **Age and history**: When was it written? How many contributors?
- **Technology**: Language version, framework version, deprecated APIs in use
- **Dependencies**: Outdated libraries, unmaintained packages
- **Architecture**: Monolith, MVC, spaghetti, well-structured?
- **Test coverage**: What percentage is tested? Are tests reliable?
- **Documentation**: Is the behavior documented anywhere?
- **Risk areas**: What parts are most fragile or least understood?

## STEP 2: DEFINE MODERNIZATION GOALS

Determine what "modern" means for this codebase:

- Language/framework version upgrade
- Architecture refactoring (monolith to services, MVC to clean architecture)
- Dependency replacement (jQuery to vanilla JS, Moment to date-fns)
- Pattern modernization (callbacks to async/await, classes to hooks)
- Build system upgrade
- Adding type safety

## STEP 3: PLAN INCREMENTAL MIGRATION

Design a migration strategy that:

### Keeps Production Working
- Never do a big-bang rewrite (Strangler Fig pattern instead)
- Create adapter layers between old and new code
- Run old and new code in parallel where possible
- Feature-flag new implementations

### Reduces Risk
- Start with the least-coupled, best-understood modules
- Write characterization tests before changing any code
- Migrate one module at a time, validate, then move on
- Keep rollback paths open

### Migration Order
1. Add tests for existing behavior (if missing)
2. Upgrade build tools and dependencies
3. Add type annotations (if adding TypeScript)
4. Migrate utility code (no side effects)
5. Migrate data access layer
6. Migrate business logic
7. Migrate UI/presentation layer
8. Remove legacy code and adapters

## STEP 4: EXECUTE

For each migration step:
- Write characterization tests if they don't exist
- Implement the new version alongside the old
- Verify behavior matches
- Switch over
- Remove the old code

## STEP 5: TRACK PROGRESS

Maintain a migration tracker:
- Modules migrated vs. remaining
- Test coverage improvement
- Dependencies updated
- Known issues or deferred items
