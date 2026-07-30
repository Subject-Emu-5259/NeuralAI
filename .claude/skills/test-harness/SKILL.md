---
name: test-harness
description: "Build a test harness: set up testing infrastructure, fixtures, mocks, and utilities for a module or project."
source: community
allowed-tools: "*"
user-invocable: true
---

# Test Harness Builder

Set up testing infrastructure for a module or project: framework configuration, fixtures, mocks, utilities, and example tests.

## STEP 1: ANALYZE TESTING NEEDS

Parse $ARGUMENTS for the module or project to set up testing for. Analyze:

- What framework is already in use (or should be used)
- What types of tests are needed (unit, integration, e2e)
- What dependencies need mocking (APIs, databases, file system, time)
- What test data/fixtures are needed

## STEP 2: CONFIGURE FRAMEWORK

Set up or verify the testing framework configuration:

- Test runner config file (vitest.config.ts, jest.config.ts, etc.)
- Test file naming conventions and locations
- Coverage configuration
- Environment setup (jsdom, node, etc.)
- TypeScript support

## STEP 3: CREATE TEST UTILITIES

Build reusable test infrastructure:

### Factories
- Test data factories for creating realistic test objects
- Use builder pattern for flexible object creation
- Default values that produce valid objects

### Mocks
- Mock implementations for external services (API clients, databases)
- Mock utilities for common patterns (timers, fetch, file system)
- Typed mocks that match real interfaces

### Fixtures
- Static test data for deterministic tests
- Setup/teardown helpers for shared state
- Database seeding utilities if applicable

### Custom Matchers
- Domain-specific assertions if needed
- Helper functions that reduce test boilerplate

## STEP 4: WRITE EXAMPLE TESTS

Create 2-3 example tests that demonstrate:
- How to use the factories and mocks
- The project's preferred test structure (describe/it, test groups)
- Both happy path and error case testing
- Async testing patterns if applicable

## STEP 5: VERIFY

Run the test suite and confirm:
- All example tests pass
- Coverage reporting works
- Watch mode functions correctly
