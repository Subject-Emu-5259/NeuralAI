---
name: "api-and-interface-design"
description: "Contract-first API design with Hyrum's Law awareness, error semantics, boundary validation, and versioning strategy."
---
# API & Interface Design

Design APIs and interfaces contract-first. Define the surface area before implementation. Every observable behavior becomes a contract, whether you intended it or not.
## STEP 0: CONFIG

> Project paths and settings are available in your context from `.pandaos/config.yaml`. Use those values. If not present, use the defaults noted in each step.

## STEP 1: IDENTIFY THE INTERFACE TYPE

From $ARGUMENTS, determine what kind of interface is being designed:

| Type | Examples | Key concerns |
|------|----------|-------------|
| **HTTP API** | REST endpoints, webhooks | URL structure, HTTP methods, status codes, pagination |
| **IPC/Bridge** | Electron main-renderer, SSE channels | Serialization cost, event naming, connection lifecycle |
| **Library API** | Exported functions, classes, hooks | Type signatures, return types, error handling |
| **Data Contract** | Database schemas, message formats | Migration strategy, backward compatibility, validation |
| **Internal Interface** | Module boundaries, service contracts | Coupling, testability, change propagation |

## STEP 2: CONTRACT-FIRST DESIGN

Write the contract before any implementation. The contract defines:

### 1. Operations
```typescript
// For each operation, define:
interface Operation {
  name: string           // verb+noun: createProject, listSessions
  input: InputType       // exact shape of what goes in
  output: OutputType     // exact shape of what comes out
  errors: ErrorType[]    // every error that can occur
  sideEffects: string[]  // what changes in the system
  idempotent: boolean    // safe to retry?
}
```

### 2. Type definitions
- Define input and output types as TypeScript interfaces or Zod schemas
- Use Pick/Omit from shared types rather than redefining fields
- Mark optional fields explicitly - nothing is "implicitly optional"
- Use branded types for IDs (`type ProjectId = string & { __brand: 'ProjectId' }`)

### 3. Error semantics
Every API must define its error contract:

```typescript
// Domain errors - the caller did something wrong or hit a known edge case
type DomainError =
  | { code: 'NOT_FOUND'; resource: string; id: string }
  | { code: 'VALIDATION_FAILED'; field: string; reason: string }
  | { code: 'CONFLICT'; message: string }

// System errors - something unexpected happened
type SystemError =
  | { code: 'INTERNAL'; message: string }  // never expose stack traces
  | { code: 'UNAVAILABLE'; retryAfter?: number }
```

Rules:
- Domain errors are part of the contract and must be documented
- System errors are NOT part of the contract - callers should not branch on them
- Never return generic "Something went wrong" - every error must be actionable
- HTTP status codes must match semantics: 400 for bad input, 404 for not found, 409 for conflict, 500 for bugs

## STEP 3: HYRUM'S LAW REVIEW

> "With a sufficient number of users of an API, all observable behaviors of your system will be depended on by somebody."

For each operation, ask:
1. **What behaviors are observable but unintentional?** (Response field ordering, timing, null vs undefined, empty array vs missing field)
2. **What will callers assume that isn't guaranteed?** (Sorting order, field presence, response time)
3. **What can you lock down now?** (Explicit field ordering, documented response shapes, versioned contracts)

### Common Hyrum's Law traps:

| Observable behavior | Risk | Prevention |
|--------------------|------|-----------|
| Field order in JSON | Callers parse by position | Document: "field order is not guaranteed" |
| Error message text | Callers match on error strings | Use error codes, not messages, for branching |
| Response timing | Callers assume synchronous completion | Document async behavior explicitly |
| Null vs undefined vs missing | Callers treat them differently | Pick one convention, enforce it everywhere |
| Array ordering | Callers assume sorted | Document: "results are unordered unless `sort` is specified" |

## STEP 4: BOUNDARY VALIDATION

Define validation at every boundary where data enters the system:

### Input validation rules:
1. Validate at the boundary, not deep inside business logic
2. Use Zod schemas for runtime validation, TypeScript for compile-time
3. Reject early with specific errors - never pass invalid data downstream
4. Sanitize strings: trim whitespace, normalize unicode, check length limits

### Boundary checklist:
- [ ] Maximum input sizes defined (string length, array length, file size)
- [ ] Required vs optional fields explicitly marked
- [ ] Enum values validated against allowed set
- [ ] Nested objects validated recursively
- [ ] Date/time formats specified (ISO 8601)
- [ ] Encoding specified (UTF-8)

## STEP 5: VERSIONING STRATEGY

### The One-Version Rule:
Maintain exactly one version of internal APIs. If you need to change the contract:

1. **Additive changes** (new optional fields, new endpoints): No version bump needed
2. **Breaking changes** (removed fields, changed types, new required fields): Requires migration plan

### For breaking changes:
```
Phase 1: Add new field/endpoint alongside old one (both work)
Phase 2: Migrate all callers to new version
Phase 3: Remove old field/endpoint
Phase 4: Clean up migration code
```

Never skip Phase 2. Never leave Phase 1 indefinitely.

### Anti-rationalization check:

| Shortcut | Why it fails |
|----------|-------------|
| "We'll version later if needed" | By then, callers depend on unversioned behaviors (Hyrum's Law) |
| "It's an internal API, we control all callers" | Internal callers still break. And internal APIs become external over time |
| "Just change it and fix what breaks" | You won't find all callers. Some breakage is silent |
| "Add a v2 endpoint and deprecate v1" | v1 never gets removed. Now you maintain two forever |
| "Use feature flags instead of versioning" | Feature flags are for features, not API contracts |

## STEP 6: WRITE THE API SPEC

Output the complete API specification:

```markdown
# API: [Name]

## Overview
[One paragraph: what this API does and who uses it]

## Operations

### [operationName]
- **Method**: GET/POST/PUT/DELETE (for HTTP) or event name (for IPC)
- **Input**: [TypeScript type or Zod schema]
- **Output**: [TypeScript type]
- **Errors**: [List of domain errors]
- **Idempotent**: yes/no
- **Side effects**: [what changes]

## Types
[All shared type definitions]

## Error Codes
[Complete error code table with meanings and caller actions]

## Versioning
[Current version, change policy, migration procedures]

## Boundaries
[Validation rules, size limits, rate limits]
```

## STEP 7: VERIFICATION GATES

Before considering the API design complete:

- [ ] Every operation has defined input, output, and error types
- [ ] No operation returns untyped data (no `any`, no `unknown` without narrowing)
- [ ] Error codes are unique and documented
- [ ] Boundary validation covers all input fields
- [ ] Breaking change strategy is defined
- [ ] Hyrum's Law review completed for each operation
- [ ] The spec is readable by someone who will implement callers, not just the server

## BEHAVIORAL RULES

1. Define the contract before writing implementation code
2. Every observable behavior is a contract - make it intentional or hide it
3. Errors are first-class API design, not afterthoughts
4. Never expose internal implementation details through the API surface
5. When in doubt, make it stricter now - loosening is additive, tightening is breaking

---

## Backend Service Architecture

When designing APIs that span multiple services, apply the following decomposition specification for each service:

```markdown
# Service Decomposition Specification

## [Service Name]
**Architecture Pattern**: [Microservices / Monolith / Serverless / Hybrid]
**Communication Pattern**: [REST / GraphQL / gRPC / Event-driven]
**Data Pattern**: [CQRS / Event Sourcing / Traditional CRUD]

### Per-Service Decisions
- **Database**: [e.g., PostgreSQL with user data encryption / read replicas]
- **Cache**: [e.g., Redis for frequently accessed records — specify TTL and invalidation trigger]
- **Queue**: [e.g., RabbitMQ for async pipeline — specify exchange/routing-key conventions]

### Service Boundary Rules
- Each service owns exactly one bounded context
- Services communicate via published events or explicit API contracts — never via shared DB tables
- Failure in one service must not cascade synchronously to unrelated services (circuit breakers required on call paths)
```

### Database Schema Design Patterns

Apply these patterns to every schema produced under this skill:

**Primary Keys**
- Use UUID (`gen_random_uuid()`) not serial integers — IDs must be safe to expose externally without leaking row counts

**Soft Delete**
- Add `deleted_at TIMESTAMP WITH TIME ZONE NULL` to any entity that must support audit history or recovery
- All queries filtering "active" records must include `WHERE deleted_at IS NULL`

**Index Strategy**
- Partial indexes for filtered queries: `CREATE INDEX idx_name ON table(col) WHERE deleted_at IS NULL`
- Covering indexes when a query selects only indexed columns — avoids heap fetches
- GIN indexes for full-text search: `CREATE INDEX idx_search ON table USING gin(to_tsvector('english', col))`
- Never index low-cardinality columns (boolean, enum with few values) in isolation

**Schema Backwards Compatibility**
- Additive changes (new nullable column, new table) are non-breaking — no migration window required
- Removing a column or changing a type requires a three-phase migration: (1) add new column, (2) dual-write + backfill, (3) drop old column
- Never rename a column directly — treat it as remove + add with a migration window

```sql
-- Example: entity table with all patterns applied
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    total_amount DECIMAL(12,2) NOT NULL CHECK (total_amount >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Partial index for active records only
CREATE INDEX idx_orders_user_active ON orders(user_id) WHERE deleted_at IS NULL;
-- Covering index for status-based listing (avoids heap fetch)
CREATE INDEX idx_orders_status_created ON orders(status, created_at DESC) WHERE deleted_at IS NULL;
```
