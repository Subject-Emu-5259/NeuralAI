---
name: documentation-and-adrs
description: "Architecture Decision Records, API documentation, and inline documentation standards. Document the why, not the what."
user-invocable: true
disable-model-invocation: false
source: pandaos
allowed-tools: Read, Write, Glob, Grep
---

# Documentation & ADRs

Write documentation that answers "why" and "when", not "what". Code shows what it does. Documentation explains why it exists, when to use it, and what to watch out for.
## STEP 0: CONFIG

> Project paths and settings are available in your context from `.pandaos/config.yaml`. Use those values. If not present, use the defaults noted in each step.

## STEP 1: IDENTIFY DOCUMENTATION TYPE

From $ARGUMENTS, determine what documentation is needed:

| Type | When to use | Output |
|------|------------|--------|
| **ADR** | A decision was made between alternatives | Architecture Decision Record |
| **API docs** | A public interface needs documentation for consumers | API reference |
| **Inline docs** | Code needs context that cannot be expressed in code | Code comments, JSDoc |
| **Runbook** | An operational procedure needs to be repeatable | Step-by-step guide |
| **Knowledge file** | Institutional knowledge needs to persist across sessions | `.claude/rules/knowledge-*.md` |

## STEP 2: ARCHITECTURE DECISION RECORDS

When a non-trivial decision was made (framework choice, architecture pattern, tradeoff accepted), write an ADR.

### ADR format:

```markdown
Path: {adrs_path}/{ADR-NNN}-{kebab-case-title}.md

# ADR-NNN: [Decision Title]

## Status
Accepted | Superseded by ADR-NNN | Deprecated

## Date
[YYYY-MM-DD]

## Context
[What is the situation that requires a decision? What constraints exist?
Include enough context that someone reading this in 6 months understands
the problem without prior knowledge.]

## Decision
[What was decided and why. Be specific - name the chosen approach.]

## Alternatives Considered

### Alternative A: [Name]
- Pros: [list]
- Cons: [list]
- Rejected because: [specific reason]

### Alternative B: [Name]
- Pros: [list]
- Cons: [list]
- Rejected because: [specific reason]

## Consequences
- [Positive consequence]
- [Negative consequence / accepted tradeoff]
- [Follow-up work required]

## References
- [Links to relevant code, docs, discussions]
```

### When to write an ADR:
- Choosing between two or more viable approaches
- Accepting a known tradeoff (performance vs. simplicity, etc.)
- Deviating from an established pattern
- Introducing a new dependency or removing one
- Changing an architectural boundary

### When NOT to write an ADR:
- The decision is obvious and has no alternatives
- The decision is easily reversible (a variable name, a CSS color)
- The decision is already documented in a rule file

## STEP 3: API DOCUMENTATION

For public interfaces (exported functions, HTTP endpoints, component props):

### API doc structure:
```typescript
/**
 * [One sentence: what this does and why you'd use it]
 *
 * @param options - [describe the options object]
 * @param options.projectId - [describe this field]
 * @returns [describe the return value and its shape]
 * @throws {NotFoundError} When the project does not exist
 *
 * @example
 * ```ts
 * const project = await getProject({ projectId: 'abc-123' })
 * ```
 *
 * @remarks
 * [Any non-obvious behavior, gotchas, or usage notes]
 */
```

### API documentation rules:
1. **Every exported function** gets a JSDoc comment with at least a one-line description
2. **Parameters and return types** are documented when not self-evident from the type signature
3. **Throws/errors** are always documented - callers need to know what can go wrong
4. **Examples** are included for any function with non-obvious usage
5. **Remarks** capture gotchas, edge cases, and "watch out for" notes

### Anti-rationalization check:

| Shortcut | Why it fails |
|----------|-------------|
| "The types are self-documenting" | Types show shape, not intent. `userId: string` doesn't explain WHERE to get the userId |
| "I'll document it later" | Later never comes. Document at creation time or accept it stays undocumented |
| "Comments get stale" | Stale comments are a maintenance problem, not a reason to avoid documentation. Update comments when you update code |
| "Everyone on the team knows this" | Team members leave. New members join. Knowledge must be persistent, not tribal |
| "The tests document the behavior" | Tests document WHAT, not WHY. They don't explain the design rationale |

## STEP 4: INLINE DOCUMENTATION STANDARDS

### Document the WHY, not the WHAT:

```typescript
// BAD: Increments counter (we can see that)
counter++

// BAD: Check if user is admin (we can read the condition)
if (user.role === 'admin') { ... }

// GOOD: Skip rate limiting for admin users because they run
// batch operations that would otherwise hit the limit constantly
if (user.role === 'admin') { ... }

// GOOD: Use setTimeout(0) to defer this update to the next tick,
// otherwise the store notification fires before React commits
// the current render, causing a "cannot update during render" error
setTimeout(() => store.notify(), 0)
```

### When to add inline comments:
- **Non-obvious "why"**: The code does something that looks wrong but is intentional
- **Workarounds**: Code that exists because of a bug/limitation elsewhere
- **Performance choices**: Why a less readable approach was chosen for speed
- **Business logic**: Rules that come from domain requirements, not technical ones
- **Danger zones**: Code that will break if modified without understanding the constraint

### When NOT to add inline comments:
- The code clearly expresses its intent through naming and structure
- The comment restates what the code does (`// Loop through users`)
- The comment is a section header (`// === HELPER FUNCTIONS ===`)
- The comment explains something that should be a better variable name

## STEP 5: KNOWLEDGE FILES

For institutional knowledge that AI agents need across sessions:

### Knowledge file format:
```markdown
Path: .claude/rules/knowledge-{topic}.md

# [Topic]

## [Section]
[Factual statements about the system]

## Key Decisions
- [Decision]: [Rationale]

## Gotchas
- [Non-obvious behavior that has caused bugs before]
```

### Knowledge file vs. ADR:
- **ADR**: Documents a specific decision at a point in time (historical)
- **Knowledge file**: Documents current state of the system (living document)

An ADR might say "We chose Zustand over Redux because..." (past tense, frozen).
A knowledge file says "The app uses Zustand for state management. Selectors must use stable references..." (present tense, updated).

## STEP 6: DOCUMENTATION AUDIT

When auditing existing documentation:

### Check for:
1. **Accuracy**: Does the documentation match the current code?
2. **Completeness**: Are all public interfaces documented?
3. **Staleness**: When was each doc last updated? Does it reference deprecated code?
4. **Accessibility**: Can a new team member find what they need?
5. **Redundancy**: Is the same information documented in multiple places? (consolidate)

### Red flags:
- Docs that reference files or functions that no longer exist
- Comments with "TODO: update this" older than 3 months
- README files that describe a different architecture than what exists
- API docs with no examples

## STEP 7: VERIFICATION GATES

Before completing documentation work:

- [ ] All ADRs have Status, Date, Context, Decision, Alternatives, and Consequences
- [ ] All exported functions have JSDoc with at least a one-line description
- [ ] No inline comments that merely restate the code
- [ ] All documentation references actual, existing code (no dead links)
- [ ] Knowledge files reflect the CURRENT state of the system
- [ ] Documentation is findable - filed in the expected location with clear naming

## STEP 8: OUTPUT

Produce one of:
- **ADR**: Written to `{adrs_path}/` with proper numbering
- **API docs**: JSDoc comments added to source files
- **Documentation audit**: Report of gaps, staleness, and recommendations
- **Knowledge file**: Written to `.claude/rules/knowledge-*.md`
- **Runbook**: Written to `{docs_path}/runbooks/` or project-appropriate location

## BEHAVIORAL RULES

1. Document the WHY, not the WHAT. Code shows what. Docs explain why.
2. ADRs are immutable once accepted - if a decision changes, write a new ADR that supersedes
3. Every public interface gets at least a one-line description
4. Stale documentation is worse than no documentation - update or remove
5. If you cannot explain a piece of code's purpose in one sentence, the code may need simplification, not documentation
6. Documentation is code. It lives in version control, gets reviewed, and stays current.
