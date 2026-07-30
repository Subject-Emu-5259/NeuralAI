---
name: tech-debt
description: "Audit and prioritize technical debt: identify code smells, outdated patterns, and create a remediation roadmap."
source: community
allowed-tools: "*"
user-invocable: true
---

# Technical Debt Audit

Systematically identify, categorize, and prioritize technical debt in a codebase.

## STEP 1: SCOPE

Parse $ARGUMENTS for:
- Specific areas to audit, or the entire project
- Known pain points to investigate
- Business priorities that affect prioritization

## STEP 2: IDENTIFY DEBT

Scan the codebase for common debt indicators:

### Code Quality Debt
- Large files (>300 lines) and large functions (>50 lines)
- High cyclomatic complexity
- Duplicated code blocks
- Deep nesting (>3 levels)
- Generic naming (data, result, handler, utils)
- Commented-out code
- TODO/FIXME/HACK comments

### Architecture Debt
- Circular dependencies
- God classes/modules that do too many things
- Tight coupling between modules
- Missing abstractions (direct database calls in UI code)
- Inconsistent patterns across similar features
- Dead code paths

### Dependency Debt
- Outdated dependencies (major versions behind)
- Deprecated APIs still in use
- Multiple libraries for the same purpose
- Missing or pinned-to-vulnerable versions

### Testing Debt
- Missing test coverage on critical paths
- Fragile tests that break on unrelated changes
- Slow test suite
- Missing integration or e2e tests

### Infrastructure Debt
- Manual deployment steps
- Missing monitoring or alerting
- Hardcoded configuration
- Missing documentation

## STEP 3: PRIORITIZE

Score each debt item on two axes:

- **Impact**: How much does this slow down development or risk production issues? (1-5)
- **Effort**: How much work to fix? (1-5, where 1 is easy)

Priority = Impact / Effort (highest ratio = best ROI to fix first)

## STEP 4: ROADMAP

Create a remediation plan:

### Quick Wins (Impact >= 3, Effort <= 2)
Fix these immediately. High value, low cost.

### Planned Work (Impact >= 3, Effort >= 3)
Schedule as dedicated sprints or alongside feature work.

### Monitor (Impact <= 2)
Track but don't prioritize. Fix opportunistically.

## STEP 5: REPORT

Present findings with:
- Total debt items by category
- Top 10 highest-priority items with file paths and descriptions
- Estimated effort for the top items
- Suggested sprint allocation (e.g., "20% of sprint capacity for debt reduction")

---

## Unity-Specific Debt Patterns

When auditing Unity projects, extend the standard debt scan with these Unity-specific checks:

### Anti-Pattern Watchlist

| Anti-Pattern | What to Look For | Why It's Debt |
|---|---|---|
| God MonoBehaviour | Any `.cs` file > 500 lines with multiple `private` field groups covering different systems | Impossible to test, breaks SRP, causes merge conflicts |
| `DontDestroyOnLoad` singleton abuse | `DontDestroyOnLoad(gameObject)` + static `Instance` field | Cross-scene coupling, initialization order fragility, impossible to reset for tests |
| `GetComponent` chains | `GetComponent<X>().GetComponent<Y>()` or `GetComponent` called in `Update()` | Performance cost; indicates objects that should be wired via Inspector or ScriptableObject |
| Magic strings | String literals used as tag names, layer names, animator parameter names, or scene names | Silent failures when names change; no refactoring safety |
| Logic in `Update()` that should be event-driven | State polling (`if (health <= 0)` every frame) instead of event subscription | Unnecessary per-frame cost; coupling via shared mutable state |

### ScriptableObject Refactor Path

When a God MonoBehaviour is identified, use this migration sequence:

1. **Extract data** — move all serialized fields that represent game state or configuration into a `ScriptableObject` (`[CreateAssetMenu]`)
2. **Replace direct references** — replace `GetComponent<GameManager>()` calls with Inspector-assigned SO asset references
3. **Add event channels** — replace polling and direct method calls with `GameEvent : ScriptableObject` channels
4. **Split remaining behavior** — decompose the original MonoBehaviour into single-responsibility components (< 150 lines each)
5. **Verify** — see validation criteria below

### Validation Criteria

Before marking a Unity debt item as resolved:

- [ ] The refactored prefab instantiates correctly in an **empty scene** (no scene dependencies)
- [ ] Every MonoBehaviour in the refactored system is **< 150 lines**
- [ ] No `GetComponent` calls exist in `Update()`, `FixedUpdate()`, or `LateUpdate()`
- [ ] No `DontDestroyOnLoad` unless the item is explicitly a cross-scene persistent service (and documented as such)
- [ ] No magic strings — all tag/layer/parameter references use `const` fields or ScriptableObject-based references
