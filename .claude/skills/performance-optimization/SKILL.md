---
name: performance-optimization
description: "Performance optimization with measure-first discipline: profiling, Core Web Vitals targets, bundle analysis, and algorithmic efficiency."
user-invocable: true
disable-model-invocation: false
model: sonnet
source: pandaos
allowed-tools: Read, Grep, Glob, Bash
---

# Performance Optimization

A measure-first approach to performance. Never optimize without profiling data. Intuition about bottlenecks is wrong more often than right.

## ABSOLUTE RULE: MEASURE FIRST

Do NOT change code for performance without evidence that it is actually slow. The workflow is always: **Profile -> Identify -> Fix -> Verify**. Premature optimization creates complexity without measurable benefit.

## STEP 1: ESTABLISH BASELINE

Before any optimization, measure current performance:

**For Web/Frontend:**
- Lighthouse scores (Performance, Accessibility, Best Practices)
- Core Web Vitals targets: LCP < 2.5s, INP < 200ms, CLS < 0.1
- Bundle size (`npx vite-bundle-visualizer` or `webpack-bundle-analyzer`)
- Network waterfall (Chrome DevTools Network tab)

**For Backend/Node.js:**
- Response time percentiles (p50, p95, p99)
- Memory usage over time (heap snapshots)
- CPU profiling (Node.js `--prof` or Chrome DevTools)
- Event loop delay

**For Electron:**
- App startup time (time from launch to interactive)
- IPC/bridge server round-trip latency
- Memory footprint (main + renderer processes)

Record these numbers. You will compare against them after optimization.

## STEP 2: IDENTIFY BOTTLENECK

Use profiling data to find the actual bottleneck. Common locations:

| Symptom | Likely Cause | How to Verify |
|---------|-------------|---------------|
| Slow initial load | Large bundle, unoptimized images, render-blocking resources | Bundle analyzer, network waterfall |
| Slow interaction | Heavy JS on main thread, layout thrashing | Performance trace, INP measurement |
| Memory growth | Leaking event listeners, unbounded caches, stale refs | Heap snapshots over time |
| Slow API responses | N+1 queries, missing indexes, unoptimized joins | Query profiling, `EXPLAIN ANALYZE` |

## STEP 3: BUNDLE ANALYSIS (Frontend)

If the bottleneck is bundle size:
1. Run bundle analyzer to identify largest modules
2. Check for: duplicate dependencies, unshaken tree imports, large libraries used for small features
3. Apply fixes: dynamic imports for routes, tree-shaking, replacing heavy libraries with lighter alternatives
4. Target: < 200KB initial JS (gzipped) for web apps

## STEP 4: PROFILING WORKFLOW

**CPU profiling:**
1. Start a profile during the slow operation
2. Identify functions consuming the most self-time (not total time)
3. Focus on the top 3 hottest functions - Pareto principle applies

**Memory profiling:**
1. Take heap snapshot before the operation
2. Perform the operation multiple times
3. Take heap snapshot after - compare retained sizes
4. Look for objects that grow but never shrink

## STEP 5: APPLY OPTIMIZATION

Fix only what profiling identified. Common optimizations by category:

**Algorithmic:** O(n^2) to O(n) via Set/Map, single-pass processing, early exits
**Rendering:** Virtualize long lists, memoize expensive computations, debounce user input
**Network:** Lazy loading, code splitting, prefetching critical resources, caching
**Memory:** Cleanup listeners, bound cache sizes, WeakRef for optional references

## STEP 6: VERIFY IMPROVEMENT

Re-run the same measurements from Step 1. Compare:
- Did the target metric improve by a meaningful amount (>10%)?
- Did any other metric regress?
- Is the optimization worth the added complexity?

If the improvement is < 10% and adds complexity, revert it.

## ANTI-RATIONALIZATION TABLE

| Shortcut | Why It Fails | Do This Instead |
|----------|-------------|-----------------|
| "This looks slow, let me optimize it" | Intuition is wrong 80% of the time | Profile first, optimize what the data shows |
| "Let me memoize everything" | Memoization has overhead (memory + comparison) | Only memoize what profiling shows is expensive |
| "I'll add caching here" | Caching adds invalidation complexity | Prove the computation is repeated AND expensive |
| "This library is smaller" | Smaller doesn't mean faster; API compatibility matters | Benchmark the actual operation, not just bundle size |

## VERIFICATION GATE

| Check | Status |
|-------|--------|
| Baseline measurements recorded before changes | |
| Bottleneck identified via profiling, not intuition | |
| Only profiling-identified issues were optimized | |
| Post-optimization measurements show >10% improvement | |
| No regressions in other metrics | |

---

## Database Performance

When the bottleneck is database-related, apply this methodology before writing any query changes:

**1. Use EXPLAIN ANALYZE, not guesswork**
```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) SELECT ...;
```
Look for: `Seq Scan` on large tables, high `rows` estimates vs actual, `Buffers: shared hit` vs `read` ratios, nested loop joins on unindexed foreign keys.

**2. Index Strategy**
- Check `pg_stat_user_indexes` for unused indexes (they still pay write cost)
- Add indexes on columns used in `WHERE`, `ORDER BY`, `JOIN ON` — in that priority order
- Prefer partial indexes (`WHERE deleted_at IS NULL`) over full-table indexes when the working set is a subset
- Use covering indexes (`INCLUDE (col)`) to eliminate heap fetches on hot read paths

**3. N+1 Detection**
- In ORM code, count the number of SQL queries per request — more than one SELECT inside a loop is always an N+1
- Fix via: eager loading (`include`/`join`), batched queries (`WHERE id = ANY($1)`), or DataLoader pattern for GraphQL
- In raw SQL, replace per-row subqueries with a single JOIN or lateral join

**4. Connection Pooling**
- Use PgBouncer or equivalent in transaction mode for APIs with many short-lived requests
- Never open a new DB connection per request in long-running services — use a connection pool with a max size of `(2 * CPU cores) + disk count`
- Validate pool exhaustion symptoms: queries queuing, p99 latency spikes with normal p50

**5. Caching Invalidation**
- Cache at the query result level only when: (a) the data changes less than it is read, AND (b) the computation or query takes > 10ms
- Define invalidation trigger explicitly before adding any cache — "invalidate on write to table X by user Y"
- Never cache mutable user-specific data without namespacing by user ID

---

## Unity Performance

When optimizing Unity projects, apply these measurement and remediation steps:

**Frame Time Budgets**
- 60 fps = 16.67ms per frame; 30 fps = 33.33ms. Target 50% CPU budget maximum to leave headroom for spikes
- Identify which ms are spent on: scripts, physics, rendering, GC — using the Unity Profiler timeline view

**Unity Profiler (Deep Profile Mode)**
- Enable Deep Profile only on isolated test scenes — it adds 10-100x overhead, not usable in production builds
- Key columns: `Self ms` (cost of the function itself, excluding callees), `GC Alloc` (allocations triggering GC)
- Sort by `GC Alloc` to find allocation hotspots before sorting by `Self ms`
- Profile on target hardware — PC numbers do not represent mobile or console performance

**Memory Profiler**
- Use the Unity Memory Profiler package (not the built-in Profiler) for heap inspection
- Compare two snapshots (before / after a play session) to find retained objects that should have been released
- Watch for: textures loaded but not unloaded, AudioClips held by destroyed GameObjects, ScriptableObject instances growing unboundedly

**GC Allocation Elimination**
- Avoid `new` inside `Update()`, `FixedUpdate()`, `OnGUI()` — any allocation in a hot loop will trigger GC pauses
- Banned patterns in hot paths: LINQ queries, string concatenation, `GetComponent<T>()` without caching, `FindObjectOfType<T>()`
- Use object pooling (`ObjectPool<T>`) for frequently instantiated/destroyed objects (bullets, particles, UI elements)
- Use `Span<T>` and `NativeArray<T>` for temporary buffer operations in performance-critical paths

**DOTS / Job System Migration Criteria**
Migrate a system to DOTS (Entities) or the C# Job System when ALL of the following are true:
1. The system processes > 1000 entities per frame
2. The system is purely data-transforming (no MonoBehaviour lifecycle dependencies)
3. Profiling shows it consuming > 2ms of the frame budget consistently
4. The team has DOTS experience or budget to learn it — do not migrate speculatively

---

## Load Testing

Use k6 for structured load testing. Always follow the multi-stage structure below — do not run a single flat load level.

**Multi-Stage Test Structure**
```javascript
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Warm-up: verify baseline behavior
    { duration: '5m', target: 50 },   // Normal load: expected daily peak
    { duration: '2m', target: 100 },  // Peak load: traffic spike
    { duration: '5m', target: 100 },  // Sustained peak: endurance check
    { duration: '2m', target: 200 },  // Stress: find the breaking point
    { duration: '3m', target: 0 },    // Cool-down: verify recovery
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95th percentile under 500ms
    http_req_failed: ['rate<0.01'],    // Error rate under 1%
  },
};
```

**Threshold Definitions**
- `p(95)` is the primary SLA threshold — not average, not p50
- `http_req_failed` rate > 1% during normal load is a blocking failure
- During stress stage, `p(99)` degradation is expected — the test passes if the system recovers during cool-down

**Report Template**
After every load test run, record:
```markdown
## Load Test Report — [Date] [System]

| Stage         | Target VUs | p50 (ms) | p95 (ms) | Error Rate |
|---------------|-----------|----------|----------|------------|
| Warm-up       | 10        |          |          |            |
| Normal Load   | 50        |          |          |            |
| Peak Load     | 100       |          |          |            |
| Sustained Peak| 100       |          |          |            |
| Stress        | 200       |          |          |            |

**Breaking Point**: [VU count where error rate exceeded 1% or p95 exceeded threshold]
**Recovery**: [Did system return to baseline during cool-down? Y/N]
**Bottleneck Identified**: [DB / App server / Network / Cache]
**Next Action**: [Index to add / Pool size to increase / Service to scale]
```
