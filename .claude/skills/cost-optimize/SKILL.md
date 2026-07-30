---
name: cost-optimize
description: "Optimize infrastructure and API costs: identify waste, right-size resources, and reduce spending."
source: community
allowed-tools: "*"
user-invocable: true
---

# Cost Optimization

Analyze and reduce infrastructure, API, and operational costs in a project.

## STEP 1: IDENTIFY COST CENTERS

Parse $ARGUMENTS for the area to optimize. Analyze the project for:

- **Cloud resources**: VMs, containers, databases, storage, CDN
- **API costs**: Third-party API calls, LLM token usage, payment processors
- **Build/CI costs**: Build minutes, artifact storage, parallel job usage
- **Data transfer**: Egress charges, cross-region traffic
- **Licensing**: Software licenses, SaaS subscriptions

## STEP 2: ANALYZE USAGE PATTERNS

For each cost center:

- Current usage levels and trends
- Peak vs. average utilization
- Idle or underutilized resources
- Redundant or duplicate services
- Over-provisioned resources

## STEP 3: IDENTIFY SAVINGS

### Infrastructure
- Right-size instances (CPU/memory matched to actual usage)
- Use spot/preemptible instances for fault-tolerant workloads
- Implement auto-scaling to match demand
- Consolidate underutilized services
- Use reserved capacity for predictable workloads
- Clean up orphaned resources (unused volumes, snapshots, IPs)

### Application
- Add caching to reduce API calls and database queries
- Optimize database queries (indexes, query plans)
- Compress assets and responses
- Implement pagination to reduce data transfer
- Use CDN for static assets
- Batch API calls where possible

### LLM/AI Costs
- Use smaller models for simpler tasks
- Cache common responses
- Reduce prompt sizes (trim unnecessary context)
- Implement streaming to reduce perceived latency without bigger models
- Use structured outputs to reduce token waste

### Build/CI
- Cache dependencies and build artifacts
- Skip unnecessary CI steps based on changed files
- Optimize Docker layers for faster builds
- Reduce test parallelism during off-peak

## STEP 4: ESTIMATE SAVINGS

For each recommendation:
- Current estimated cost
- Projected cost after optimization
- Implementation effort
- Risk level

## STEP 5: PRIORITIZE

Order recommendations by ROI (savings / effort). Present as a ranked list with implementation steps.
