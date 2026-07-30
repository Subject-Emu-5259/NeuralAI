---
name: data-driven-feature
description: "Build data-driven features: from schema design to API to UI, with analytics and A/B testing considerations."
source: community
allowed-tools: "*"
user-invocable: true
---

# Data-Driven Feature Builder

Build a feature end-to-end from data model to API to UI, with built-in analytics and data quality considerations.

## STEP 1: DEFINE THE FEATURE

Parse $ARGUMENTS for:
- Feature description and user stories
- Data requirements (what data is needed, where it comes from)
- Analytics requirements (what to measure, what success looks like)
- Scale expectations (users, data volume, query patterns)

## STEP 2: DESIGN THE DATA MODEL

Design the schema:

- Tables/collections needed with relationships
- Column types, constraints, and defaults
- Indexes for expected query patterns
- Consider denormalization for read-heavy paths
- Plan for soft deletes vs. hard deletes
- Add audit fields (created_at, updated_at, created_by)

## STEP 3: BUILD THE API LAYER

Implement the data access and API:

- CRUD operations with proper validation
- Query endpoints with filtering, sorting, pagination
- Authorization checks (who can read/write what)
- Rate limiting for public endpoints
- Error handling with meaningful error codes
- API documentation (types, examples)

## STEP 4: IMPLEMENT THE UI

Build the user interface:

- Data fetching with loading and error states
- Optimistic updates for better UX
- Form validation matching API validation
- Empty states and edge case handling
- Responsive design considerations

## STEP 5: ADD ANALYTICS

Instrument the feature for measurement:

- Track key user actions (create, view, update, delete)
- Measure engagement (time on feature, return rate)
- Define success metrics and dashboards
- Add feature flags for gradual rollout
- Plan A/B testing if comparing approaches

## STEP 6: DATA QUALITY

Ensure data integrity:

- Input validation at API boundary
- Database constraints for data integrity
- Data consistency checks for complex operations
- Monitoring for data anomalies
- Backup and recovery plan
