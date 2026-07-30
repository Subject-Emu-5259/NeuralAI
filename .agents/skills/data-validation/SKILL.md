---
name: "data-validation"
description: "Audit and implement data validation: schemas, input sanitization, boundary checks, and type safety across data flows."
---
# Data Validation Audit & Implementation

Audit data validation across a codebase and implement missing validation at system boundaries.

## STEP 1: IDENTIFY BOUNDARIES

Parse $ARGUMENTS for the area to audit. Identify all data entry points:

- API endpoints (request bodies, query params, headers)
- Form inputs and user-facing fields
- File uploads and imports (CSV, JSON, XML)
- Environment variables and configuration
- Database reads (data may be corrupted or stale)
- External API responses
- Message queue payloads
- WebSocket messages

## STEP 2: AUDIT EXISTING VALIDATION

For each boundary, check:

- Is input validated at all?
- Is validation happening at the boundary (not deep in business logic)?
- Are validation errors descriptive and actionable?
- Is validation consistent across similar endpoints?
- Are edge cases handled (empty strings, null, undefined, negative numbers, unicode, very long strings)?
- Is there type coercion that could mask errors?

## STEP 3: DESIGN VALIDATION SCHEMAS

For each gap found, create validation schemas:

- Use the project's preferred validation library (Zod, Yup, Joi, class-validator, etc.)
- Define schemas at the boundary layer
- Include:
  - Type checks
  - Required vs. optional fields
  - String length limits
  - Number ranges
  - Enum/whitelist values
  - Format validation (email, URL, UUID, date)
  - Custom business rules

## STEP 4: IMPLEMENT

Add validation at each identified boundary:

- Parse and validate input before processing
- Return clear error messages with field-level details
- Use consistent error format across all endpoints
- Log validation failures for monitoring

## STEP 5: VERIFY

Confirm validation is working:
- Write tests for valid inputs (should pass)
- Write tests for invalid inputs (should fail with correct errors)
- Test edge cases (empty, null, max length, special characters)
