---
name: config-validate
description: "Validate configuration files: check for errors, missing values, security issues, and inconsistencies."
source: community
allowed-tools: "*"
user-invocable: true
---

# Configuration Validator

Validate configuration files across a project for correctness, completeness, security, and consistency.

## STEP 1: DISCOVER CONFIGURATION

Scan the project for configuration files:

- Environment files (.env, .env.local, .env.production)
- App config (config.ts, settings.json, config.yaml)
- Infrastructure (docker-compose.yml, Dockerfile, k8s manifests)
- CI/CD (.github/workflows, .gitlab-ci.yml, Jenkinsfile)
- Build tools (tsconfig.json, vite.config.ts, webpack.config.js)
- Linting/formatting (.eslintrc, .prettierrc, biome.json)
- Package management (package.json, requirements.txt, go.mod)

## STEP 2: VALIDATE SYNTAX

For each config file:
- Parse and verify syntax is valid
- Check for common formatting issues
- Verify JSON/YAML/TOML structure is well-formed

## STEP 3: CHECK COMPLETENESS

Look for missing or incomplete configuration:

- Required environment variables referenced in code but not in .env
- Config values with TODO or placeholder values
- Missing entries compared to .env.example or .env.template
- Default values that should be overridden per environment

## STEP 4: CHECK SECURITY

Flag security issues:

- Secrets or API keys committed in config files
- Overly permissive CORS configuration
- Debug mode enabled in production configs
- Default passwords or tokens
- Sensitive data in non-secret storage
- Insecure connection strings (HTTP instead of HTTPS)

## STEP 5: CHECK CONSISTENCY

Verify config files are consistent with each other:

- Port numbers match across services
- Environment names are consistent
- Version numbers align (Node version in .nvmrc, Dockerfile, CI)
- TypeScript strict mode settings match across configs

## STEP 6: REPORT

Present findings grouped by severity:
- **Errors**: Syntax issues, missing required values
- **Security**: Exposed secrets, insecure defaults
- **Warnings**: Inconsistencies, deprecated options
- **Info**: Optimization opportunities
