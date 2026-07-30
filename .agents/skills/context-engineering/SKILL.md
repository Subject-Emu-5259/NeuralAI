---
name: "context-engineering"
description: "Context window management for AI agents: rules files, context packing, prompt structuring, and MCP integration."
---
# Context Engineering

Optimize what information an AI agent receives to maximize output quality. Context is the bottleneck - what you include matters more than what you prompt.
## STEP 0: CONFIG

> Project paths and settings are available in your context from `.pandaos/config.yaml`. Use those values. If not present, use the defaults noted in each step.

## STEP 1: AUDIT EXISTING CONTEXT

Inventory what context sources already exist in the project:

### Check for:
1. **CLAUDE.md** - project-level instructions
2. **Rules files** in `{rules_path}` - categorized instructions
3. **Knowledge files** - `knowledge-*.md` prefix for domain context
4. **MCP servers** - `.mcp.json` or tool configurations
5. **Type definitions** - `src/shared/types/` or similar
6. **Existing documentation** - `docs/`, `README.md`, inline comments

### Assess each source:
- Is it up to date? (Check git log for last modification)
- Is it actionable? (Does it tell the agent WHAT to do, or just describe the system?)
- Is it redundant? (Does it repeat information available elsewhere?)
- Is it appropriately scoped? (Project-wide rules vs. feature-specific context)

## STEP 2: CONTEXT CATEGORIES

Organize context into four categories, from highest to lowest priority:

### 1. CONSTRAINTS (Must include)
Rules the agent must never violate. These go in `.claude/rules/` with clear naming:

- `principle-*.md` - architectural principles, coding standards
- `rule-*.md` - specific behavioral rules (file size limits, naming conventions)
- `knowledge-*.md` - domain knowledge the agent needs

Format: Imperative statements. "Never do X", "Always do Y", "When Z, then W".

### 2. PATTERNS (Include when relevant)
Examples of correct implementations the agent should follow:

- Code patterns from the existing codebase (discovered via search, not written from scratch)
- API response formats and data shapes
- File organization conventions
- Error handling patterns specific to this project

Format: Show the pattern, explain WHY it exists, show what the wrong version looks like.

### 3. DOMAIN KNOWLEDGE (Include for complex domains)
Information the agent cannot infer from code alone:

- Business logic explanations
- User workflow descriptions
- External system integration details
- Historical decisions and their rationale (ADRs)

Format: Concise prose. Link to detailed docs rather than inlining everything.

### 4. EPHEMERAL CONTEXT (Include per-task)
Task-specific information that changes frequently:

- Current task requirements ($ARGUMENTS)
- Relevant file contents (read on demand, not pre-loaded)
- Recent changes (git diff, recent commits)
- Runtime state (error logs, test output)

Format: Fetched dynamically. Never hardcoded into rules files.

## STEP 3: CONTEXT PACKING RULES

### What to include:
- Rules that prevent known failure modes (learned from past mistakes)
- Type definitions for code the agent will modify
- Patterns from adjacent code the agent should match
- Boundary conditions and constraints

### What to exclude:
- Entire file contents when only a few lines are relevant
- Historical context that doesn't affect current decisions
- Documentation written for humans (tutorials, guides)
- Code the agent will not modify or interact with
- Redundant information (same rule stated three ways)

### Anti-rationalization check:

| Shortcut | Why it fails |
|----------|-------------|
| "Include everything just in case" | Context window is finite. Noise drowns signal. Agent quality degrades with irrelevant context |
| "The agent will figure it out from the code" | Agents cannot infer business logic, historical decisions, or implicit constraints from code alone |
| "One big CLAUDE.md with all rules" | Large monolithic files are harder to maintain and update. Modular rules are composable |
| "Copy the docs into the rules" | Docs are for humans. Rules are for agents. Different audience, different format |
| "Skip rules, just give good prompts" | Prompts are ephemeral. Rules persist across sessions. Institutional knowledge needs durability |

## STEP 4: RULES FILE STRUCTURE

When creating or updating rules files, follow this structure:

```markdown
# [Category]: [Topic]

## [Section]
[Imperative instructions - what to do, not what to know]

## Anti-Patterns
[Specific things to avoid, with brief rationale]
```

### Naming conventions:
- `principle-{domain}.md` - broad architectural principles (e.g., `principle-code-quality.md`)
- `rule-{specific-concern}.md` - specific behavioral rules (e.g., `rule-fixed-positioning.md`)
- `knowledge-{topic}.md` - domain knowledge (e.g., `knowledge-vercel-apps.md`)

### Rule quality checklist:
1. **Actionable**: The agent knows exactly what to do after reading it
2. **Scoped**: The rule applies to a specific situation, not everything
3. **Justified**: Includes WHY (one sentence) so the agent can generalize
4. **Testable**: You could verify whether the agent followed the rule
5. **Current**: Reflects the actual codebase state, not a past or aspirational state

## STEP 5: MCP CONTEXT INTEGRATION

When MCP servers provide tools, ensure the agent knows:

1. **Tool purpose**: What each tool does (description in the tool schema)
2. **When to use**: Which situations call for which tools
3. **Preference order**: When multiple tools could work, which to prefer (e.g., "prefer pandaos-browser over playwright")
4. **Parameter gotchas**: Known issues with parameter names or formats

Document MCP-specific context in the project's CLAUDE.md under a "Connected Apps" or "Available Tools" section.

## STEP 6: VERIFICATION

After updating context:

1. **Completeness**: Could a new agent session complete common tasks with only this context?
2. **Consistency**: Do any rules contradict each other?
3. **Freshness**: Do all rules reflect the current codebase? (Search for referenced files/patterns)
4. **Size**: Is the total context within budget? (`minimal` < 2KB rules, `medium` < 10KB, `comprehensive` < 25KB)

## STEP 7: OUTPUT

Produce one of:
- **Audit report**: List of existing context, gaps, and recommendations
- **New/updated rules files**: Written to the appropriate paths
- **Context budget plan**: What to include/exclude for a specific task or agent

## BEHAVIORAL RULES

1. Never duplicate information across rules files - reference, don't repeat
2. Rules must be imperative ("Do X", "Never Y"), not descriptive ("The system uses X")
3. Every rule must have a reason. If you cannot explain WHY, the rule may not be needed
4. Context is a budget. Adding context has a cost (displaces other context). Justify additions.
5. Test context by imagining a fresh agent session - would it make the right decisions?
