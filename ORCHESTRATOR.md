# ⚙️ NeuralAI Agentic Orchestrator (v7.0 Prototype)

The Orchestrator is the "brain" of the agentic layer. It transforms NeuralAI from a reactive chatbot into a proactive operator capable of decomposing complex goals into executable sub-tasks.

## 🏗️ Orchestration Architecture

### 1. The Manager-Worker Pattern
NeuralAI operates as the **Manager Agent**. For complex, long-horizon, or parallelizable tasks, the Manager spawns **Worker Agents** via the `/zo/ask` API.

- **Manager**: Handles goal decomposition, resource allocation, synthesis of results, and final quality assurance.
- **Worker**: A stateless, task-specific Zo invocation optimized for a single objective (e.g., "Research Topic X", "Audit File Y", "Generate Component Z").

### 2. Task Decomposition Workflow
1.  **Goal Analysis**: The Manager analyzes the user request to determine if it is "Simple" (single turn) or "Complex" (agentic).
2.  **Plan Generation**: If complex, the Manager generates a **Directed Acyclic Graph (DAG)** of tasks.
3.  **Worker Dispatch**: Workers are called in parallel or sequence using the `/zo/ask` API.
4.  **Synthesis**: The Manager aggregates worker outputs, verifies them against the original goal, and presents the result.

## 🛠️ Implementation Tools
- **`/zo/ask` API**: The primary mechanism for spawning Workers.
- **Knowledge Base**: Shared context provided to Workers to ensure alignment.
- **Task Registry**: A log of active and completed sub-tasks to prevent redundant work.

## 🚦 Execution Protocols
- **Parallel Execution**: Use Python `asyncio` or `run_parallel_cmds` to trigger multiple worker calls.
- **Verification Loop**: Every worker output must be validated by the Manager before being integrated into the final response.
- **Fallback**: If a worker fails, the Manager attempts one retry with a refined prompt before reporting a blocker.
