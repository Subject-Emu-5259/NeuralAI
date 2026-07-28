# tools/agent_runner.py
# Phase 8: Agentic Autonomy — plan + execute multi-step tasks across tools, no human in loop.
import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

from tools.tool_handler import ToolHandler

_TASKS_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_tasks.jsonl")


class AgentRunner:
    """Decomposes a goal into tool-executable steps and runs them autonomously."""

    def __init__(self):
        self.handler = ToolHandler()
        self.max_steps = int(os.environ.get("AGENT_MAX_STEPS", "6"))

    def _log(self, task_id: str, event: Dict[str, Any]) -> None:
        row = {"task_id": task_id, "ts": datetime.now(timezone.utc).isoformat(), **event}
        try:
            with open(_TASKS_DB, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass

    def plan(self, goal: str) -> List[Dict[str, str]]:
        """Map a natural-language goal to a step list. Rule-based planner (safe, deterministic)."""
        g = goal.lower()
        steps: List[Dict[str, str]] = []
        if any(k in g for k in ["research", "investigate", "find out", "what is", "latest", "news"]):
            steps.append({"tool": "research", "params": {"query": goal, "top_k": 5}, "reason": "gather sources"})
        if "summarize" in g:
            steps.append({"tool": "summarize", "params": {"text": "{previous}"}, "reason": "condense"})
        if "save" in g or "remember" in g:
            steps.append({"tool": "remember", "params": {"content": goal, "category": "autonomous"}, "reason": "persist to memory"})
        if "translate" in g:
            steps.append({"tool": "translate", "params": {"lang": "en", "text": "{previous}"}, "reason": "translate"})
        if not steps:
            # Default: web search + remember
            steps.append({"tool": "web_search", "params": {"query": goal, "top_k": 5}, "reason": "default search"})
        return steps[: self.max_steps]

    def run(self, goal: str) -> Dict[str, Any]:
        task_id = f"task_{int(time.time())}"
        steps = self.plan(goal)
        self._log(task_id, {"event": "start", "goal": goal, "steps": len(steps)})
        results: List[Dict[str, Any]] = []
        context = ""
        for i, step in enumerate(steps):
            params = dict(step["params"])
            # substitute {previous} with last textual output
            for k, v in params.items():
                if isinstance(v, str) and "{previous}" in v:
                    params[k] = v.replace("{previous}", context[:4000])
            try:
                res = self.handler.execute(step["tool"], params)
                text = res.get("output", "")
                context = text
                results.append({"step": i + 1, "tool": step["tool"], "success": res.get("success", True), "output": text[:2000]})
                self._log(task_id, {"event": "step", "step": i + 1, "tool": step["tool"], "ok": res.get("success", True)})
            except Exception as e:
                results.append({"step": i + 1, "tool": step["tool"], "success": False, "error": str(e)})
                self._log(task_id, {"event": "step_error", "step": i + 1, "tool": step["tool"], "error": str(e)})
        self._log(task_id, {"event": "done", "goal": goal})
        return {
            "success": True,
            "task_id": task_id,
            "goal": goal,
            "steps_run": len(results),
            "results": results,
            "final_context": context[:2000],
            "output": f"🤖 Autonomous task {task_id} complete — {len(results)} step(s) executed.\n{context[:800]}",
        }

    def history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not os.path.exists(_TASKS_DB):
            return []
        rows = []
        with open(_TASKS_DB) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        return rows[-limit:]
