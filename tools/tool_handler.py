# tools/tool_handler.py
#
# Central handler for all tool execution
# - Routes tool calls to appropriate tool class
# - Returns formatted results for chat display

import sys
import os
import logging
import re
from typing import Dict, Any, Optional
import time
from datetime import datetime, timezone, timedelta

# Local DB handle mirroring services/webui_service.DATABASE (no cross-module import to avoid circular deps)
_IA_DATABASE = "/home/workspace/Projects/NeuralAI/data/neuralai.db"


def _domain_of(url: str) -> str:
    """Return a short readable host (e.g. bbc.com) from a URL."""
    try:
        from urllib.parse import urlparse
        net = urlparse(url).netloc
        return net[4:] if net.startswith("www.") else (net or "source")
    except Exception:
        return "source"


def get_db():
    import sqlite3
    from pathlib import Path
    Path("/home/workspace/Projects/NeuralAI/data").mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(_IA_DATABASE)

logger = logging.getLogger("NeuralAI.ToolHandler")
import logging
logger = logging.getLogger(__name__)

# Add tools to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.code_sandbox import CodeSandbox, execute
from tools.file_manager import FileManager
from tools.web_fetcher import WebFetcher
from tools.web_browser import get_session, close_session
from tools.web_search import WebSearch
from tools.db_connector import DatabaseConnector
from tools.summarize import summarize_sources
from tools.git_assistant import GitAssistant
from tools.image_generator import image_generator
from tools.media_generator import (
    generate_text,
    generate_video,
    generate_audio,
    generate_embeddings,
    realtime_voice_url,
)


class MediaGenerator:
    """Thin wrapper restoring the legacy class API over the refactored
    module-level media functions in tools.media_generator."""

    def text(self, prompt: str, model: str = "openai") -> dict:
        return generate_text(prompt)

    def video(self, prompt: str) -> dict:
        res = generate_video(prompt)
        if res.get("success") and "video_url" in res:
            res["url"] = res["video_url"]
        return res

    def audio(self, text: str, voice: str = "alloy") -> dict:
        res = generate_audio(text, voice)
        if res.get("success") and "audio_url" in res:
            res["url"] = res["audio_url"]
        return res

    def realtime_voice(self) -> dict:
        url = realtime_voice_url()
        return {"success": True, "url": url, "model": "pollinations-realtime"}

    def embed(self, text: str, model: str = "openai-3-small") -> dict:
        return generate_embeddings(text, model)


media = MediaGenerator()
from tools.voice_transcriber import voice_transcriber
from tools.knowledge_graph import save_memory, search_memory, get_graph, recall, extract_and_store
from tools.agentic import run as agentic_run

# --- reminder helpers (used by /remind) ---
def _parse_remind_time(text: str) -> Optional[float]:
    """Parse a natural-ish time into a unix epoch (UTC). Returns None if unparseable."""
    import re as _re
    low = text.lower().strip()
    now = time.time()
    m = _re.match(r"^(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$", low)
    if m:
        n = int(m.group(1)); unit = m.group(2)[0]
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return now + n * mult
    # also accept "<number> <word>" like "30 seconds" / "in 30 seconds"
    mw = _re.match(r"(?:in\s+)?(\d+)\s*(second|minute|hour|day|sec|min|hr)s?\b", low)
    if mw:
        n = int(mw.group(1)); unit = mw.group(2)[0]
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return now + n * mult
    mt = _re.match(r"tomorrow\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", low)
    if mt:
        hh = int(mt.group(1)); mm = int(mt.group(2) or 0)
        if mt.group(3) == "pm" and hh < 12: hh += 12
        if mt.group(3) == "am" and hh == 12: hh = 0
        t = (datetime.now() + timedelta(days=1)).replace(hour=hh, minute=mm, second=0, microsecond=0)
        return t.timestamp()
    ma = _re.match(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})", text.strip())
    if ma:
        try:
            return datetime.strptime(f"{ma.group(1)} {ma.group(2)}:{ma.group(3)}", "%Y-%m-%d %H:%M").timestamp()
        except Exception:
            return None
    return None


def _remind_msg(text: str) -> str:
    """Extract the message portion after a leading time expression."""
    import re as _re
    cleaned = _re.sub(r"^\s*(\d+\s*(s|m|h|d|sec|min|hr|hour|day|mins|hrs|hours|days|seconds)\b)", "", text, flags=_re.I).strip()
    cleaned = _re.sub(r"^\s*tomorrow\s+\d{1,2}(?::\d{2})?\s*(am|pm)?\b", "", cleaned, flags=_re.I).strip()
    cleaned = _re.sub(r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}\b", "", cleaned).strip()
    cleaned = _re.sub(r"^\s*in\s+\d+\s*(second|minute|hour|day|sec|min|hr)s?\b", "", cleaned, flags=_re.I).strip()
    return cleaned or text



try:
    from tools.refine import refine_text as _refine_text
except Exception:
    _refine_text = None

# Tools whose output should be passed through refine_text inside execute()
_REFINE_KINDS = {
    "news": "news",
    "web_search": "web_search",
    "web_fetcher": "web_fetcher",
    "research": "research",
}


class ToolHandler:
    """Central handler for tool execution."""

    def __init__(self, workspace: str = "/home/workspace"):
        self.workspace = workspace
        self.code_sandbox = CodeSandbox()
        self.file_manager = FileManager(base_dir=workspace)
        self.web_fetcher = WebFetcher()
        self.web_search = WebSearch()
        self.db_connector = DatabaseConnector()
        self.git_assistant = GitAssistant(repo_path=workspace)
        self.media = media
        self.knowledge_graph = True  # module-level functions, no instance state needed

    def execute(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool: Tool name (terminal, code_exec, file_manager, web_fetcher, database, git)
            params: Tool-specific parameters
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "data": dict (tool-specific)
            }
        """
        handlers = {
            "terminal": self._handle_terminal,
            "code_exec": self._handle_code_exec,
            "file_manager": self._handle_file_manager,
            "web_fetcher": self._handle_web_fetcher,
            "web_browser": self._handle_web_browser,
            "web_search": self._handle_web_search,
            "research": self._handle_research,
            "database": self._handle_database,
            "git": self._handle_git,            "image": self._handle_image,
            "speak": self._handle_speak,
            "summarize": self._handle_summarize,
            "translate": self._handle_translate,
            "news": self._handle_news,
            "youtube": self._handle_youtube,
            "text": self._handle_text,
            "video": self._handle_video,
            "audio": self._handle_audio,
            "voice": self._handle_voice,
            "embed": self._handle_embed,            "remember": self._handle_remember,
            "recall": self._handle_recall,
            "graph": self._handle_graph,
            "agent": self._handle_agent,
            "autosave": self._handle_autosave,
            "calc": self._handle_calc,
            "doc": self._handle_doc,
            "vision": self._handle_vision,
            "remind": self._handle_remind,
        }
        handler = handlers.get(tool)
        if handler is None:
            return {"success": False, "output": "", "error": f"Unknown tool: {tool}", "data": {}}
        try:
            result = handler(params)
        except Exception as e:
            logger.error(f"[tool_handler] {tool} failed: {e}")
            return {"success": False, "output": "", "error": str(e), "data": {}}
        # Refine display output for web tools so /api/tool returns clean markdown
        # regardless of whether the call came from the NL router or a slash command.
        if _refine_text is not None and tool in _REFINE_KINDS and isinstance(result, dict) and result.get("success"):
            raw = result.get("output", "")
            if raw:
                try:
                    result["output"] = _refine_text(raw, _REFINE_KINDS[tool])
                except Exception:
                    pass
        return result

        
    def _handle_remember(self, params: Dict[str, Any]) -> Dict[str, Any]:
        content = params.get("content", "")
        if not content and isinstance(params.get("text"), str):
            content = params["text"]
        if not content:
            return {"success": False, "error": "content required", "output": "", "data": {}}
        res = save_memory(content, tags=params.get("tags", ""),
                          relation_to=params.get("relation_to", ""),
                          relation=params.get("relation", ""))
        return {"success": res.get("success", False),
                "output": f"🧠 Remembered (id={res.get('id','?')}, remote={res.get('remote', False)})",
                "error": "", "data": res}

    def _handle_recall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", params.get("q", ""))
        if not query:
            return {"success": False, "error": "query required", "output": "", "data": {}}
        out = recall(query, limit=int(params.get("limit", 5)))
        return {"success": True, "output": out, "error": "", "data": {}}

    def _handle_graph(self, params: Dict[str, Any]) -> Dict[str, Any]:
        node_id = params.get("id", "")
        if not node_id:
            return {"success": False, "error": "id required", "output": "", "data": {}}
        g = get_graph(node_id)
        return {"success": True, "output": json.dumps(g), "error": "", "data": g}

    def _handle_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task = params.get("task", params.get("query", ""))
        if not task:
            return {"success": False, "error": "task required", "output": "", "data": {}}
        res = agentic_run(task, tool_handler=self)
        return {"success": res.get("success", False),
                "output": f"🤖 Agent completed {res.get('steps',0)} steps:\n\n{res.get('brief','')}",
                "error": "", "data": res}

    def _handle_autosave(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text", params.get("content", ""))
        if not text:
            return {"success": False, "error": "text required", "output": "", "data": {}}
        res = extract_and_store(text)
        return {"success": res.get("success", False),
                "output": f"🧠 Conversation ingested for auto-memory (remote={res.get('remote', False)})",
                "error": "", "data": res}
        handler = handlers.get(tool)
        if not handler:
            return {
                "success": False,
                "output": "",
                "error": f"Unknown tool: {tool}",
                "data": {}
            }
        
        try:
            return handler(params)
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Tool execution error: {str(e)}",
                "data": {}
            }

    def _handle_terminal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle terminal/shell commands."""
        command = params.get("command", "")
        if not command:
            return {
                "success": False,
                "output": "",
                "error": "No command provided",
                "data": {}
            }
        
        result = self.code_sandbox.run_bash(command)
        
        output = result["output"]
        if result["error"]:
            output += f"\n[stderr]\n{result['error']}"
        
        return {
            "success": result["success"],
            "output": output,
            "error": "",
            "data": {
                "exit_code": result["exit_code"],
                "execution_time": result["execution_time"]
            }
        }

    def _handle_code_exec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code execution requests."""
        code = params.get("code", "")
        language = params.get("language", "python")
        
        if not code:
            return {
                "success": False,
                "output": "",
                "error": "No code provided",
                "data": {}
            }
        
        # If code doesn't look like actual code, it might be a message
        # asking to run something mentioned elsewhere
        if not any(kw in code for kw in ["def ", "function ", "print(", "console.log", "import "]):
            return {
                "success": False,
                "output": "",
                "error": "No executable code detected. Provide code to run.",
                "data": {}
            }
        
        result = execute(code, language=language)
        
        output = result["output"]
        if result["error"]:
            output += f"\n[error]\n{result['error']}"
        
        return {
            "success": result["success"],
            "output": output,
            "error": "",
            "data": {
                "exit_code": result["exit_code"],
                "execution_time": result["execution_time"],
                "language": language
            }
        }

    def _handle_file_manager(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file operations."""
        query = params.get("query", "").lower()
        
        # Determine operation from query
        if "list" in query or "show files" in query or "what files" in query:
            path = params.get("path", ".")
            result = self.file_manager.list_dir(path)
            
            if result["success"]:
                output = f"Directory: {result['path']}\n\n"
                output += f"Directories ({result['total_dirs']}):\n"
                for d in result["directories"]:
                    output += f"  📁 {d['name']}\n"
                output += f"\nFiles ({result['total_files']}):\n"
                for f in result["files"]:
                    size = self._format_size(f["size"])
                    output += f"  📄 {f['name']} ({size})\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Failed to list directory"),
                "data": result
            }
        
        elif "read" in query:
            # Extract path from query
            path = params.get("path", "")
            if not path:
                # Try to extract from query
                import re
                match = re.search(r"read (?:file )?['\"]?([^\s'\"]+)['\"]?", query)
                if match:
                    path = match.group(1)
            
            result = self.file_manager.read_file(path)
            
            if result["success"]:
                output = f"File: {result['path']}\n"
                output += f"Size: {result['size']} bytes | Lines: {result['lines']}\n\n"
                output += f"```\n{result['content']}\n```"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Failed to read file"),
                "data": result
            }
        
        elif "search" in query or "find" in query:
            # Extract search pattern
            pattern = params.get("pattern", "")
            if not pattern:
                # Try to extract from query
                import re
                match = re.search(r"(?:search|find) (?:for )?['\"]?([^\s'\"]+)['\"]?", query)
                if match:
                    pattern = match.group(1)
                else:
                    pattern = query.split()[-1]  # Use last word as fallback
            
            search_content = "content" in query or "in files" in query
            result = self.file_manager.search(pattern, search_content=search_content)
            
            if result["success"]:
                output = f"Search for '{pattern}': {result['total']} results\n\n"
                for r in result["results"][:20]:
                    if r.get("line"):
                        output += f"📄 {r['path']}:{r['line']}\n  {r['match']}\n\n"
                    else:
                        output += f"📄 {r['path']}\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Search failed"),
                "data": result
            }
        
        # Default: list current directory
        result = self.file_manager.list_dir()
        
        if result["success"]:
            output = f"Directory: {result['path']}\n\n"
            output += f"Directories ({result['total_dirs']}):\n"
            for d in result["directories"][:10]:
                output += f"  📁 {d['name']}\n"
            output += f"\nFiles ({result['total_files']}):\n"
            for f in result["files"][:10]:
                size = self._format_size(f["size"])
                output += f"  📄 {f['name']} ({size})\n"
            
            return {
                "success": True,
                "output": output,
                "error": "",
                "data": result
            }
        return {
            "success": False,
            "output": "",
            "error": result.get("error", "Failed to list directory"),
            "data": result
        }

    def _handle_web_fetcher(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web fetching operations."""
        url = params.get("url", "")
        
        if not url:
            return {
                "success": False,
                "output": "",
                "error": "No URL provided",
                "data": {}
            }
        
        # Fetch and parse the URL
        result = self.web_fetcher.fetch(url)
        
        if result["success"]:
            output = f"URL: {result['url']}\n"
            output += f"Title: {result['title']}\n"
            output += f"Status: {result['status']}\n\n"
            output += f"Content Preview:\n{result['text'][:1500]}...\n\n"
            output += f"Links: {len(result['links'])} found\n"
            output += f"Images: {len(result['images'])} found"
            
            return {
                "success": True,
                "output": output,
                "error": "",
                "data": result
            }
        return {
            "success": False,
            "output": "",
            "error": result.get("error", "Failed to fetch URL"),
            "data": result
        }

    def _handle_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database operations."""
        query = params.get("query", "").lower()
        sql = params.get("sql", "")
        
        # Default to NeuralAI's database if not connected
        if not self.db_connector.active_db:
            db_path = os.path.join(self.workspace, "Projects/NeuralAI/from-scratch/web_ui/neuralai.db")
            if os.path.exists(db_path):
                self.db_connector.connect_sqlite(db_path, "neuralai")
            else:
                # Create in-memory DB for testing
                self.db_connector.connect_sqlite(":memory:", "memory")
        
        if "show tables" in query or "list tables" in query:
            result = self.db_connector.tables()
            
            if result["success"]:
                output = f"Tables ({result['count']}):\n"
                for t in result["tables"]:
                    output += f"  📊 {t}\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Failed to list tables"),
                "data": result
            }
        
        elif "schema" in query:
            result = self.db_connector.schema()
            
            if result["success"]:
                output = "Database Schema:\n\n"
                for table in result["tables"]:
                    output += f"📊 {table['name']}:\n"
                    for col in table["columns"]:
                        pk = " 🔑" if col["primary_key"] else ""
                        output += f"  - {col['name']}: {col['type']}{pk}\n"
                    output += "\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Failed to get schema"),
                "data": result
            }
        
        elif sql:
            result = self.db_connector.query(sql)
            
            if result["success"]:
                output = f"Query: {sql}\n\n"
                if result["rows"]:
                    output += f"Results ({result['row_count']} rows):\n"
                    # Format as table
                    if result["columns"]:
                        output += "| " + " | ".join(result["columns"]) + " |\n"
                        output += "|" + "|".join(["---" for _ in result["columns"]]) + "|\n"
                    for row in result["rows"][:20]:
                        values = [str(v) for v in row.values()]
                        output += "| " + " | ".join(values) + " |\n"
                else:
                    output += f"Affected {result['row_count']} rows"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Query failed"),
                "data": result
            }
        
        # Default: show tables
        result = self.db_connector.tables()
        output = f"Connected to: {self.db_connector.active_db}\n\n"
        output += f"Tables ({result.get('count', 0)}):\n"
        for t in result.get("tables", []):
            output += f"  📊 {t}\n"
        
        return {
            "success": True,
            "output": output,
            "error": "",
            "data": result
        }

    def _handle_git(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle git operations."""
        action = params.get("action", "status").lower()
        
        # Check if in a git repo
        if not self.git_assistant.is_repo()["is_repo"]:
            return {
                "success": False,
                "output": "",
                "error": "Not a git repository",
                "data": {}
            }
        
        if "status" in action:
            result = self.git_assistant.status()
            
            if result["success"]:
                output = f"Branch: {result['branch']}\n"
                output += f"Ahead: {result['ahead']} | Behind: {result['behind']}\n\n"
                
                if result["staged"]:
                    output += "Staged:\n"
                    for f in result["staged"]:
                        output += f"  ✅ {f}\n"
                
                if result["modified"]:
                    output += "Modified:\n"
                    for f in result["modified"]:
                        output += f"  📝 {f}\n"
                
                if result["untracked"]:
                    output += "Untracked:\n"
                    for f in result["untracked"]:
                        output += f"  ❓ {f}\n"
                
                if not any([result["staged"], result["modified"], result["untracked"]]):
                    output += "Working directory clean ✨"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        elif "log" in action:
            result = self.git_assistant.log(count=10)
            
            if result["success"]:
                output = f"Recent commits ({result['count']}):\n\n"
                for c in result["commits"]:
                    output += f"📝 {c['hash']} - {c['message']}\n"
                    output += f"   {c['author']} • {c['date']}\n\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        elif "branch" in action:
            result = self.git_assistant.branch(list_all=True)
            
            if result["success"]:
                output = f"Current: {result['current']}\n\n"
                output += "Branches:\n"
                for b in result["branches"]:
                    marker = "→ " if b == result["current"] else "  "
                    output += f"{marker}{b}\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        elif "diff" in action:
            result = self.git_assistant.diff()
            
            if result["success"]:
                output = "Git Diff:\n\n"
                output += f"```diff\n{result['diff']}\n```"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        elif "remote" in action:
            result = self.git_assistant.remote()
            
            if result["success"]:
                output = "Remotes:\n"
                for name, url in result["remotes"].items():
                    output += f"  {name}: {url}\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        # Default: show status
        result = self.git_assistant.status()
        
        if result["success"]:
            output = f"Branch: {result['branch']}\n"
            output += f"Ahead: {result['ahead']} | Behind: {result['behind']}\n\n"
            
            if result["staged"]:
                output += "Staged:\n"
                for f in result["staged"]:
                    output += f"  ✅ {f}\n"
            
            if result["modified"]:
                output += "Modified:\n"
                for f in result["modified"]:
                    output += f"  📝 {f}\n"
            
            if result["untracked"]:
                output += "Untracked:\n"
                for f in result["untracked"]:
                    output += f"  ❓ {f}\n"
            
            if not any([result["staged"], result["modified"], result["untracked"]]):
                output += "Working directory clean ✨"
            
            return {
                "success": True,
                "output": output,
                "error": "",
                "data": result
            }
        
        return {
            "success": False,
            "output": "",
            "error": "Git operation failed",
            "data": {}
        }


    def _handle_web_browser(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Agentic browsing with Playwright (real web surfing)."""
        url = params.get("url", "")
        steps = params.get("steps", [])
        session_id = params.get("session_id", "default")
        if not url:
            return {"success": False, "output": "", "error": "No URL provided", "data": {}}
        try:
            sess = get_session(session_id)
            result = sess.run(url, steps or [])
            if params.get("close_session"):
                close_session(session_id)
            return {"success": True, "output": result, "error": "", "data": {}}
        except Exception as e:
            return {"success": False, "output": "", "error": f"Browse error: {e}", "data": {}}

    def _handle_web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Web search via provider or built-in fetcher: /web <query>."""
        query = (params.get("query") or "").strip()
        if not query:
            return {"success": False, "output": "", "error": "No query provided", "data": {}}
        try:
            results = self.web_search.search(query, top_k=int(params.get("top_k", 5)))
            if isinstance(results, str):
                return {"success": True, "output": results, "error": "", "data": {}}
            if not results:
                return {"success": False, "output": "", "error": f"No results for '{query}'", "data": {}}
            out = ["**Search results**\n"]
            seen = set()
            for i, r in enumerate(results, 1):
                url = r.get("url", "")
                if url in seen:
                    continue
                seen.add(url)
                title = r.get("title", "").strip()
                snippet = (r.get("snippet", "") or "").strip()
                src = r.get("source", "") or ""
                line = f"{i}. [{title}]({url})" if url else f"{i}. {title}"
                if src and src != title:
                    line += f" — {src}"
                if snippet and snippet != src:
                    line += f"\n   {snippet[:220]}"
                out.append(line)
            return {"success": True, "output": "\n\n".join(out), "error": "", "data": {"results": results}}
        except Exception as e:
            return {"success": False, "output": "", "error": f"Search error: {e}", "data": {}}


    def _handle_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """AI image generation: /img <prompt>."""
        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            return {"success": False, "error": "Usage: /img <prompt>"}
        try:
            res = image_generator.generate(prompt)
            if not isinstance(res, dict) or not res.get("success", False):
                err = res.get("error", "Image generation failed") if isinstance(res, dict) else str(res)
                return {"success": False, "error": err, "prompt": prompt}
            url = res.get("image_url") or res.get("url")
            path = res.get("image_path") or res.get("path")
            summary = f"Image generated: {url}" if url else f"Image saved to {path}"
            return {
                "success": True,
                "output": summary,
                "url": url,
                "path": path,
                "prompt": prompt,
            }
        except Exception as e:
            return {"success": False, "error": f"Image generation failed: {e}"}
    def _handle_speak(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Text-to-speech: /speak <text>."""
        text = (params.get("text") or "").strip()
        if not text:
            return {"success": False, "error": "Usage: /speak <text>"}
        try:
            from tools.tts import text_to_speech
            audio_url = text_to_speech(text)
            return {"success": True, "audio_url": audio_url, "text": text}
        except Exception as e:
            return {"success": False, "error": f"TTS failed: {e}"}

    def _handle_summarize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize a URL or raw text: /summarize <url|text>."""
        text = (params.get("text") or "").strip()
        if not text:
            return {"success": False, "error": "Usage: /summarize <url or text>"}
        try:
            if text.startswith("http://") or text.startswith("https://"):
                fetched = WebFetcher().fetch(text)
                content = fetched.get("text") or fetched.get("content") or ""
                src_label = text
            else:
                content = text
                src_label = "provided text"
            brief = summarize_sources([{"title": src_label, "text": content}], query="summary")
            return {"success": True, "summary": brief, "source": src_label}
        except Exception as e:
            return {"success": False, "error": f"Summarize failed: {e}"}

    def _handle_translate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Translate text: /translate <lang> <text> (default lang=es)."""
        text = (params.get("text") or "").strip()
        target = (params.get("target") or "es").strip()
        if not text:
            return {"success": False, "error": "Usage: /translate <target_lang> <text>"}
        try:
            from tools.translate import translate_text
            out = translate_text(text, target)
            return {"success": True, "translation": out, "target": target, "source": text}
        except Exception as e:
            return {"success": False, "error": f"Translate failed: {e}"}
    def _handle_news(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """News search: /news <topic>."""
        topic = (params.get("query") or params.get("topic") or "").strip()
        if not topic:
            return {"success": False, "output": "", "error": "No topic provided", "data": {}}
        try:
            t = re.sub(r"^(the\s+)?(latest|recent|current|today'?s?|newest|top|breaking|headlines?)\s+", "", topic, flags=re.I).strip()
            if not t or t.lower() in ("news", "headlines", "latest"):
                t = "top news today"
            q = t if t.lower().endswith("news") else t + " news"
            results = self.web_search.search(q, top_k=int(params.get("top_k", 6)))
            if isinstance(results, str):
                return {"success": True, "output": results, "error": "", "data": {}}
            if not results:
                return {"success": False, "output": "", "error": f"No news for '{topic}'", "data": {}}
            out = ["**Latest headlines**\n"]
            seen = set()
            for i, r in enumerate(results, 1):
                url = r.get("url", "")
                if url in seen:
                    continue
                seen.add(url)
                title = r.get("title", "").strip()
                src = r.get("source", "") or ""
                snippet = (r.get("snippet", "") or "").strip()
                line = f"{i}. **{title}**"
                if src:
                    line += f"\n   {src}"
                if snippet and snippet != src:
                    line += f"\n   {snippet}"
                # Emit a clickable link. If the resolver returned a real publisher
                # URL, use it directly. If it's still a Google News tracking blob,
                # strip the ?oc= tracking param so the link is short + stable
                # (clickable to Google's reader) instead of a 300-char blob.
                if url:
                    if "news.google.com" in url:
                        clean = url.split("?")[0]
                        label = (src + " · Read on Google News") if src else "Read on Google News"
                    else:
                        clean = url
                        label = src if src else _domain_of(url)
                    line += f"\n   🔗 [{label}]({clean})"
                out.append(line)
            return {"success": True, "output": "\n\n".join(out), "error": "", "data": {"results": results}}
        except Exception as e:
            return {"success": False, "output": "", "error": f"News error: {e}", "data": {}}

    def _handle_youtube(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """YouTube transcript + summary: /yt <url>."""
        url = (params.get("url") or params.get("query") or "").strip()
        if not url or "youtube.com" not in url and "youtu.be" not in url:
            return {"success": False, "output": "", "error": "Provide a YouTube URL", "data": {}}
        try:
            import urllib.request, json, re
            api = "https://noembed.com/embed?url=" + urllib.parse.quote(url)
            req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                meta = json.loads(r.read().decode("utf-8", errors="ignore"))
            title = meta.get("title", url)
            author = meta.get("author_name", "")
            # Use summary chain on the video page description if available
            summary = summarize_sources([{"title": title, "url": url, "text": meta.get("title", "")}], query=title, max_sentences=4)
            out = f"▶️ YouTube: {title}\n👤 {author}\n\n{summary}\n\nWatch: {url}"
            return {"success": True, "output": out, "error": "", "data": {"title": title, "author": author}}
        except Exception as e:
            return {"success": False, "output": "", "error": f"YouTube error: {e}", "data": {}}

    def _handle_research(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Research chain: /research <topic> -> web_search -> fetch top N -> summarize -> one brief."""
        topic = (params.get("query") or params.get("topic") or "").strip()
        if not topic:
            return {"success": False, "output": "", "error": "No topic provided", "data": {}}
        top_n = int(params.get("top_n", params.get("top_k", 4)))
        try:
            results = self.web_search.search(topic, top_k=top_n)
            if isinstance(results, str):
                return {"success": True, "output": results, "error": "", "data": {}}
            if not results:
                return {"success": False, "output": "", "error": f"No results for '{topic}'", "data": {}}
            sources = []
            for r in results[:top_n]:
                url = r.get("url", "")
                if not url:
                    continue
                try:
                    fetched = self.web_fetcher.extract_text(url)
                    text = fetched.get("text", "") or fetched.get("content", "")
                except Exception:
                    text = ""
                sources.append({"title": r.get("title", url), "url": url, "text": text})
            brief = summarize_sources(sources, query=topic, max_sentences=5)
            out = f"📚 Research: {topic}\n\n{brief}"
            src_list = "\n".join(f"- {s['url']}" for s in sources if s.get("url"))
            if src_list:
                out += f"\n\nSources:\n{src_list}"
            return {"success": True, "output": out, "error": "", "data": {"sources": [s.get("url") for s in sources]}}
        except Exception as e:
            return {"success": False, "output": "", "error": f"Research error: {e}", "data": {}}


    # ---- Pollinations unified media tools (text/video/audio/voice/embed) ----
    def _handle_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unified LLM text/chat completion: /text <prompt>."""
        prompt = (params.get("prompt") or "").strip()
        model = (params.get("model") or "openai").strip()
        if not prompt:
            return {"success": False, "error": "Usage: /text <prompt>"}
        try:
            res = self.media.text(prompt=prompt, model=model)
            if not res.get("success"):
                return {"success": False, "error": res.get("error", "text gen failed")}
            return {"success": True, "output": res["text"], "error": "",
                    "data": {"model": res.get("model"), "provider": res.get("provider")}}
        except Exception as e:
            return {"success": False, "error": f"Text error: {e}"}

    def _handle_video(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Text-to-video generation: /video <prompt>."""
        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            return {"success": False, "error": "Usage: /video <prompt>"}
        try:
            res = self.media.video(prompt=prompt)
            if not res.get("success"):
                return {"success": False, "error": res.get("error", "video gen failed")}
            return {"success": True, "output": f"🎬 Video ready: {res['url']}", "error": "",
                    "data": {"url": res["url"], "provider": res.get("provider")}}
        except Exception as e:
            return {"success": False, "error": f"Video error: {e}"}

    def _handle_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """NeuralAI Pollinations TTS: /audio <text>."""
        text = (params.get("text") or "").strip()
        voice = (params.get("voice") or "alloy").strip()
        if not text:
            return {"success": False, "error": "Usage: /audio <text>"}
        try:
            res = self.media.audio(text=text, voice=voice)
            if not res.get("success"):
                return {"success": False, "error": res.get("error", "audio gen failed")}
            return {"success": True, "output": f"🔊 Audio ready: {res['url']}", "error": "",
                    "data": {"url": res["url"], "provider": res.get("provider")}}
        except Exception as e:
            return {"success": False, "error": f"Audio error: {e}"}

    def _handle_voice(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Realtime voice session info: /voice."""
        try:
            info = self.media.realtime_voice()
            if not info.get("success"):
                return {"success": False, "error": info.get("error", "realtime voice unavailable")}
            return {"success": True,
                    "output": f"🎙️ Realtime voice WebSocket ready:\n{info['url']}\n\nConnect from the browser/client to stream voice. Model: {info.get('model')}",
                    "error": "", "data": info}
        except Exception as e:
            return {"success": False, "error": f"Voice error: {e}"}

    def _handle_embed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Text embeddings: /embed <text>."""
        text = (params.get("text") or "").strip()
        model = (params.get("model") or "openai-3-small").strip()
        if not text:
            return {"success": False, "error": "Usage: /embed <text>"}
        try:
            res = self.media.embed(text=text, model=model)
            if not res.get("success"):
                return {"success": False, "error": res.get("error", "embed failed")}
            vec = res.get("embedding", [])
            return {"success": True,
                    "output": f"🧮 Embedding ready — model {res.get('model')}, dims {len(vec)}. (stored; first 5: {vec[:5]})",
                    "error": "", "data": res}
        except Exception as e:
            return {"success": False, "error": f"Embed error: {e}"}

    def _handle_calc(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """/calc - safe math/code eval via local sympy + restricted eval."""
        expr = (params.get("expr") or params.get("expression") or params.get("text") or "").strip()
        if not expr:
            return {"success": False, "error": "Usage: /calc <expression>"}
        try:
            import sympy
            from sympy.parsing.sympy_parser import parse_expr as sp_parse, standard_transformations, implicit_multiplication_application
            try:
                parsed = sp_parse(expr, transformations=standard_transformations + (implicit_multiplication_application,))
                val = sympy.N(parsed)
                out = f"🧮 {expr} = {val}"
                return {"success": True, "output": out, "error": "", "data": {"result": str(val)}}
            except Exception:
                pass
            allowed = {"__builtins__": {}}
            val = eval(expr, allowed, {})
            out = f"🧮 {expr} = {val}"
            return {"success": True, "output": out, "error": "", "data": {"result": str(val)}}
        except Exception as e:
            return {"success": False, "error": f"Calc error: {e}"}

    def _handle_doc(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """/doc <file> - summarize a workspace file."""
        path = (params.get("path") or params.get("file") or params.get("text") or "").strip()
        if not path:
            return {"success": False, "error": "Usage: /doc <workspace file path>"}
        try:
            full = path if path.startswith("/") else os.path.join(self.workspace, path)
            if not os.path.exists(full):
                return {"success": False, "error": f"File not found: {full}"}
            res = self.file_manager.read_file(full, max_size=60000)
            if not res.get("success"):
                return {"success": False, "error": res.get("error", "read failed")}
            content = res.get("content", "")
            if len(content) > 4000:
                try:
                    from tools.summarize import summarize_sources
                    s = summarize_sources([{"text": content[:20000]}], max_words=200)
                    summary = s.get("summary", content[:1500])
                except Exception:
                    summary = content[:1500] + "\n…(truncated)"
            else:
                summary = content
            out = f"📄 {os.path.basename(full)} ({res.get('size', len(content))} bytes)\n\n{summary}"
            return {"success": True, "output": out, "error": "", "data": {"path": full}}
        except Exception as e:
            return {"success": False, "error": f"Doc error: {e}"}

    def _handle_vision(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """/vision <image> - describe an image via OpenRouter vision model."""
        ref = (params.get("image") or params.get("url") or params.get("path") or params.get("text") or "").strip()
        if not ref:
            return {"success": False, "error": "Usage: /vision <image url or /workspace path>"}
        try:
            import base64, requests, mimetypes
            key = os.environ.get("Open_Router_API") or os.environ.get("OPENROUTER_API_KEY")
            if not key:
                return {"success": False, "error": "OpenRouter key missing (vision unavailable)"}
            if ref.startswith("http://") or ref.startswith("https://"):
                data_url = ref
            else:
                full = ref if ref.startswith("/") else os.path.join(self.workspace, ref)
                if not os.path.exists(full):
                    return {"success": False, "error": f"Image not found: {full}"}
                mime = mimetypes.guess_type(full)[0] or "image/png"
                b64 = base64.b64encode(open(full, "rb").read()).decode()
                data_url = f"data:{mime};base64,{b64}"
            model = "google/gemini-2.5-flash"
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }]}, timeout=60,
            )
            if resp.status_code != 200:
                return {"success": False, "error": f"Vision API {resp.status_code}: {resp.text[:200]}"}
            desc = resp.json()["choices"][0]["message"]["content"]
            return {"success": True, "output": f"👁️ {desc}", "error": "", "data": {"model": model}}
        except Exception as e:
            return {"success": False, "error": f"Vision error: {e}"}

    def _handle_remind(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """/remind <time> <msg> - schedule a reminder via Zo (SMS/email/Telegram)."""
        raw = (params.get("text") or params.get("message") or "").strip()
        if not raw:
            return {"success": False, "error": "Usage: /remind <time> <message>"}
        try:
            fire = _parse_remind_time(raw)
            if not fire:
                return {"success": False, "error": "Could not parse time. Use e.g. '30m', '2h', 'tomorrow 9am', '2026-07-20 14:00'."}
            msg = _remind_msg(raw)
            db = get_db()
            db.execute(
                "INSERT INTO reminders (fire_at, message, channel, done, created_at) VALUES (?, ?, ?, 0, ?)",
                (fire, msg, (params.get("channel") or "sms"), datetime.now(timezone.utc).isoformat()),
            )
            db.commit()
            db.close()
            human = datetime.fromtimestamp(fire, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %I:%M %p %Z")
            return {
                "success": True,
                "output": f"⏰ Reminder set for {human}: {msg}\n(Channel: {params.get('channel') or 'sms'}. Fired by neuralai-reminder-daemon.)",
                "error": "", "data": {"fire_at": fire, "message": msg},
            }
        except Exception as e:
            return {"success": False, "error": f"Remind error: {e}"}

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


# Global handler instance
_handler: Optional[ToolHandler] = None


def get_handler(workspace: str = "/home/workspace") -> ToolHandler:
    """Get or create the global tool handler."""
    global _handler
    if _handler is None:
        _handler = ToolHandler(workspace=workspace)
    return _handler


def run_tool(tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool and return the result."""
    handler = get_handler()
    return handler.execute(tool, params)


if __name__ == "__main__":
    # Test the tool handler
    handler = ToolHandler()
    
    print("Testing file manager:")
    result = handler.execute("file_manager", {"query": "list files"})
    print(result["output"])
    
    print("\nTesting git:")
    result = handler.execute("git", {"action": "status"})
    print(result["output"])