# neuralai_router.py
#
# Clean routing logic for NeuralAI
# - Local model is DEFAULT
# - Uplink ONLY for heavy tasks
# - Tools: terminal, code_exec, code_gen, image_gen, file_manager, web_fetcher, database, git

from typing import Tuple, Optional, Dict, List
import re


# Tool detection patterns
TOOL_PATTERNS: Dict[str, List[str]] = {
    # Image generation - highest priority for creative requests
    "image_gen": [
        "generate", "image", "drawing", "picture", "photo", "artwork", "art",
        "create an image", "create a image", "create image",
        "generate an image", "generate a image", "generate image",
        "make an image", "make a image", "make image",
        "draw an image", "draw a image", "draw image",
        "create a picture", "generate a picture", "make a picture",
        "create a photo", "generate a photo", "make a photo",
        "create a moon image", "create a sun image", "create a star image",
        "create an ai image", "generate an ai image", "ai image",
        "create a drawing", "generate a drawing", "make a drawing",
        "create artwork", "generate artwork", "make artwork",
        "create art", "generate art", "make art",
        "image of", "picture of", "photo of", "drawing of",
        "show me an image", "show me a picture", "show me a photo",
        "i want an image", "i want a picture", "i want a photo",
        "generate me an", "create me an", "make me an",
        # Multimodal / Editing patterns
        "edit this image", "edit the image", "modify this image", "modify the image",
        "change this image", "change the image", "transform this image",
        "edit my photo", "edit my picture", "stylize this", "make a variant",
        "variation of", "edit image", "modify image", "enhance image"
    ],
    
    "terminal": ["shell ", "command ", "bash ", "terminal "],
    
    "code_exec": [
        "run this code", "execute this code", "run the code",
        "run this python", "execute this python", "run python code",
        "run python", "execute python",
        "run this javascript", "execute this javascript", "run js code",
        "run js", "execute js", "execute javascript",
        "execute the script", "run the script", "run code", "execute code"
    ],
    
    # Code generation requests - AI should write AND run code
    "code_gen": [
        "write code", "write a program", "write a script", "write a function",
        "write python", "write javascript", "write js",
        "create a program", "create a script", "create code",
        "generate code", "generate a program", "generate a script",
        "make a program", "make a script", "make code",
        "build a program", "build a script", "build code",
        "code for", "program for", "script for",
        "you write", "write your own", "you do it",
        "u do it", "u write", "you generate",
        # More flexible patterns
        "write a", "write some", "write the",
        "can you write", "could you write", "please write",
        "i want you to write", "help me write",
        "create a", "create some", "make a", "make some",
        "generate a", "generate some", "build a",
        "write me", "write us", "write something"
    ],
    
    "file_manager": [
        "read file", "write file", "create file", "delete file",
        "list files", "show files", "search files", "find files",
        "file operations", "open file", "edit file",
        "list directory", "show directory", "what files",
        "search for", "find in files", "files"
    ],
    
    "web_fetcher": [
        "fetch url", "get url", "fetch this url", "get content from",
        "scrape this page", "scrape url", "fetch page",
        "get webpage", "fetch website", "extract from url",
        "read website", "download page", "get page content", "fetch"
    ],
    
    "database": [
        "query database", "query the database", "database query",
        "connect to database", "sqlite query", "run sql",
        "execute sql", "database operations", "db query",
        "show tables", "get schema", "database schema"
    ],
    
    
    "voice_transcribe": [
        "transcribe", "voice note", "audio file", "transcription",
        "convert to text", "speech to text", "voice to text",
        "what did they say", "listen to this"
    ],
    
    "research": [
        "research", "search for", "look up", "find information",
        "what is", "tell me about", "explain", "investigate",
        "search the web", "web search", "google"
    ],

    "git": [
        "git status", "git commit", "git push", "git pull", "git branch",
        "git log", "git diff", "git add", "git stash",
        "create a commit", "make a commit", "push to git",
        "create a pr", "create pull request", "git pr",
        "create issue", "git issue"
    ]
}


def neuralai_route(msg: str) -> Tuple[str, Optional[str]]:
    """
    Route messages to the correct handler.
    
    Returns:
        (route_type, tool_name)
        
    Routes:
        ("local", None) → Main SmolLM2 model
        ("uplink", None) → Neural Uplink agent network
        ("tool", "image_gen") → Image generation
        ("tool", "terminal") → Shell execution
        ("tool", "code_exec") → Code sandbox (run provided code)
        ("tool", "code_gen") → Code generation + execution (AI writes code)
        ("tool", "file_manager") → File operations
        ("tool", "web_fetcher") → Web fetching
        ("tool", "database") → Database operations
        ("tool", "git") → Git operations
    """
    lower = msg.lower().strip()
    
    # 1. Check for image generation FIRST (highest priority for creative)
    for pattern in TOOL_PATTERNS["image_gen"]:
        if pattern in lower:
            return ("tool", "image_gen")
    
    # 2. Terminal commands
    for prefix in TOOL_PATTERNS["terminal"]:
        if lower.startswith(prefix):
            return ("tool", "terminal")
    
    # 3. Check code generation requests BEFORE code execution
    for pattern in TOOL_PATTERNS["code_gen"]:
        if pattern in lower:
            return ("tool", "code_gen")
    
    # 4. Check other tools (order matters for priority)
    tool_order = ["git", "database", "web_fetcher", "file_manager", "code_exec"]
    for tool in tool_order:
        for pattern in TOOL_PATTERNS[tool]:
            if pattern in lower:
                return ("tool", tool)
    
    # 5. Check for Research/Analysis -> UPLINK
    for pattern in TOOL_PATTERNS["research"]:
        if pattern in lower:
            return ("uplink", None)
            
    # 6. EVERYTHING ELSE → MAIN MODEL (default)
    return ("local", None)


def detect_tool(msg: str) -> Optional[str]:
    """Detect if a message should trigger a tool."""
    route, tool = neuralai_route(msg)
    if route == "tool":
        return tool
    return None


def extract_tool_params(msg: str, tool: str) -> Dict[str, str]:
    """Extract parameters for a tool from the message."""
    lower = msg.lower()
    
    if tool == "image_gen":
        # Check for editing intent
        is_edit = any(kw in lower for kw in ["edit", "modify", "change", "transform", "variant", "variation", "enhance"])
        
        # Extract the image description
        prompt = msg
        # Remove trigger phrases
        triggers = [
            "create an image of", "create a image of", "create image of",
            "generate an image of", "generate a image of", "generate image of",
            "make an image of", "make a image of", "make image of",
            "create an image", "generate an image", "make an image",
            "show me an image of", "show me a image of",
            "i want an image of", "i want a image of",
            "generate me an", "create me an", "make me an",
            "edit this image to", "edit the image to", "modify this image to",
            "change this image to", "transform this image to", "edit my photo to",
            "edit this image", "edit the image", "modify this image",
            "change this image", "transform this image", "edit my photo"
        ]
        for trigger in triggers:
            if trigger in lower:
                idx = lower.find(trigger)
                prompt = msg[idx + len(trigger):].strip()
                break
        
        return {
            "prompt": prompt or msg, 
            "style": "realistic",
            "mode": "edit" if is_edit else "gen"
        }
    
    if tool == "terminal":
        for prefix in TOOL_PATTERNS["terminal"]:
            if lower.startswith(prefix):
                return {"command": msg[len(prefix):].strip()}
        return {"command": msg}
    
    if tool == "code_exec":
        language = "python"
        if "javascript" in lower or "js" in lower:
            language = "javascript"
        
        code_match = re.search(r'```(?:python|javascript|js)?\s*([\s\S]*?)```', msg)
        if code_match:
            return {"language": language, "code": code_match.group(1).strip()}
        
        code_indicators = ['print(', 'console.log(', 'def ', 'function ', 'import ', 'var ', 'let ', 'const ']
        for indicator in code_indicators:
            if indicator in msg:
                idx = msg.find(indicator)
                return {"language": language, "code": msg[idx:].strip()}
        
        return {"language": language, "code": msg}
    
    if tool == "code_gen":
        # Extract what code to generate
        description = msg
        triggers = ["write code to", "write a program to", "write a script to",
                    "create a program to", "create a script to", "generate code to"]
        for trigger in triggers:
            if trigger in lower:
                idx = lower.find(trigger)
                description = msg[idx + len(trigger):].strip()
                break
        return {"description": description or msg}
    
    if tool == "web_fetcher":
        url_pattern = r'https?://[^\s]+'
        match = re.search(url_pattern, msg)
        if match:
            return {"url": match.group(0)}
        return {}
    
    if tool == "git":
        for pattern in TOOL_PATTERNS["git"]:
            if pattern in lower:
                return {"action": pattern.replace("git ", "").strip()}
        return {"action": "status"}
    
    if tool == "file_manager":
        query = msg
        for pattern in TOOL_PATTERNS["file_manager"]:
            if pattern in lower:
                idx = lower.find(pattern)
                query = msg[idx + len(pattern):].strip()
                break
        return {"query": query or msg}
    
    if tool == "database":
        sql_keywords = ["select", "insert", "update", "delete", "create", "drop"]
        for kw in sql_keywords:
            if kw in lower:
                return {"has_sql": "true"}
        return {"query": msg}
    
    return {}


def should_use_uplink(msg: str) -> bool:
    """Legacy compatibility."""
    route, _ = neuralai_route(msg)
    return route == "uplink"


def strip_terminal_prefix(msg: str) -> str:
    """Strip terminal prefixes from message."""
    lower = msg.lower()
    for p in TOOL_PATTERNS["terminal"]:
        if lower.startswith(p):
            return msg[len(p):].strip()
    return msg.strip()


TOOL_DESCRIPTIONS = {
    "image_gen": "Generate images using AI",
    "terminal": "Execute shell commands",
    "code_exec": "Run Python or JavaScript code safely",
    "code_gen": "Write and execute code",
    "file_manager": "Read, write, search, and manage files",
    "web_fetcher": "Fetch and parse web content",
    "database": "Query SQLite databases",
    "git": "Git operations (status, commit, push, branch, PR)"
}


def get_tool_help() -> str:
    """Get help text for all available tools."""
    help_text = "Available tools:\n\n"
    for tool, desc in TOOL_DESCRIPTIONS.items():
        help_text += f"• {tool}: {desc}\n"
    return help_text
