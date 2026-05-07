#!/usr/bin/env python3
"""
Neural Uplink - Multi-Agent AI System
=====================================
A standalone tool that runs 4 specialized AI agents in parallel:
- DIALOG: Conversation and reasoning
- DATA: Data analysis and retrieval  
- OPS: Operations and execution
- WORLD: World-building and creative tasks

Usage:
  Start: python3 neural_uplink.py
  Query: curl -X POST http://localhost:8000/uplink -d '{"prompt": "analyze this..."}'
"""

import os
import sys
import json
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, Response
import threading
import queue

# Configuration
PORT = int(os.environ.get("UPLINK_PORT", "8000"))
MODEL_SERVICE = os.environ.get("MODEL_SERVICE", "http://localhost:7001")

app = Flask(__name__)

# Agent definitions
AGENTS = {
    "dialog": {
        "name": "DIALOG",
        "role": "Conversation & Reasoning",
        "system": "You are DIALOG, an AI agent specialized in conversation, reasoning, and explanation. Be concise and insightful.",
        "color": "🔵"
    },
    "data": {
        "name": "DATA", 
        "role": "Data Analysis",
        "system": "You are DATA, an AI agent specialized in data analysis, patterns, and retrieval. Focus on facts and numbers.",
        "color": "🟢"
    },
    "ops": {
        "name": "OPS",
        "role": "Operations & Execution", 
        "system": "You are OPS, an AI agent specialized in operations, execution, and practical solutions. Be actionable.",
        "color": "🟡"
    },
    "world": {
        "name": "WORLD",
        "role": "World-Building & Creativity",
        "system": "You are WORLD, an AI agent specialized in creative thinking, world-building, and ideation. Be imaginative.",
        "color": "🟣"
    }
}


async def query_agent(agent_name: str, prompt: str, timeout: int = 30) -> dict:
    """Query a single agent via the model service."""
    agent = AGENTS.get(agent_name)
    if not agent:
        return {"agent": agent_name, "error": "Unknown agent"}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Build prompt with agent's system message
            full_prompt = f"<|im_start|>system\n{agent['system']}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            async with session.post(
                f"{MODEL_SERVICE}/generate",
                json={"prompt": prompt, "max_tokens": 128, "temperature": 0.7},
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "agent": agent["name"],
                        "role": agent["role"],
                        "color": agent["color"],
                        "response": data.get("response", ""),
                        "tokens": data.get("tokens_generated", 0)
                    }
                return {"agent": agent["name"], "error": f"HTTP {resp.status}"}
    except asyncio.TimeoutError:
        return {"agent": agent["name"], "error": "timeout"}
    except Exception as e:
        return {"agent": agent["name"], "error": str(e)}


async def query_all_agents(prompt: str) -> list:
    """Query all 4 agents in parallel."""
    tasks = [
        query_agent("dialog", prompt),
        query_agent("data", prompt),
        query_agent("ops", prompt),
        query_agent("world", prompt)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter and format results
    outputs = []
    for result in results:
        if isinstance(result, Exception):
            continue
        if "error" not in result or not result.get("error"):
            outputs.append(result)
    
    return outputs


def fuse_responses(responses: list) -> str:
    """Combine agent responses into a coherent output."""
    if not responses:
        return "[Neural Uplink] No agent responses. Please try again."
    
    lines = ["🧠 **Neural Uplink - Multi-Agent Analysis**\n"]
    
    for resp in responses:
        color = resp.get("color", "⚪")
        name = resp.get("agent", "Agent")
        role = resp.get("role", "")
        response = resp.get("response", "")
        
        if response and len(response) > 10:
            lines.append(f"\n{color} **{name}** ({role}):")
            lines.append(f"> {response.strip()}")
    
    return "\n".join(lines)


# ==================
# API ENDPOINTS
# ==================

@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({
        "status": "ready",
        "port": PORT,
        "agents": list(AGENTS.keys()),
        "model_service": MODEL_SERVICE
    })


@app.route("/uplink", methods=["POST"])
def uplink():
    """Main uplink endpoint - queries all agents."""
    data = request.get_json()
    prompt = data.get("prompt", data.get("message", ""))
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Run async query
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(query_all_agents(prompt))
        fused = fuse_responses(results)
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "agent_count": len(results),
            "responses": results,
            "fused": fused
        })
    finally:
        loop.close()


@app.route("/uplink/stream", methods=["POST"])
def uplink_stream():
    """Streaming uplink - yields responses as they arrive."""
    data = request.get_json()
    prompt = data.get("prompt", data.get("message", ""))
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    def generate():
        # Query each agent sequentially for streaming
        import requests as sync_requests
        
        full_response = "🧠 **Neural Uplink Analysis**\n\n"
        
        for agent_name, agent in AGENTS.items():
            try:
                resp = sync_requests.post(
                    f"{MODEL_SERVICE}/generate",
                    json={"prompt": prompt, "max_tokens": 100, "temperature": 0.7},
                    timeout=30
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    response = data.get("response", "")
                    
                    if response and len(response) > 5:
                        chunk = f"{agent['color']} **{agent['name']}**: {response.strip()}\n\n"
                        full_response += chunk
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                        
            except Exception as e:
                yield "data: " + json.dumps({"content": agent["color"] + " **" + agent["name"] + "**: Error - " + str(e) + "\n\n"}) + "\n\n"
        
        yield "data: [DONE]\n\n"
    
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


@app.route("/agent/<agent_name>", methods=["POST"])
def single_agent(agent_name):
    """Query a single specific agent."""
    if agent_name not in AGENTS:
        return jsonify({"error": f"Unknown agent: {agent_name}. Available: {list(AGENTS.keys())}"}), 400
    
    data = request.get_json()
    prompt = data.get("prompt", data.get("message", ""))
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(query_agent(agent_name, prompt))
        return jsonify(result)
    finally:
        loop.close()


# ==================
# CLI INTERFACE
# ==================

def cli_mode():
    """Interactive CLI for Neural Uplink."""
    print("\n🧠 Neural Uplink - Multi-Agent AI System")
    print("=" * 50)
    print("Commands:")
    print("  <prompt>     - Query all agents")
    print("  !<agent>     - Query specific agent (dialog/data/ops/world)")
    print("  !help        - Show this help")
    print("  !quit        - Exit")
    print()
    
    while True:
        try:
            user_input = input("Uplink> ").strip()
            
            if not user_input:
                continue
            
            if user_input == "!quit":
                print("Goodbye!")
                break
            
            if user_input == "!help":
                print("\nAgents:")
                for name, agent in AGENTS.items():
                    print(f"  {agent['color']} {agent['name']}: {agent['role']}")
                print()
                continue
            
            # Query specific agent
            if user_input.startswith("!"):
                agent_name = user_input[1:].lower().split()[0]
                prompt = " ".join(user_input.split()[1:]) or "Hello"
                
                if agent_name in AGENTS:
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(query_agent(agent_name, prompt))
                    loop.close()
                    
                    print(f"\n{result.get('color', '⚪')} {result.get('agent')}:")
                    print(f"  {result.get('response', result.get('error', 'No response'))}\n")
                else:
                    print(f"Unknown agent: {agent_name}\n")
                continue
            
            # Query all agents
            print("\nQuerying all agents...")
            loop = asyncio.new_event_loop()
            results = loop.run_until_complete(query_all_agents(user_input))
            loop.close()
            
            print(fuse_responses(results))
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Uplink - Multi-Agent AI System")
    parser.add_argument("--cli", action="store_true", help="Run in interactive CLI mode")
    parser.add_argument("--port", type=int, default=PORT, help="Port for HTTP API")
    parser.add_argument("--query", type=str, help="Single query and exit")
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(query_all_agents(args.query))
        loop.close()
        print(fuse_responses(results))
    elif args.cli:
        cli_mode()
    else:
        # HTTP API mode
        PORT = args.port
        print(f"\n🧠 Neural Uplink Starting...")
        print(f"   Port: {PORT}")
        print(f"   Model Service: {MODEL_SERVICE}")
        print(f"   Agents: {', '.join(AGENTS.keys())}")
        print(f"\n   API Endpoints:")
        print(f"     POST /uplink          - Query all agents")
        print(f"     POST /uplink/stream   - Streaming response")
        print(f"     POST /agent/<name>    - Query specific agent")
        print(f"     GET  /health          - Health check")
        print()
        
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
