import asyncio
import aiohttp
import os
import json

# Configuration
MODEL_NAME = "byok:97f38ab9-cce0-48fb-bfb0-f956ec13f723"
API_URL = "https://api.zo.computer/zo/ask"
TOKEN = os.environ.get("ZO_CLIENT_IDENTITY_TOKEN")

async def spawn_worker(session, task_id, prompt):
    """
    Spawns a child Zo agent (Worker) to handle a specific sub-task.
    """
    print(f"[Manager] Dispatching Worker {task_id}: {prompt[:50]}...")
    
    payload = {
        "input": f"TASK_ID: {task_id}\n\n{prompt}",
        "model_name": MODEL_NAME,
    }
    
    headers = {
        "authorization": TOKEN,
        "content-type": "application/json",
        "Accept": "application/json",
    }
    
    try:
        async with session.post(API_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"[Worker {task_id}] Completed successfully.")
                return {"id": task_id, "output": data.get("output"), "status": "success"}
            else:
                print(f"[Worker {task_id}] Failed with status {resp.status}")
                return {"id": task_id, "output": None, "status": "error"}
    except Exception as e:
        print(f"[Worker {task_id}] Exception: {str(e)}")
        return {"id": task_id, "output": None, "status": "exception"}

async def manager_orchestrate(main_goal):
    """
    The Manager implementation of the Orchestrator logic.
    """
    print(f"🚀 Starting Orchestration for Goal: {main_goal}\n")
    
    # 1. Goal Decomposition (In a real scenario, another LLM call would generate this list)
    # For the prototype, we manually decompose a sample goal: "Research the transition from 
    # Ancient Civilizations to Classical Antiquity and identify key technological shifts."
    tasks = [
        {"id": "worker_1", "prompt": "Research the key characteristics of the Greek Archaic period and the rise of the City-State (Polis)."},
        {"id": "worker_2", "prompt": "Analyze the technological shifts in metallurgy and warfare during the transition to Classical Antiquity."},
        {"id": "worker_3", "prompt": "Summarize the influence of Phoenician trade on the spread of the alphabet and early Mediterranean diplomacy."}
    ]
    
    async with aiohttp.ClientSession() as session:
        # 2. Parallel Worker Dispatch
        print(f"[Manager] Spawning {len(tasks)} Workers in parallel...")
        worker_futures = [spawn_worker(session, t["id"], t["prompt"]) for t in tasks]
        results = await asyncio.gather(*worker_futures)
    
    # 3. Synthesis
    print("\n[Manager] All Workers returned. Synthesizing results...")
    final_report = "## 🏛️ Orchestrated Synthesis: Transition to Classical Antiquity\n\n"
    
    for res in results:
        if res["status"] == "success":
            final_report += f"### Result from {res['id']}\n{res['output']}\n\n"
        else:
            final_report += f"### Result from {res['id']}\n[Worker failed to provide data]\n\n"
            
    return final_report

if __name__ == "__main__":
    if not TOKEN:
        print("Error: ZO_CLIENT_IDENTITY_TOKEN not found in environment.")
    else:
        goal = "Research the transition from Ancient Civilizations to Classical Antiquity."
        final_output = asyncio.run(manager_orchestrate(goal))
        
        # Save the synthesized result to the workspace
        output_path = "/home/workspace/Projects/NeuralAI/orchestrator_prototype_result.md"
        with open(output_path, "w") as f:
            f.write(final_output)
            
        print(f"\n✅ Orchestration complete. Synthesis saved to: {output_path}")
