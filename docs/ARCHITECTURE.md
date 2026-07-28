# NeuralAI Architecture v2.0 - Microservices

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      NeuralAI Stack                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   neuralai   │     │   neuralai   │     │   neuralai   │ │
│  │    -model    │     │   -tools     │     │   -webui     │ │
│  │              │     │              │     │              │ │
│  │  Port: 7001  │     │  Port: 7002  │     │  Port: 5000  │ │
│  │  Mode: http  │     │  Mode: http  │     │  Mode: http  │ │
│  │  Public: No  │     │  Public: No  │     │  Public: Yes │ │
│  │              │     │              │     │              │ │
│  │  ┌────────┐  │     │  ┌────────┐  │     │  ┌────────┐  │ │
│  │  │ Model  │  │     │  │ Sandbox│  │     │  │  Web   │  │ │
│  │  │ LoRA   │  │     │  │ Files  │  │     │  │  UI    │  │ │
│  │  │        │  │     │  │ Shell  │  │     │  │        │  │ │
│  │  └────────┘  │     │  └────────┘  │     │  └────────┘  │ │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘ │
│         │                    │                    │          │
│         └────────────────────┴────────────────────┘          │
│                     HTTP API Calls                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   /NeuralAI/ Storage  │
              │   ├─ images/          │
              │   ├─ documents/       │
              │   ├─ voice/           │
              │   └─ cache/           │
              └───────────────────────┘
```

## Services

### 1. neuralai-model (Port 7001)
**Purpose:** Model inference only

- Loads SmolLM2-360M with LoRA adapter once on startup
- Keeps model in memory (no reload per request)
- Exposes `/generate` and `/generate/stream` endpoints
- Private (no public URL)

**Health Check:**
```bash
curl http://localhost:7001/health
```

### 2. neuralai-tools (Port 7002)
**Purpose:** Sandboxed execution

- Code execution (Python, JavaScript)
- Shell commands with timeout
- File operations on /NeuralAI/ storage
- Private (no public URL)

**Health Check:**
```bash
curl http://localhost:7002/health
```

### 3. neuralai-webui (Port 5000)
**Purpose:** User interface

- Serves the web UI
- Routes chat to model service
- Routes tools to tools service
- Public URL: https://neuralai-webui-deandrewharris.zocomputer.io

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

## Dependency Chain

Like your VM setup with systemd dependencies:

```
neuralai-model ─┐
                 ├─► neuralai-webui
neuralai-tools ─┘
```

Model and Tools can start in parallel. WebUI waits for both.

## Benefits vs Monolithic

| Issue | Monolithic | Microservices |
|-------|------------|---------------|
| Model crash | Entire app down | Only model service restarts |
| Memory leak | Affects everything | Isolated to one service |
| Code execution | Same process as model | Separate sandbox process |
| Debugging | Hard to trace | Clear service boundaries |
| Scaling | All or nothing | Scale each independently |

## File Structure

```
NeuralAI/
├── services/
│   ├── model_service.py     # Model inference API
│   ├── tools_service.py     # Sandbox & storage API
│   ├── webui_service.py     # Web interface
│   ├── start_all.sh          # Start all services
│   ├── stop_all.sh           # Stop all services
│   └── status.sh             # Check service status
├── checkpoints/
│   └── v2_model/             # Trained LoRA adapter
└── from-scratch/
    └── web_ui/
        └── templates/        # Web UI templates
```

## Storage

Personal cloud computer storage at `/home/workspace/NeuralAI/`:
- `images/` - AI-generated images
- `documents/` - User documents
- `voice/` - Voice recordings
- `cache/` - Temporary files

## Management Commands

**Start all services:**
```bash
cd /home/workspace/Projects/NeuralAI/services
./start_all.sh
```

**Stop all services:**
```bash
./stop_all.sh
```

**Check status:**
```bash
./status.sh
```

**Or use Zo service management:**
- List: `list_user_services`
- Check: `service_doctor("neuralai-model")`
- Restart: `update_user_service(service_id)`

## Comparison to Your VM Setup

| Your Setup | NeuralAI on Zo |
|------------|----------------|
| VM w/ GPU | Zo server (CPU) |
| Podman quadlets | Zo service management |
| systemd dependencies | Service startup order |
| NFS for storage | Built-in /NeuralAI/ storage |
| Separate containers | Separate service processes |

Next step: Add GPU for model service, split tools into more services if needed.
