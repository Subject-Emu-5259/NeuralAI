# NeuralAI Service Architecture

## Podman Quadlet → Zo Service Mapping

| Podman Concept | Zo Equivalent | NeuralAI Implementation |
|----------------|---------------|------------------------|
| `.container` files | `register_user_service()` | 3 registered services |
| systemd `Requires=` | Service startup order | model→webui dependency |
| systemd `After=` | Service health checks | tools→webui dependency |
| NFS volumes | `/home/workspace/NeuralAI/` | Built-in persistent storage |
| `Restart=always` | Zo supervisor | Auto-restart on crash |
| `.network` files | `localhost` networking | Services talk via HTTP |

## Service Definitions (like Quadlets)

### neuralai-model.service (port 7001)
```ini
# Equivalent to: neuralai-model.container
[Service]
Entrypoint=python3 /home/workspace/Projects/NeuralAI/services/model_service.py
Port=7001
Restart=always
Environment=MODEL_PATH=/home/workspace/Projects/NeuralAI/checkpoints/v2_model

[Unit]
Description=NeuralAI Model Inference Service
# Model must be ready before webui
Before=neuralai-webui.service
```

### neuralai-tools.service (port 7002)
```ini
# Equivalent to: neuralai-tools.container
[Service]
Entrypoint=python3 /home/workspace/Projects/NeuralAI/services/tools_service.py
Port=7002
Restart=always
Volume=/home/workspace/NeuralAI:/data:rw

[Unit]
Description=NeuralAI Tools Service (sandbox, storage)
# Tools must be ready before webui
Before=neuralai-webui.service
```

### neuralai-webui.service (port 5000)
```ini
# Equivalent to: neuralai-webui.container
[Service]
Entrypoint=python3 /home/workspace/Projects/NeuralAI/services/webui_service.py
Port=5000
Public=true
Restart=always

[Unit]
Description=NeuralAI Web Interface
Requires=neuralai-model.service
Requires=neuralai-tools.service
After=neuralai-model.service
After=neuralai-tools.service
```

## Dependency Graph

```
                    ┌─────────────────┐
                    │  neuralai-model │ :7001
                    │  (inference)    │
                    └────────┬────────┘
                             │
                             │ Requires
                             ▼
┌─────────────────┐    ┌─────────────────┐
│  neuralai-tools │───▶│  neuralai-webui │ :5000 (public)
│  (sandbox)      │    │  (router)       │
└─────────────────┘    └─────────────────┘
       :7002                   │
                                │
                       Public URL:
                       https://neuralai-webui-deandrewharris.zocomputer.io
```

## Storage (like NFS volume)

```
/home/workspace/NeuralAI/
├── images/          # Generated images (shared)
├── documents/       # User documents for RAG (shared)
├── conversations/   # Chat history (shared)
├── models/          # Model checkpoints (shared)
└── cache/           # Temporary files (shared)
```

All services mount `/home/workspace/NeuralAI/` - like an NFS volume shared across containers.

## Management Commands

### Like systemctl for Podman:
```bash
# Check status
systemctl --user status neuralai-*.service
# In Zo:
service_doctor("neuralai-model")

# Restart a service
systemctl --user restart neuralai-model.service
# In Zo:
update_user_service(service_id)

# View logs
journalctl --user -u neuralai-model.service
# In Zo:
cat /dev/shm/neuralai-model.log
```

### Start/Stop All:
```bash
# Podman quadlet style
./services/start_all.sh    # Starts in dependency order
./services/stop_all.sh     # Stops in reverse order
./services/status.sh       # Shows all service states
```

## Restart Behavior

| Failure Scenario | What Happens |
|------------------|--------------|
| Model OOM crash | Only model service restarts, webui stays up |
| Tools timeout | Only tools service restarts |
| WebUI crash | Only webui restarts, model stays loaded |
| Model unhealthy | Webui returns 503, auto-recovery in progress |

## Benefits (Same as Your Podman Setup)

1. **Isolation**: Each service in its own process
2. **Independence**: Failures don't cascade
3. **Dependencies**: Proper startup ordering
4. **Shared storage**: All services access same `/NeuralAI/`
5. **Declarative**: Service configs are simple Python files
6. **Auto-restart**: Supervisor handles recovery
7. **Observability**: Separate logs per service

## Current Status

```
✅ neuralai-model  (7001) - Ready, model loaded
✅ neuralai-tools  (7002) - Ready, sandbox active
✅ neuralai-webui  (5000) - Ready, public URL live
```

Public: https://neuralai-webui-deandrewharris.zocomputer.io
