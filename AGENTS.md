# NeuralAI Maintenance Agent Memory

## 🛡️ PROTECTED FEATURES (DO NOT MODIFY)
1.  **Google-Style Design**: The UI layout, colors, and 'Google Sans' typography are SACRED.
2.  **Premium Terminal**: The JetBrains Mono terminal design is FINAL. Do not switch to simplified versions.
3.  **Navigation Tabs**: The 'Chat', 'Files', 'Terminal', and 'Settings' tabs are core to the user experience.
4.  **Uplink Status**: The Model Status and Uplink details in the sidebar must remain visible.
5.  **Dynamic Memory & Rules**: These features MUST remain database-driven. Never use hardcoded mock data for them.

## ⚙️ TECHNICAL ARCHITECTURE
- **Unified Core**: `services/neural_core_service.py` is the single source of truth for both model inference and the Web API.
- **Database**: `data/neuralai.db` stores all users, settings, memory, and rules.
- **Frontend**: `from-scratch/web_ui/templates/index.html` is a high-complexity, self-contained UI with premium styling.

## 🚀 MAINTENANCE GUIDELINES
- Always verify changes against the live URL.
- Before editing UI, re-read the Design Integrity rules in `GEMINI.md`.
- Never truncate files when using `edit_file_llm`.
