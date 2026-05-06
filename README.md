# LockGuard RL v4

**Multi-Agent Simulation-Gated Reinforcement Learning with Causal Governance**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20054703.svg)](https://doi.org/10.5281/zenodo.20054703)

---

### Descripción

LockGuard RL v4 es un framework avanzado de **Reinforcement Learning multi-agente seguro** que aborda el problema de la **irreversibilidad colectiva**: cuando acciones individualmente seguras generan resultados catastróficos a nivel de equipo o sistema.

**Componentes principales:**
- **Collective Reality Gate (CRG)** — Decide si el equipo puede ejecutar en realidad o debe permanecer en Dream Mode
- **Causal Irreversibility Module (CIM)** — Atribución causal de riesgo (\~90% precisión)
- **Hierarchical EGDS (H-EGDS)** — Gobernanza temporal jerárquica (Agente → Equipo → Sistema)
- **Bandwidth-aware Communication** — Comunicación inteligente según nivel de riesgo

---

### Resultados Destacados (Financial LOB, N=8 agentes)

- Reducción del **71-84%** en Collective Irreversible Action Rate (C-IAR)
- Eventos críticos colectivos reducidos a **0.3-0.5%**
- Mejora en Sharpe Ratio: **1.41 → 1.89** (simulación)
- Implementation Shortfall: **18.4 → 11.3 bps**

---

### Estructura del Repositorio

```bash
LockGuard-RL-v4/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/default.yaml
├── code/
│   ├── agents/mappo_lockguard.py
│   ├── lockguard_core/
│   │   ├── crg.py
│   │   ├── cim.py
│   │   └── hegds.py
│   ├── envs/lob_env.py
│   └── utils/logger.py
├── experiments/
│   └── train_multiagent.py
├── results/
├── notebooks/
├── docs/
└── checkpoints/
