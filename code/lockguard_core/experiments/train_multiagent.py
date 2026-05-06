"""
LockGuard RL v4 - Multi-Agent Training Script
=============================================
Script principal de entrenamiento para LockGuard RL v4
"""

import torch
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

from code.envs.lob_env import MultiAgentLOBEnv
from code.agents.mappo_lockguard import MAPPOWithLockGuard
from code.lockguard_core.crg import create_crg
from code.lockguard_core.cim import create_cim
from code.lockguard_core.hegds import create_hegds
from code.utils.logger import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_episodes', type=int, default=5000)
    args = parser.parse_args()

    # Cargar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Semilla
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando entrenamiento LockGuard RL v4 en {device}")

    # Crear entorno
    env = MultiAgentLOBEnv(
        n_agents=config['env']['n_agents'],
        episode_length=config['env']['episode_length'],
        **config.get('env', {})
    )

    # Crear componentes LockGuard
    crg = create_crg(config)
    cim = create_cim(config)
    hegds = create_hegds(config)

    # Crear agente principal
    agent = MAPPOWithLockGuard(
        env=env,
        config=config,
        device=device,
        crg=crg,
        cim=cim,
        hegds=hegds
    )

    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(log_dir=f"results/run_{timestamp}")

    print(f"✅ Entrenamiento iniciado con {config['env']['n_agents']} agentes")
    print(f"   Total episodios: {args.total_episodes}")

    # === Bucle principal de entrenamiento ===
    best_reward = -float('inf')

    for episode in range(args.total_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done:
            # Seleccionar acción
            actions = agent.select_action(obs)

            # Ejecutar en el entorno
            next_obs, reward, done, info = env.step(actions)

            # Actualizar componentes LockGuard
            lockguard_info = agent.update_lockguard(obs, actions, reward, next_obs, info)

            # Guardar transición
            agent.store_transition(obs, actions, reward, next_obs, done, info)

            obs = next_obs
            episode_reward += reward.mean().item()  # Recompensa promedio del equipo
            step += 1

        # Actualizar política
        if episode % config['training']['update_freq'] == 0:
            loss_info = agent.update()
            logger.log_metrics({**loss_info, 'episode_reward': episode_reward}, episode)

        # Logging
        if episode % 50 == 0 or episode == args.total_episodes - 1:
            cii = lockguard_info.get('cii', torch.tensor(0.0)).mean().item()
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"C-II: {cii:.3f} | "
                  f"Steps: {step}")

            # Guardar mejor modelo
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save("checkpoints/best_model")

    # Guardar modelo final
    agent.save("checkpoints/lockguard_v4_final")
    print("🎉 Entrenamiento completado!")
    print(f"   Mejor recompensa: {best_reward:.2f}")


if __name__ == "__main__":
    main()
