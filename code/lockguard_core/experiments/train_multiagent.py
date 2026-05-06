"""
LockGuard RL v4 - Multi-Agent Training Script (Actualizado)
=======================================================

Script principal de entrenamiento que integra todos los módulos LockGuard.
"""

import torch
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

from code.envs.lob_env import make_lob_env
from code.agents.mappo_lockguard import MAPPOWithLockGuard
from code.lockguard_core.crg import create_crg
from code.lockguard_core.cim import create_cim
from code.lockguard_core.hegds import create_hegds
from code.utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description="LockGuard RL v4 Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_episodes', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=500)
    args = parser.parse_args()

    # ==================== SETUP ====================
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Semilla para reproducibilidad
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando LockGuard RL v4 en {device} | Seed: {args.seed}")

    # Crear entorno
    env = make_lob_env(config)

    # Crear componentes LockGuard
    crg = create_crg(config)
    cim = create_cim(config)
    hegds = create_hegds(config)

    # Crear agente completo
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
    print(f"   Episodios totales: {args.total_episodes}")

    best_reward = -float('inf')
    best_episode = 0

    # ==================== TRAINING LOOP ====================
    for episode in range(args.total_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done:
            # Seleccionar acción con LockGuard
            actions = agent.select_action(obs)

            # Ejecutar paso
            next_obs, reward, done, truncated, info = env.step(actions)

            # Actualizar módulos LockGuard
            lockguard_info = agent.update_lockguard(obs, actions, reward, next_obs, info)

            # Guardar experiencia
            agent.store_transition(obs, actions, reward, next_obs, done, info)

            obs = next_obs
            episode_reward += reward.mean().item()
            step += 1

        # Actualizar política cada N episodios
        if episode % config['training'].get('update_freq', 32) == 0:
            loss_info = agent.update()
            logger.log_metrics({
                **loss_info,
                'episode_reward': episode_reward,
                'cii': lockguard_info.get('cii', 0),
                'step': step
            }, episode)

        # Logging periódico
        if episode % 50 == 0 or episode == args.total_episodes - 1:
            cii = lockguard_info.get('cii', torch.tensor(0.0)).mean().item()
            print(f"Ep {episode:5d} | Reward: {episode_reward:8.2f} | "
                  f"C-II: {cii:.3f} | Steps: {step}")

            # Guardar mejor modelo
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_episode = episode
                agent.save("checkpoints/best_model")
                print(f"   → Nuevo mejor modelo guardado (Ep {episode})")

        # Guardado periódico
        if episode % args.save_freq == 0 and episode > 0:
            agent.save(f"checkpoints/model_episode_{episode}")

    # ==================== FINAL ====================
    agent.save("checkpoints/lockguard_v4_final")
    
    print("\n🎉 ¡Entrenamiento completado exitosamente!")
    print(f"   Mejor episodio: {best_episode} | Reward: {best_reward:.2f}")
    print(f"   Resultados guardados en: results/run_{timestamp}")


if __name__ == "__main__":
    main()
