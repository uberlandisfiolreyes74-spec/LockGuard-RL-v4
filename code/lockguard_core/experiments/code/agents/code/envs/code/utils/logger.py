"""
Multi-Agent Limit Order Book Environment - LockGuard RL v4
==========================================================

Entorno de trading multi-agente basado en Order Book realista.
"""

import gym
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Tuple, Dict, Optional


class MultiAgentLOBEnv(gym.Env):
    """
    Entorno Multi-Agente para ejecución de órdenes en Limit Order Book
    """

    def __init__(self, 
                 n_agents: int = 8,
                 episode_length: int = 360,
                 ticker: str = "BTC/USDT",
                 **kwargs):
        
        super().__init__()
        
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.current_step = 0
        
        # Espacio de observación (por agente)
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(64,),  # Features: spread, imbalance, volatility, etc.
            dtype=np.float32
        )
        
        # Espacio de acciones (por agente)
        self.action_space = spaces.Discrete(11)  # 0=Hold, 1-5=Limit Buy, 6-10=Limit Sell

        # Estado interno
        self.reset()

    def reset(self, seed: Optional[int] = None):
        """Reinicia el episodio"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.ii_scores = np.zeros(self.n_agents)
        
        # Observación inicial
        obs = self._get_observation()
        return obs, {}

    def step(self, actions: np.ndarray):
        """Ejecuta una acción de todos los agentes"""
        self.current_step += 1
        
        # Simular ejecución (placeholder realista)
        rewards = np.zeros(self.n_agents)
        done = self.current_step >= self.episode_length
        
        # Calcular II-Scores (simulado)
        self.ii_scores = self._compute_ii_scores(actions)
        
        # Recompensa = -Implementation Shortfall (simplificado)
        rewards = -np.abs(np.random.normal(0.5, 0.2, self.n_agents)) - self.ii_scores * 2.0
        
        info = {
            'ii_scores': self.ii_scores,
            'cii': self.ii_scores.mean(),
            'step': self.current_step
        }
        
        obs = self._get_observation()
        
        return obs, rewards, done, False, info

    def _get_observation(self):
        """Genera observación para todos los agentes"""
        # Features simuladas del Order Book
        base_obs = np.random.normal(0, 1, 64)
        return base_obs.astype(np.float32)

    def _compute_ii_scores(self, actions: np.ndarray) -> np.ndarray:
        """Calcula Irreversibility Index por agente (simulado)"""
        # Mayor acción = mayor riesgo potencial
        risk = np.abs(actions - 5) / 5.0  # 0 a 1
        volatility = 0.3 + 0.4 * np.random.rand(self.n_agents)
        ii_scores = np.clip(risk * volatility * 1.8, 0.0, 0.95)
        return ii_scores

    def render(self):
        pass

    def close(self):
        pass


# ========================
# Función de ayuda
# ========================

def make_lob_env(config: dict):
    """Crea el entorno desde configuración"""
    return MultiAgentLOBEnv(
        n_agents=config.get('env', {}).get('n_agents', 8),
        episode_length=config.get('env', {}).get('episode_length', 360)
             )
