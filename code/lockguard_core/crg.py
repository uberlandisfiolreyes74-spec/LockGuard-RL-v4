"""
Collective Reality Gate (CRG) - LockGuard RL v4
===============================================

Componente clave que decide si un equipo de agentes puede ejecutar acciones
en el mundo real o debe permanecer en Dream Mode (simulación) para evitar
lock-in colectivo.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CollectiveRealityGate(nn.Module):
    """
    Collective Reality Gate (CRG)
    
    Evalúa el riesgo colectivo (CII) y decide si el equipo entero
    puede pasar de Dream Mode a ejecución real.
    """

    def __init__(self, 
                 beta_c: float = 25.0,
                 theta_exec_team: float = 0.60,
                 device: str = "auto"):
        super().__init__()
        
        self.beta_c = beta_c                    # Steepness del sigmoid
        self.theta_exec_team = theta_exec_team  # Umbral de ejecución colectiva
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else torch.device(device)

    def forward(self, 
                cii_scores: torch.Tensor,
                return_prob: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            cii_scores: Tensor de Collective II-Score
                        Shape: [batch_size] o [batch_size, n_agents]
            return_prob: Si True, retorna también la probabilidad cruda
            
        Returns:
            crg: Tensor binario (0 = Dream Mode, 1 = Reality)
            prob: Probabilidad de ejecutar en realidad (opcional)
        """
        # Si viene por agente, promediamos o tomamos el máximo según necesidad
        if cii_scores.dim() > 1:
            cii_mean = cii_scores.mean(dim=-1)   # Promedio colectivo
        else:
            cii_mean = cii_scores

        # Calculamos la probabilidad de ejecución segura
        prob = torch.sigmoid(self.beta_c * (self.theta_exec_team - cii_mean))
        
        # Decisión binaria (hard threshold en inferencia)
        crg = (prob > 0.5).float()
        
        if return_prob:
            return crg, prob
        return crg, None

    def get_threshold(self) -> float:
        """Retorna el umbral actual de ejecución colectiva"""
        return self.theta_exec_team

    def set_threshold(self, new_threshold: float):
        """Permite ajustar dinámicamente el umbral"""
        self.theta_exec_team = new_threshold


# ========================
# Función de ayuda rápida
# ========================

def create_crg(config: dict) -> CollectiveRealityGate:
    """Crea una instancia de CRG desde configuración"""
    return CollectiveRealityGate(
        beta_c=config.get('lockguard', {}).get('beta_c', 25.0),
        theta_exec_team=config.get('lockguard', {}).get('theta_exec_team', 0.60)
  )
