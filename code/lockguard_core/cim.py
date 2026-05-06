"""
Causal Irreversibility Module (CIM) - LockGuard RL v4
====================================================

Módulo responsable de la atribución causal de irreversibilidad colectiva.
Calcula:
- Collective II-Score (CII)
- Average Causal Effect (ACE) por agente
- Synergy entre pares de agentes (emergencia)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import numpy as np


class CausalIrreversibilityModule(nn.Module):
    """
    Causal Irreversibility Module (CIM)
    """

    def __init__(self, 
                 n_agents: int = 8,
                 kappa: float = 0.4,
                 rho_threshold: float = 0.3,
                 device: str = "auto"):
        super().__init__()
        
        self.n_agents = n_agents
        self.kappa = kappa
        self.rho_threshold = rho_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else torch.device(device)

    def compute_cii(self, 
                    ii_scores: torch.Tensor, 
                    rho_matrix: torch.Tensor) -> torch.Tensor:
        """
        Calcula el Collective Irreversibility Index (CII)
        
        Args:
            ii_scores: Tensor [batch_size, n_agents] o [n_agents]
            rho_matrix: Matriz de correlación entre agentes [n_agents, n_agents]
            
        Returns:
            cii: Collective II-Score
        """
        if ii_scores.dim() == 1:
            ii_scores = ii_scores.unsqueeze(0)  # Añadir batch dimension

        batch_size = ii_scores.shape[0]

        # Máximo individual
        max_ii = ii_scores.max(dim=-1)[0]

        # Término de interacción (emergencia)
        interaction = torch.zeros(batch_size, device=self.device)
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                if rho_matrix[i, j] > self.rho_threshold:
                    interaction += (rho_matrix[i, j] * 
                                  ii_scores[:, i] * 
                                  ii_scores[:, j])

        cii = max_ii + self.kappa * interaction
        return cii.squeeze()

    def compute_ace(self, ii_scores: torch.Tensor) -> torch.Tensor:
        """
        Calcula Average Causal Effect (ACE) simplificado por agente
        (Versión aproximada - se puede reemplazar por estimador más avanzado)
        """
        # Versión simplificada: contribución proporcional al II individual
        ace = ii_scores * 0.75  # Factor de causalidad base
        return ace

    def compute_pair_synergy(self, 
                           ii_scores: torch.Tensor, 
                           rho_matrix: torch.Tensor) -> torch.Tensor:
        """
        Calcula sinergia causal entre pares de agentes
        """
        synergy = torch.zeros((self.n_agents, self.n_agents), device=self.device)
        
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                if rho_matrix[i, j] > self.rho_threshold:
                    # Efecto combinado vs suma individual
                    combined = ii_scores[i] * ii_scores[j] * rho_matrix[i, j]
                    individual = ii_scores[i] + ii_scores[j]
                    synergy[i, j] = combined - individual * 0.5
                    synergy[j, i] = synergy[i, j]
        
        return synergy

    def forward(self, 
                ii_scores: torch.Tensor, 
                rho_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo del módulo CIM
        """
        cii = self.compute_cii(ii_scores, rho_matrix)
        ace = self.compute_ace(ii_scores)
        synergy = self.compute_pair_synergy(ii_scores, rho_matrix)

        return {
            'cii': cii,
            'ace': ace,
            'pair_synergy': synergy,
            'max_ii': ii_scores.max(dim=-1)[0] if ii_scores.dim() > 1 else ii_scores.max()
        }


# ========================
# Función de ayuda
# ========================

def create_cim(config: dict) -> CausalIrreversibilityModule:
    """Crea instancia de CIM desde configuración"""
    return CausalIrreversibilityModule(
        n_agents=config.get('env', {}).get('n_agents', 8),
        kappa=config.get('lockguard', {}).get('kappa', 0.4),
        rho_threshold=0.3
              )
