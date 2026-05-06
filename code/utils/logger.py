"""
Logger Utility for LockGuard RL v4
==================================

Sistema de logging avanzado para seguimiento de experimentos,
métricas de LockGuard y resultados.
"""

import os
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class Logger:
    """
    Logger personalizado para LockGuard RL v4
    """

    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Archivos de logging
        self.log_file = self.run_dir / "training_log.csv"
        self.metrics_file = self.run_dir / "metrics.json"
        self.config_file = self.run_dir / "config.yaml"
        
        self.episode_data = []
        self.current_metrics = {}

    def log_metrics(self, metrics: Dict[str, Any], episode: int):
        """Registra métricas por episodio"""
        metrics['episode'] = episode
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.episode_data.append(metrics)
        self.current_metrics = metrics

        # Guardar en CSV
        self._save_to_csv(metrics)
        
        # Guardar checkpoint JSON cada 50 episodios
        if episode % 50 == 0:
            self._save_json()

    def _save_to_csv(self, metrics: Dict):
        """Guarda métricas en formato CSV"""
        file_exists = self.log_file.exists()
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                           for k, v in metrics.items()})

    def _save_json(self):
        """Guarda todas las métricas en JSON"""
        with open(self.run_dir / "metrics_history.json", 'w') as f:
            json.dump(self.episode_data, f, indent=2)

    def log_config(self, config: Dict):
        """Guarda la configuración usada"""
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def save_model_info(self, model_path: str, info: Dict):
        """Guarda información del modelo guardado"""
        info['saved_at'] = datetime.now().isoformat()
        with open(self.run_dir / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)

    def print_summary(self):
        """Imprime resumen al final del entrenamiento"""
        if not self.episode_data:
            return
            
        rewards = [ep.get('episode_reward', 0) for ep in self.episode_data]
        cii_values = [ep.get('cii', 0) for ep in self.episode_data]
        
        print("\n" + "="*60)
        print("📊 RESUMEN DEL ENTRENAMIENTO - LockGuard RL v4")
        print("="*60)
        print(f"Total episodios     : {len(rewards)}")
        print(f"Recompensa promedio : {np.mean(rewards):.2f}")
        print(f"Recompensa máxima   : {np.max(rewards):.2f}")
        print(f"C-II promedio       : {np.mean(cii_values):.4f}")
        print(f"Último C-II         : {cii_values[-1]:.4f}")
        print(f"Log guardado en     : {self.run_dir}")
        print("="*60)


# ========================
# Función de ayuda
# ========================

def create_logger(config: dict) -> Logger:
    """Crea logger desde configuración"""
    log_dir = config.get('logging', {}).get('log_dir', 'results/logs')
    return Logger(log_dir=log_dir)
