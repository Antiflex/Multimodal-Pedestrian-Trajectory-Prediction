#coding: utf-8
import os
import torch
import sqlite3
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, db_path, homography_path, transform_coords=False):
        """
        Initialise le dataset en chargeant les séquences et l'homographie.
        """
        self.db_path = db_path
        
        # 1. Chargement de la matrice d'homographie
        # np.loadtxt gère très bien les fichiers texte avec des espaces ou tabulations
        if os.path.exists(homography_path):
            self.homography = np.loadtxt(homography_path)
        else:
            print(f"⚠️ Attention : Fichier homographie introuvable à {homography_path}")
            self.homography = np.eye(3) # Matrice identité par défaut
            
        # 2. Chargement des données depuis SQLite
        conn = sqlite3.connect(self.db_path)
        
        # On suppose que la table s'appelle 'dataset_T_length_20delta_coordinates'
        # comme vu lors de notre exploration.
        query = "SELECT * FROM dataset_T_length_20delta_coordinates"
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        
        # 3. Groupement par 'data_id'
        # Chaque data_id correspond à une séquence complète de 20 frames
        self.sequences = dict(tuple(self.df.groupby('data_id')))
        self.data_ids = list(self.sequences.keys())

    def __len__(self):
        """Retourne le nombre total de séquences disponibles."""
        return len(self.data_ids)

    def __getitem__(self, idx):
        """
        Récupère une séquence spécifique et la sépare en :
        - 8 pas observés (X, Y)
        - 12 pas futurs (X, Y)
        """
        data_id = self.data_ids[idx]
        seq_df = self.sequences[data_id]
        
        # On s'assure que les données sont triées par frame_num pour respecter la chronologie
        seq_df = seq_df.sort_values(by='frame_num')
        
        # Extraction des coordonnées absolues (pour la visualisation)
        # On convertit en tenseurs PyTorch (Float)
        positions = torch.tensor(seq_df[['pos_x', 'pos_y']].values, dtype=torch.float32)
        
        # Extraction des deltas (vitesses relatives, utiles pour l'entraînement)
        deltas = torch.tensor(seq_df[['pos_x_delta', 'pos_y_delta']].values, dtype=torch.float32)
        
        # Découpage 8 -> 12 (Standard evaluation setting)
        obs_traj = positions[:8]        # [8, 2]
        pred_traj_gt = positions[8:]    # [12, 2] (Ground Truth)
        
        obs_deltas = deltas[:8]         # [8, 2]
        pred_deltas_gt = deltas[8:]     # [12, 2]
        
        # On retourne un dictionnaire très propre pour alimenter le modèle
        return {
            'data_id': data_id,
            'obs_traj': obs_traj,
            'pred_traj_gt': pred_traj_gt,
            'obs_deltas': obs_deltas,
            'pred_deltas_gt': pred_deltas_gt,
            'homography': torch.tensor(self.homography, dtype=torch.float32)
        }

from torch.utils.data import DataLoader