#coding: utf-8
import os
import torch
import sqlite3
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, db_path, homography_path):
        """
        Initialise le dataset en chargeant les séquences et l'homographie (sans l'appliquer).
        """
        self.db_path = db_path
        self.base_dir = os.path.dirname(self.db_path)
        
        # charge la matrice d'homographie sans l'appliquer
        if os.path.exists(homography_path):
            self.homography = np.loadtxt(homography_path)
        else:
            print(f"Fichier homographie introuvable : {homography_path}")
            self.homography = np.eye(3)
            
        # chargement des données depuis .db
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM dataset_T_length_20delta_coordinates"
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        
        # groupement par séquences
        self.sequences = dict(tuple(self.df.groupby('data_id')))
        self.data_ids = list(self.sequences.keys())

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        seq_df = self.sequences[data_id]
        seq_df = seq_df.sort_values(by='frame_num')
        
        # trajectoires
        positions = torch.tensor(seq_df[['pos_x', 'pos_y']].values, dtype=torch.float32)
        deltas = torch.tensor(seq_df[['pos_x_delta', 'pos_y_delta']].values, dtype=torch.float32)
        
        # centrage
        start_pos = positions[0].clone()
        positions_norm = positions - start_pos
        
        obs_traj = positions_norm[:8]
        pred_traj_gt = positions_norm[8:]
        obs_deltas = deltas[:8]
        pred_deltas_gt = deltas[8:]
        
        # image scene
        current_frame_num = int(seq_df.iloc[7]['frame_num'])
        img_path = os.path.join(self.base_dir, "visual_data", f"frame{current_frame_num:06d}.jpg")
        
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        else:
            img_tensor = torch.zeros((3, 480, 640), dtype=torch.float32)
        
        return {
            'data_id': data_id,
            'obs_traj': obs_traj,
            'pred_traj_gt': pred_traj_gt,
            'obs_deltas': obs_deltas,
            'pred_deltas_gt': pred_deltas_gt,
            'start_pos': start_pos,
            'scene_image': img_tensor
        }