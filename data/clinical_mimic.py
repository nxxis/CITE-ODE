"""Data loading and preprocessing utilities for MIMIC-style cohorts.

This module provides a reproducible path from a long-format clinical
table to batched PyTorch sequences with padding and masks. For the
ICDM submission we ship a pre-saved CSV artifact and a deterministic
collate function so reviewers can reproduce the exact training
behaviour without external database access.

The pivoting logic and expected column layout are documented near the
`get_mimic_dataloader` implementation.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from google.cloud import bigquery

class MIMICContinuousDataset(Dataset):
    """Dataset returning per-stay time series and static demographic vars.

    Each item is `(x, t, c, y, d)` where `x` is the sequence of vitals
    (time-major), `t` are hours-since-admission timestamps, `c` are
    per-timestep confounders, `y` is the episode label, and `d` is the
    static demographic vector used in fairness analyses.
    """
    def __init__(self, dataframe):
        self.df = dataframe
        self.stays = self.df['stay_id'].unique()
        self.features = ['heart_rate', 'nbp_mean', 'resp_rate', 'temperature']
        self.confounders = ['rolling_measurement_density_6h', 'minutes_since_last_obs']
        
        # Demographic Fairness Attributes
        self.demographics = ['age', 'gender_encoded']

    def __len__(self):
        return len(self.stays)

    def __getitem__(self, idx):
        stay_data = self.df[self.df['stay_id'] == self.stays[idx]]
        
        x = torch.tensor(stay_data[self.features].values, dtype=torch.float32)
        
        # Normalize time to start at 0 (Hours since admission)
        t_raw = torch.tensor(pd.to_datetime(stay_data['charttime']).astype('int64').values // 10**9, dtype=torch.float32)
        t = (t_raw - t_raw[0]) / 3600.0 
        
        c = torch.tensor(stay_data[self.confounders].values, dtype=torch.float32)
        y = torch.tensor(stay_data['hospital_expire_flag'].iloc[0], dtype=torch.float32)
        
        # Extract protected attributes (Age, Gender) - these are static per patient
        d = torch.tensor(stay_data[self.demographics].iloc[0].values, dtype=torch.float32)
        
        return x, t, c, y, d

def tide_collate_fn(batch):
    # Unpack the demographic vector `d` and include it in the batch
    xs, ts, cs, ys, ds = zip(*batch)
    
    # Pad variable length sequences with 0.0
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    t_padded = pad_sequence(ts, batch_first=True, padding_value=0.0)
    c_padded = pad_sequence(cs, batch_first=True, padding_value=0.0)
    
    # Stack static variables
    y_batch = torch.stack(ys)
    d_batch = torch.stack(ds) 
    
    # Create boolean mask (True where data exists, False where padded)
    lengths = torch.tensor([len(x) for x in xs])
    max_len = lengths.max()
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    
    return x_padded, t_padded, c_padded, y_batch, d_batch, mask

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

# using saved csv for reproducibility
def get_mimic_dataloader(csv_path='data/mimic_cemr_cohort.csv', batch_size=32):
    """Load the pre-saved cohort CSV and return a batched DataLoader.

    To ensure reproducibility for reviewers, this function defaults to
    reading a local CSV artifact. The processing steps (pivot, fill,
    normalization) mirror the data extraction used for our experiments.
    """
    print(f"Loading static clinical cohort from {csv_path}...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: Cannot find the dataset artifact at {csv_path}. Ensure it is downloaded.")
        
    # Read the raw 'long format' data exactly as it was pulled from BigQuery
    df_long = pd.read_csv(csv_path)

    # Pivot long signals into parallel feature columns
    id_mapping = {220045: 'heart_rate', 220181: 'nbp_mean', 220210: 'resp_rate'}
    df_long['vital_name'] = df_long['itemid'].map(id_mapping)

    index_cols = ['subject_id', 'hadm_id', 'stay_id', 'label', 'age', 'gender', 'charttime', 'delta_t', 'obs_density']
    df = df_long.pivot_table(index=index_cols, columns='vital_name', values='valuenum', aggfunc='mean').reset_index()

    # Inject structural temperature filler if missing from cohort
    if 'temperature' not in df.columns:
        df['temperature'] = np.nan

    # Forward/backward fill irregular sampling gaps
    df = df.sort_values(by=['stay_id', 'charttime'])
    vital_cols = ['heart_rate', 'nbp_mean', 'resp_rate', 'temperature']
    df[vital_cols] = df.groupby('stay_id')[vital_cols].ffill().bfill().fillna(0)

    # Align explicit column names for the PyTorch Dataset
    df = df.rename(columns={
        'obs_density': 'rolling_measurement_density_6h',
        'delta_t': 'minutes_since_last_obs',
        'label': 'hospital_expire_flag'
    })

    # --- PROCESS DEMOGRAPHICS ---
    # Binary encoding for gender
    if 'gender' in df.columns and df['gender'].dtype == object:
        df['gender_encoded'] = df['gender'].map({'M': 1.0, 'F': 0.0}).fillna(0.5)
    elif 'gender' in df.columns:
        df['gender_encoded'] = df['gender'].astype(float)
    else:
        df['gender_encoded'] = 0.5 # Fallback if missing

    # Force hardware-level floats for PyTorch
    safe_cols = ['rolling_measurement_density_6h', 'minutes_since_last_obs', 'hospital_expire_flag', 'age', 'gender_encoded']
    for col in safe_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # --- Z-Score Normalization ---
    cols_to_normalize = vital_cols + ['rolling_measurement_density_6h', 'minutes_since_last_obs', 'age', 'gender_encoded']
    for col in cols_to_normalize:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    print(f"Static data processing complete. Matrix shape: {df.shape}")
    
    return DataLoader(MIMICContinuousDataset(df), batch_size=batch_size, shuffle=True, collate_fn=tide_collate_fn, drop_last=True)

# =====================================================================
# DEPRECATED: LIVE DATABASE EXTRACTION LOGIC (DO NOT RUN)
# =====================================================================
# def get_mimic_dataloader(project_id='ANONYMIZED_PROJECT_ID', batch_size=32, limit=50000):
#     """Pull cohort from BigQuery, pivot vitals, and return a DataLoader.

#     Parameters
#     ----------
#     project_id : str
#         GCP project containing the prepared `tide_data.final_tide_dataset` table.
#     batch_size : int
#         DataLoader batch size.
#     limit : int
#         Row limit for the BigQuery read (useful to speed dev runs).
#     """
#     print("Pulling clinical cohorts from BigQuery...")
#     client = bigquery.Client(project=project_id)
#     query = f"SELECT * FROM `{project_id}.tide_data.final_tide_dataset` LIMIT {limit}"
#     df_long = client.query(query).to_dataframe()

#     # Pivot long signals into parallel feature columns
#     # `id_mapping` maps itemids from the clinical table to human names
#     id_mapping = {220045: 'heart_rate', 220181: 'nbp_mean', 220210: 'resp_rate'}
#     df_long['vital_name'] = df_long['itemid'].map(id_mapping)

#     index_cols = ['subject_id', 'hadm_id', 'stay_id', 'label', 'age', 'gender', 'charttime', 'delta_t', 'obs_density']
#     df = df_long.pivot_table(index=index_cols, columns='vital_name', values='valuenum', aggfunc='mean').reset_index()

#     # Inject structural temperature filler if missing from cohort
#     if 'temperature' not in df.columns:
#         df['temperature'] = np.nan

#     # Forward/backward fill irregular sampling gaps, then replace any
#     # remaining NaNs with 0. This is a pragmatic choice for model input.
#     df = df.sort_values(by=['stay_id', 'charttime'])
#     vital_cols = ['heart_rate', 'nbp_mean', 'resp_rate', 'temperature']
#     df[vital_cols] = df.groupby('stay_id')[vital_cols].ffill().bfill().fillna(0)

#     # Align explicit column names for the PyTorch Dataset
#     df = df.rename(columns={
#         'obs_density': 'rolling_measurement_density_6h',
#         'delta_t': 'minutes_since_last_obs',
#         'label': 'hospital_expire_flag'
#     })

#     # --- PROCESS DEMOGRAPHICS ---
#     # Binary encoding for gender. Unknown/missing -> 0.5 placeholder.
#     if 'gender' in df.columns and df['gender'].dtype == object:
#         df['gender_encoded'] = df['gender'].map({'M': 1.0, 'F': 0.0}).fillna(0.5)
#     elif 'gender' in df.columns:
#         df['gender_encoded'] = df['gender'].astype(float)
#     else:
#         df['gender_encoded'] = 0.5 # Fallback if missing

#     # Force hardware-level floats for PyTorch
#     safe_cols = ['rolling_measurement_density_6h', 'minutes_since_last_obs', 'hospital_expire_flag', 'age', 'gender_encoded']
#     for col in safe_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

#     # --- Z-Score Normalization ---
#     # We include demographics in normalization to keep the discriminator
#     # gradients numerically stable during adversarial training.
#     cols_to_normalize = vital_cols + ['rolling_measurement_density_6h', 'minutes_since_last_obs', 'age', 'gender_encoded']
#     for col in cols_to_normalize:
#         df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

#     print(f"Data processing complete. Matrix shape: {df.shape}")
    
#     return DataLoader(MIMICContinuousDataset(df), batch_size=batch_size, shuffle=True, collate_fn=tide_collate_fn, drop_last=True)