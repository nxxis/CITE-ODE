import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class MIMICContinuousDataset(Dataset):
    def __init__(self, dataframe):
        self.features = ['heart_rate', 'nbp_mean', 'resp_rate', 'temperature']
        self.confounders = ['rolling_measurement_density_6h', 'minutes_since_last_obs']
        self.demographics = ['age', 'gender_encoded', 'race_encoded']
        self.grouped_dict = {}
        
        for stay_id, group in dataframe.groupby('stay_id'):
            if len(group) > 60:
                resampled_indices = np.linspace(0, len(group) - 1, 60, dtype=int)
                group = group.iloc[resampled_indices]
            self.grouped_dict[stay_id] = group
        self.stays = list(self.grouped_dict.keys())

    def __len__(self): return len(self.stays)
    def __getitem__(self, idx):
        stay_data = self.grouped_dict[self.stays[idx]]
        x = torch.tensor(stay_data[self.features].values, dtype=torch.float32)
        t = torch.tensor(stay_data['elapsed_hours'].values, dtype=torch.float32)
        c = torch.tensor(stay_data[self.confounders].values, dtype=torch.float32)
        y = torch.tensor(stay_data['hospital_expire_flag'].iloc[0], dtype=torch.float32)
        d = torch.tensor(stay_data[self.demographics].iloc[0].values, dtype=torch.float32)
        return x, t, c, y, d

def tide_collate_fn(batch):
    xs, ts, cs, ys, ds = zip(*batch)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    t_padded = pad_sequence(ts, batch_first=True, padding_value=0.0)
    c_padded = pad_sequence(cs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(x) for x in xs])
    mask = torch.arange(lengths.max()).expand(len(lengths), lengths.max()) < lengths.unsqueeze(1)
    return x_padded, t_padded, c_padded, torch.stack(ys), torch.stack(ds), mask

def get_mimic_dataloader(csv_path='data/mimic_cemr_cohort.csv', batch_size=64):
    df_long = pd.read_csv(csv_path)
    id_mapping = {220045: 'heart_rate', 220181: 'nbp_mean', 220210: 'resp_rate'}
    if 'vital_name' not in df_long.columns:
        df_long['vital_name'] = df_long['itemid'].map(id_mapping)
    df = df_long.pivot_table(index=['subject_id', 'hadm_id', 'stay_id', 'label', 'age', 'gender', 'raw_race', 'charttime', 'delta_t', 'obs_density'], columns='vital_name', values='valuenum', aggfunc='mean').reset_index()
    if 'temperature' not in df.columns: df['temperature'] = np.nan
    df = df.sort_values(by=['stay_id', 'charttime'])
    
    df['epoch_seconds'] = pd.to_datetime(df['charttime'], errors='coerce').astype('int64') // 10**9
    df['elapsed_hours'] = df.groupby('stay_id')['epoch_seconds'].transform(lambda x: (x - x.iloc[0]) / 3600.0)
    vital_cols = ['heart_rate', 'nbp_mean', 'resp_rate', 'temperature']
    df[vital_cols] = df.groupby('stay_id')[vital_cols].ffill().bfill().fillna(0)
    df = df.rename(columns={'obs_density': 'rolling_measurement_density_6h', 'delta_t': 'minutes_since_last_obs', 'label': 'hospital_expire_flag'})
    df['gender_encoded'] = df['gender'].map({'M': 1.0, 'F': 0.0}).fillna(0.5)

    def standardize_race(r):
        r = str(r).upper()
        if 'WHITE' in r: return 0.0
        elif 'BLACK' in r: return 1.0
        elif 'HISPANIC' in r or 'LATINO' in r: return 2.0
        elif 'ASIAN' in r: return 3.0
        else: return 4.0
    df['race_encoded'] = df['raw_race'].apply(standardize_race)
    
    safe_cols = ['rolling_measurement_density_6h', 'minutes_since_last_obs', 'hospital_expire_flag', 'age', 'gender_encoded', 'race_encoded']
    for col in safe_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
    for col in vital_cols + ['rolling_measurement_density_6h', 'minutes_since_last_obs', 'age']:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    return DataLoader(MIMICContinuousDataset(df), batch_size=batch_size, shuffle=True, collate_fn=tide_collate_fn, drop_last=True)
