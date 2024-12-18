import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

from utils.mobility_data_manager import DataManager


class MobilityDataset(Dataset):
    def __init__(self, df, input_days=7, output_days=7):
        self.input_days = input_days
        self.output_days = output_days
        self.daily_points = 48  # 48 intervalos de 30min por día
        self.df = df

        # Obtener los días totales
        all_days = sorted(self.df["d"].unique())

        # Numero total de días disponibles
        total_days = len(all_days)

        # Numero de secuencias (input_days + output_days) que podemos extraer
        self.max_idx = total_days - (input_days + output_days)

        self.all_days = all_days

    def __len__(self):
        return self.max_idx

    def __getitem__(self, idx):
        # Días de entrada: desde all_days[idx] hasta all_days[idx+input_days-1]
        input_day_range = self.all_days[idx : idx + self.input_days]
        # Días de salida: desde all_days[idx+input_days] hasta all_days[idx+input_days+output_days-1]
        output_day_range = self.all_days[
            idx + self.input_days : idx + self.input_days + self.output_days
        ]

        input_vector = self.build_vector_for_days(input_day_range, is_output=False)
        output_vector = self.build_vector_for_days(output_day_range, is_output=True)

        # input_vector: tensor de tamaño [input_days * daily_points * 3]
        # output_vector: tensor de tamaño [output_days * daily_points * 2]

        return input_vector, output_vector

    def build_vector_for_days(self, days, is_output=False):
        """
        Construye el vector de características para varios días.
        - Si is_output=True, solo (x,y) por ping (48*2*output_days)
        - Si is_output=False, (x,y,missing_flag) por ping (48*3*input_days)
        """

        day_vectors = []
        for d in days:
            day_data = self.df[self.df["d"] == d]

            # Crear una matriz para el día: (48,2) de (x,y)
            # Primero, generamos un array base con (0,0) y luego marcamos faltantes
            day_pings = np.zeros((self.daily_points, 2), dtype=np.float32)
            missing_flags = np.ones(
                (self.daily_points,), dtype=np.float32
            )  # 1: faltante por defecto

            # Cada ping se identifica por el tiempo t (0,30,...,1410)
            # Creamos un map t->index: t_index = t//30
            # Asumimos que t va de 0 a 1410 en intervalos de 30
            for _, row in day_data.iterrows():
                t = int(row["t"])
                idx_t = t // 30
                day_pings[idx_t, 0] = row["x"]
                day_pings[idx_t, 1] = row["y"]
                missing_flags[idx_t] = 0  # hay dato real

            if is_output:
                # Para la salida, solo x,y
                # dim final: (48*2,)
                day_vector = day_pings.flatten()  # (48*2)
            else:
                # Para la entrada, x,y,missing_flag
                # Unir day_pings con missing_flags
                # day_pings: (48,2), missing_flags: (48,)
                # concatenamos a nivel de features: result (48,3)
                day_vector = np.concatenate(
                    [day_pings, missing_flags[:, None]], axis=1
                ).flatten()  # (48*3)

            day_vectors.append(day_vector)

        # Unir todos los días
        all_days_vec = np.concatenate(day_vectors, axis=0)
        return torch.tensor(all_days_vec, dtype=torch.float32)


class SimpleNN(nn.Module):
    def __init__(self, input_days=7, output_days=7, daily_points=48):
        super(SimpleNN, self).__init__()
        # Calcular tamaño de entrada:
        # input_days * (48 pings/dia) * (3 features/ping) = input_days * 48 * 3
        input_size = input_days * daily_points * 3
        # Tamaño de salida:
        # output_days * 48 pings/dia * 2 features/ping = output_days * 48 * 2
        output_size = output_days * daily_points * 2

        hidden_size = 256
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for Xv, Yv in val_loader:
                Xv, Yv = Xv.to(device), Yv.to(device)
                outputs = model(Xv)
                loss = criterion(outputs, Yv)
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        )


train_day = 45
val_day = 60


def split_data_normalized(df):
    train = df[df["d"] < train_day]
    val = df[(df["d"] >= train_day) & (df["d"] < val_day)]
    test = df[df["d"] >= val_day]
    return train, val, test


csv_file = "../Data/cityA_groundtruthdata.csv/cityA_groundtruthdata.csv"
manager = DataManager(
    csv_path=csv_file,
)
df = manager.load_csv()

# Suponiendo que ya dividiste en train_df, val_df, test_df
input_days = 7
output_days = 7


train_df, val_df, test_df = split_data_normalized(df)

train_dataset = MobilityDataset(
    train_df, input_days=input_days, output_days=output_days
)
val_dataset = MobilityDataset(val_df, input_days=input_days, output_days=output_days)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = SimpleNN(input_days=input_days, output_days=output_days)
train_model(model, train_loader, val_loader, epochs=1, lr=0.001)
