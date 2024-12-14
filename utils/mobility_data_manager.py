import os
import pandas as pd
import gzip
import h5py
import numpy as np


class DataManager:
    """
    Clase para manejar la carga, guardado y consulta de datos de movilidad humana.
    """

    def __init__(self, csv_path=None, hdf5_path=None):
        """
        Inicializa el gestor con las rutas opcionales de los archivos CSV y HDF5.

        Parámetros:
        ---
        - csv_path (str): Ruta al archivo CSV comprimido.
        - hdf5_path (str): Ruta al archivo HDF5.
        """
        self.csv_path = csv_path
        self.hdf5_path = hdf5_path
        self.dataset = None

    def save_to_hdf5(self, csv_path=None, hdf5_path=None):
        """
        Convierte un archivo CSV comprimido en un archivo HDF5 estructurado.

        Parámetros:
        ---
        - csv_path (str): Ruta al archivo CSV comprimido.
        - hdf5_path (str): Ruta para guardar el archivo HDF5.

        Retorna:
        ---
        - str: Ruta del archivo HDF5 guardado.
        """
        csv_path = csv_path or self.csv_path
        if not csv_path:
            raise ValueError("Se debe proporcionar una ruta válida al archivo CSV.")

        with gzip.open(csv_path, "rt", encoding="ISO-8859-1") as f:
            df = pd.read_csv(f, on_bad_lines="skip")

        num_usuarios = df["uid"].max() + 1
        num_dias = df["d"].max() + 1
        pings_por_dia = 48

        hdf5_path = hdf5_path or os.path.splitext(csv_path)[0] + ".hdf5"
        with h5py.File(hdf5_path, "w") as f:
            dataset = f.create_dataset(
                "mobility_matrix",
                (num_usuarios, num_dias, pings_por_dia, 2),
                dtype="f",
                fillvalue=np.nan,
            )
            self._populate_dataset(dataset, df, True)
        self.hdf5_path = hdf5_path
        print(f"Archivo HDF5 guardado en: {hdf5_path}")
        return hdf5_path

    def load_hdf5(self, hdf5_path=None):
        """
        Carga un archivo HDF5 para acceder a los datos.

        Parámetros:
        ---
        - hdf5_path (str): Ruta al archivo HDF5.

        Retorna:
        ---
        - h5py.Dataset: Dataset con los datos de movilidad.
        """
        hdf5_path = hdf5_path or self.hdf5_path
        if not hdf5_path or not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                "Ruta al archivo HDF5 no válida o archivo no encontrado."
            )
        self.dataset = h5py.File(hdf5_path, "r")["mobility_matrix"]
        return self.dataset

    def load_csv(self, csv_path=None):
        """
        Carga un archivo CSV comprimido en un DataFrame de Pandas.

        Parámetros:
        ---
        - csv_path (str): Ruta al archivo CSV comprimido.

        Retorna:
        ---
        - pd.DataFrame: DataFrame con los datos de movilidad.
        """
        csv_path = csv_path or self.csv_path
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(
                "Ruta al archivo CSV no válida o archivo no encontrado."
            )
        with gzip.open(csv_path, "rt", encoding="ISO-8859-1") as f:
            return pd.read_csv(f, on_bad_lines="skip")

    def _populate_dataset(self, dataset, df, autocomplete=False):
        """
        Rellena el dataset HDF5 con datos desde el DataFrame.

        Parámetros:
        ---
        - dataset (h5py.Dataset): Dataset a rellenar.
        - df (pd.DataFrame): DataFrame con los datos de movilidad.
        """
        print("Init population...")
        for row in df.itertuples(index=False):
            dataset[row.uid, row.d, row.t] = [row.x, row.y]
        print("end population")

    def close_dataset(self, dataset):
        """
        Cierra el archivo HDF5.

        Parámetros:
        ---
        - dataset (h5py.Dataset): Dataset que se va a cerrar.
        """
        if hasattr(dataset, "file"):
            dataset.file.close()


if __name__ == "__main__":
    manager = DataManager(
        csv_path="C:\Brian\Tesis\Challenge\cityA_groundtruthdata.csv\cityA_groundtruthdata.csv",
        hdf5_path="C:\Brian\Tesis\Challenge\cityA_groundtruthdata.csv\cityA_groundtruthdata.hdf5",
    )
    hdf5_path = manager.save_to_hdf5()
    dataset = manager.load_hdf5(hdf5_path)
    coords = manager.query_data(user_id=0, day=0, timeslot=1, dataset=dataset)
    print(f"Coordenadas: {coords}")
    manager.close_dataset(dataset)
