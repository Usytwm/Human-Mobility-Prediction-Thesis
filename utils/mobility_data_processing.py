from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from multiprocessing import freeze_support
from metrics import (
    DTWMetric,
    GeoBLEUMetric,
    LPPMetric,
    Metric,
)
from mobility_data_manager import DataManager
from scipy.interpolate import CubicSpline


class MobilityDataExtractor:
    """
    Clase para extraer datos de entrenamiento y validación de los conjuntos de datos de movilidad.
    """

    @staticmethod
    def limit_dataset_days(dataset, max_days):
        """
        Limita el conjunto de datos a un número específico de días.

        Args:
            dataset: El conjunto de datos completo.
            max_days: Número máximo de días a considerar.

        Returns:
            Un subconjunto del dataset limitado en días.
        """
        return dataset[:, :max_days, :, :]

    @staticmethod
    def extract_training_data_slice(dataset, num_dias, num_usuarios):
        """
        Extrae una porción de los datos de entrenamiento excluyendo el cuadrante inferior derecho.
        """
        total_usuarios, total_dias, pings_por_dia, coordenadas = dataset.shape

        matriz_sin_cuadrante_inf_derecho = np.full(
            (total_usuarios, total_dias, pings_por_dia, coordenadas),
            np.nan,
            dtype=dataset.dtype,
        )

        matriz_sin_cuadrante_inf_derecho[:num_usuarios, :num_dias] = dataset[
            :num_usuarios, :num_dias
        ]
        return matriz_sin_cuadrante_inf_derecho

    @staticmethod
    def extract_validation_data_slice(dataset, num_dias, num_usuarios):
        """
        Extrae la porción de los datos de validación del cuadrante inferior derecho.
        """
        return dataset[-num_usuarios:, -num_dias:, :, :]

    @staticmethod
    def create_training_validation_sets(dataset, val_dias, usuario_cutoff):
        """
        Crea los conjuntos de entrenamiento y validación a partir del conjunto de datos.
        """
        total_dias = dataset.shape[1]
        train_dias = total_dias - val_dias
        usuarios_restantes = dataset.shape[0] - usuario_cutoff
        training_set = MobilityDataExtractor.extract_training_data_slice(
            dataset, train_dias, usuario_cutoff
        )
        validation_set = MobilityDataExtractor.extract_validation_data_slice(
            dataset, val_dias, usuarios_restantes
        )
        return training_set, validation_set


class Predictor(ABC):
    @abstractmethod
    def predict(self, data, val_dias, usuario_offset):
        pass


class NaiveWeekRepeatPredictor(Predictor):
    def predict(self, data, val_dias, usuario_offset):
        """
        Genera predicciones para la región de validación usando la semana anterior.
        """
        usuarios_restantes = data.shape[0] - usuario_offset
        pings_por_dia, coordenadas = data.shape[2], data.shape[3]
        num_dias = data.shape[1]

        predicciones = np.empty(
            (usuarios_restantes, val_dias, pings_por_dia, coordenadas)
        )
        predicciones[:] = np.nan

        for usuario in range(usuarios_restantes):
            for dia in range(val_dias):
                semana_pasada = dia % 7
                x = usuario_offset + usuario
                y = num_dias - val_dias - 7 + semana_pasada  # Días anteriores correctos
                coord_semana_pasada = data[x, y]
                predicciones[usuario, dia] = self._complete_missing_values(
                    coord_semana_pasada
                )
                predict_actual = predicciones[usuario, dia]
                if np.any(np.isnan(predict_actual)):
                    print("WARNING: Hay valores NaN en las predicciones.")

        return predicciones

    def _complete_missing_values(self, day_data):
        """
        Completa los valores faltantes (NaN) en los datos diarios usando interpolación cúbica
        y regresión polinómica para extrapolación.

        Args:
            day_data: Arreglo de shape (pings_por_dia, coordenadas) con los datos de un día.

        Returns:
            Arreglo completado sin valores NaN, con interpolación cúbica y extrapolación polinómica.
        """
        completed_data = day_data.copy()
        for coord in range(
            completed_data.shape[1]
        ):  # Itera sobre cada coordenada (x e y)
            values = completed_data[:, coord]
            indices = np.arange(len(values))

            # Detectar valores válidos y NaN
            valid_indices = ~np.isnan(values)
            if valid_indices.sum() == 0:  # Caso extremo: todos son NaN
                completed_data[:, coord] = 0  # Asigna cero a todos
                continue

            valid_x = indices[valid_indices]  # Índices de valores válidos
            valid_y = values[valid_indices]  # Valores válidos

            # Interpolación cúbica si hay suficientes puntos
            if len(valid_x) > 1:
                cs = CubicSpline(valid_x, valid_y, extrapolate=False)
                interpolated = cs(indices)
            else:
                # Si solo hay un valor válido, rellenar todo con ese valor
                interpolated = np.full_like(values, valid_y[0])

            # Extrapolación para los extremos NaN (usando regresión lineal simple)
            nan_start = np.isnan(interpolated[: valid_x[0]])  # NaN al inicio
            nan_end = np.isnan(interpolated[valid_x[-1] + 1 :])  # NaN al final

            # Extrapolación al inicio
            if nan_start.any():
                slope_start = (
                    (valid_y[1] - valid_y[0]) / (valid_x[1] - valid_x[0])
                    if len(valid_x) > 1
                    else 0
                )
                interpolated[: valid_x[0]] = valid_y[0] + slope_start * (
                    indices[: valid_x[0]] - valid_x[0]
                )

            # Extrapolación al final
            if nan_end.any():
                slope_end = (
                    (valid_y[-1] - valid_y[-2]) / (valid_x[-1] - valid_x[-2])
                    if len(valid_x) > 1
                    else 0
                )
                interpolated[valid_x[-1] + 1 :] = valid_y[-1] + slope_end * (
                    indices[valid_x[-1] + 1 :] - valid_x[-1]
                )

            # Garantizar que los valores sean mayores o iguales a cero
            interpolated = np.maximum(interpolated, 0)

            # Asignar los valores completados
            completed_data[:, coord] = interpolated

        return completed_data

    def _complete_missing_values_graphics(self, day_data):
        """
        Completa los valores faltantes (NaN) en los datos diarios usando interpolación cúbica
        y regresión lineal para extrapolación, y grafica la trayectoria completada.

        Args:
            day_data: Arreglo de shape (pings_por_dia, coordenadas) con los datos de un día.

        Returns:
            Arreglo completado sin valores NaN, con interpolación cúbica y extrapolación lineal.
        """
        completed_data = day_data.copy()
        for coord in range(
            completed_data.shape[1]
        ):  # Itera sobre cada coordenada (x e y)
            values = completed_data[:, coord]
            indices = np.arange(len(values))

            # Detectar valores válidos y NaN
            valid_indices = ~np.isnan(values)
            if valid_indices.sum() == 0:  # Caso extremo: todos son NaN
                completed_data[:, coord] = 0  # Asigna cero a todos
                continue

            valid_x = indices[valid_indices]  # Índices de valores válidos
            valid_y = values[valid_indices]  # Valores válidos

            # Interpolación cúbica
            if len(valid_x) > 1:
                cs = CubicSpline(valid_x, valid_y, extrapolate=False)
                interpolated = cs(indices)
            else:
                interpolated = np.full_like(values, valid_y[0])

            # Extrapolación lineal para los extremos
            nan_start = np.isnan(interpolated[: valid_x[0]])
            if nan_start.any():
                slope_start = (
                    (valid_y[1] - valid_y[0]) / (valid_x[1] - valid_x[0])
                    if len(valid_x) > 1
                    else 0
                )
                interpolated[: valid_x[0]] = valid_y[0] + slope_start * (
                    indices[: valid_x[0]] - valid_x[0]
                )

            nan_end = np.isnan(interpolated[valid_x[-1] + 1 :])
            if nan_end.any():
                slope_end = (
                    (valid_y[-1] - valid_y[-2]) / (valid_x[-1] - valid_x[-2])
                    if len(valid_x) > 1
                    else 0
                )
                interpolated[valid_x[-1] + 1 :] = valid_y[-1] + slope_end * (
                    indices[valid_x[-1] + 1 :] - valid_x[-1]
                )

            # Garantizar que los valores sean mayores o iguales a cero
            interpolated = np.maximum(interpolated, 0)

            # Asignar valores interpolados
            completed_data[:, coord] = interpolated

        # Graficar la trayectoria completa en el plano (x, y)
        plt.figure(figsize=(8, 6))
        valid_points = ~np.isnan(day_data[:, 0]) & ~np.isnan(day_data[:, 1])

        plt.plot(
            day_data[valid_points, 0],
            day_data[valid_points, 1],
            "ro",
            label="Puntos originales",
        )
        plt.plot(
            completed_data[:, 0],
            completed_data[:, 1],
            # "-o",
            color="blue",
            label="Trayectoria completada",
        )

        plt.title("Movimiento del usuario en el plano (x, y)")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.legend()
        plt.grid()
        plt.show()

        return completed_data


class Formatter:
    @staticmethod
    def __format_trajectory_data(data):
        formatted_data = []
        num_usuarios, num_dias, pings_por_dia, _ = data.shape
        for usuario in range(num_usuarios):
            user_trajectory = []
            for dia in range(num_dias):
                for ping in range(pings_por_dia):
                    point = (
                        dia,
                        ping,
                        data[usuario, dia, ping, 0],
                        data[usuario, dia, ping, 1],
                    )
                    user_trajectory.append(point)
            if user_trajectory:
                formatted_data.append(user_trajectory)
        return formatted_data

    @staticmethod
    def create_comparable_trajectories(predictions, validation):
        formatted_predictions = Formatter.__format_trajectory_data(predictions)
        formatted_validation = Formatter.__format_trajectory_data(validation)
        generated_per_user, reference_per_user = [], []

        for user_idx in range(len(formatted_validation)):
            user_predictions_data = formatted_predictions[user_idx]
            user_validation_data = formatted_validation[user_idx]
            prediction_dict = {
                (entry[0], entry[1]): (entry[2], entry[3])
                for entry in user_predictions_data
            }
            generated, reference = [], []
            for day, hour, ref_x, ref_y in user_validation_data:
                if not (np.isnan(ref_x) or np.isnan(ref_y)):
                    if (day, hour) in prediction_dict:
                        pred_x, pred_y = prediction_dict[(day, hour)]
                        if np.isnan(pred_x) or np.isnan(pred_y):
                            print(f"NaN prediction at {day}, {hour}")
                        generated.append((day, hour, pred_x, pred_y))
                        reference.append((day, hour, ref_x, ref_y))
            generated_per_user.append(generated)
            reference_per_user.append(reference)

        return generated_per_user, reference_per_user


def main(ruta_hdf5, metrics: List[Metric], predictor: Predictor):
    manager = DataManager()
    dataset = manager.load_hdf5(ruta_hdf5)

    # Limitar el dataset a 60 días
    total_dias = 60
    val_dias = 15
    usuario_cutoff = int(dataset.shape[0] - 3000)
    dataset = MobilityDataExtractor.limit_dataset_days(dataset, total_dias)

    # Crear conjuntos de entrenamiento y validación
    training_set, validation_set = (
        MobilityDataExtractor.create_training_validation_sets(
            dataset, val_dias, usuario_cutoff
        )
    )

    # Generar predicciones
    predicciones = predictor.predict(dataset, val_dias, usuario_offset=usuario_cutoff)

    # Crear trayectorias comparables
    generated, reference = Formatter.create_comparable_trajectories(
        predicciones, validation_set
    )

    for metric in metrics:
        score = metric.calculate(generated, reference)
        print(f"{metric.__class__.__name__} score: {score:.2f}")


if __name__ == "__main__":
    freeze_support()
    ruta_hdf5 = "C:\\Brian\\Tesis\\Challenge\\cityA_groundtruthdata.csv\\cityA_groundtruthdata.hdf5"
    metrics = [LPPMetric(), GeoBLEUMetric(), DTWMetric()]
    predictor = NaiveWeekRepeatPredictor()
    main(ruta_hdf5, metrics[2:], predictor)
