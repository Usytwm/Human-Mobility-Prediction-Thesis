from abc import ABC, abstractmethod
import geobleu
import numpy as np


class Metric(ABC):
    """
    Clase base abstracta para diferentes métricas.
    """

    @abstractmethod
    def calculate(self, predictions, validation):
        """
        Calcula el puntaje de la métrica basada en las predicciones y los datos de validación.

        Args:
            predictions: Los datos predichos.
            validation: Los datos de validación.

        Returns:
            El puntaje calculado.
        """
        pass


class LPPMetric(Metric):
    """
    Métrica para calcular la Precisión de Predicción de Ubicación (LPP).
    """

    def calculate(self, predictions, validation):
        return calculate_lpp(validation, predictions)


class MAEMetric(Metric):
    """
    Métrica para calcular el Error Medio Absoluto (MAE).
    """

    def calculate(self, predictions_per_user, validation_per_user):
        return calculate_error_metrics(predictions_per_user, validation_per_user)["MAE"]


class GeoBLEUMetric(Metric):
    """
    Métrica para calcular el puntaje GeoBLEU.
    """

    def calculate(self, predictions_per_user, validation_per_user):
        return calculate_geobleu_for_quadrant(predictions_per_user, validation_per_user)


class DTWMetric(Metric):
    """
    Métrica para calcular el puntaje de Dynamic Time Warping (DTW).
    """

    def calculate(self, predictions_per_user, validation_per_user):
        return calculate_dtw_for_quadrant(predictions_per_user, validation_per_user)


def calculate_lpp(predictions_per_user, validation_per_user, tolerance=4.0):
    """
    Calcula la Precisión de Predicción de Ubicación (LPP), ignorando los valores NaN,
    y permitiendo un margen de error definido por un umbral.

    Args:
        predictions_per_user (list of list of tuples): Lista de trayectorias generadas por cada usuario.
        validation_per_user (list of list of tuples): Lista de trayectorias reales por cada usuario.
        tolerance (float): Margen de error en unidades de distancia para considerar una predicción correcta.

    Returns:
        float: La precisión de ubicación como porcentaje de coincidencias dentro del margen.
    """
    correct_predictions = 0
    total_predictions = 0

    for predicted_traj, actual_traj in zip(predictions_per_user, validation_per_user):
        for predicted_point, actual_point in zip(predicted_traj, actual_traj):
            # Asegurar que las coordenadas reales no sean NaN
            if not any(np.isnan(coord) for coord in actual_point[2:]):
                # Calcular distancia euclidiana entre puntos predichos y reales
                distance = np.sqrt(
                    (predicted_point[2] - actual_point[2]) ** 2
                    + (predicted_point[3] - actual_point[3]) ** 2
                )
                # Contar como correcta si la distancia está dentro del margen de tolerancia
                if distance <= tolerance:
                    correct_predictions += 1
                total_predictions += 1

    # Calcular el porcentaje de precisión
    lpp = (
        (correct_predictions / total_predictions) * 100
        if total_predictions > 0
        else 0.0
    )
    return lpp


def calculate_error_metrics(predictions_per_user, validation_per_user):
    """
    Calcula métricas de error entre trayectorias predichas y reales:
    - Error Medio Absoluto (MAE)
    - Error Máximo
    - Distribución de Errores

    Args:
        predictions_per_user (list of list of tuples): Trayectorias predichas [(d, t, x, y)].
        validation_per_user (list of list of tuples): Trayectorias reales [(d, t, x, y)].

    Returns:
        dict: Métricas calculadas.
    """
    errors = []

    for predicted_traj, actual_traj in zip(predictions_per_user, validation_per_user):
        for predicted_point, actual_point in zip(predicted_traj, actual_traj):
            # Ignorar puntos donde las coordenadas reales sean NaN
            if not any(np.isnan(coord) for coord in actual_point[2:]):
                # Calcular distancia euclidiana
                distance = np.sqrt(
                    (predicted_point[2] - actual_point[2]) ** 2
                    + (predicted_point[3] - actual_point[3]) ** 2
                )
                errors.append(distance)

    # Calcular métricas
    mae = np.mean(errors) if errors else 0.0
    max_error = np.max(errors) if errors else 0.0

    # Retornar resultados
    return {"MAE": mae, "Max Error": max_error, "Errors": errors}


def calculate_geobleu_for_quadrant(predictions_per_user, validation_per_user):
    """
    Calcula el puntaje promedio de GeoBLEU para cada usuario comparando las trayectorias generadas y de referencia.

    Args:
        predictions_per_user (list of list of tuples): Lista de trayectorias generadas por cada usuario.
        validation_per_user (list of list of tuples): Lista de trayectorias de referencia por cada usuario.

    Returns:
        float: El puntaje promedio de GeoBLEU para todos los usuarios comparados.
    """
    total_score = 0
    valid_users = 0
    for predictions, validation in zip(predictions_per_user, validation_per_user):
        if predictions and validation:  # Aseguramos que haya datos comparables
            score = geobleu.calc_geobleu(predictions, validation, processes=3)
            total_score += score
            valid_users += 1
    return total_score / valid_users if valid_users else 0


def calculate_dtw_for_quadrant(predictions_per_user, validation_per_user):
    """
    Calcula el puntaje promedio de DTW para cada usuario comparando las trayectorias generadas y de referencia.

    Args:
        predictions_per_user (list of list of tuples): Lista de trayectorias generadas por cada usuario.
        validation_per_user (list of list of tuples): Lista de trayectorias de referencia por cada usuario.

    Returns:
        float: El puntaje promedio de DTW para todos los usuarios comparados.
    """
    total_score = 0
    valid_users = 0
    for predictions, validation in zip(predictions_per_user, validation_per_user):
        if predictions and validation:  # Aseguramos que haya datos comparables
            score = geobleu.calc_dtw(predictions, validation)
            total_score += score
            valid_users += 1
    return total_score / valid_users if valid_users else 0
