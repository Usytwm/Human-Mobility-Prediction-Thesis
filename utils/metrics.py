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


def calculate_lpp(predictions_per_user, validation_per_user):
    """
    Calcula la Precisión de Predicción de Ubicación (LPP), ignorando los valores NaN.

    Args:
        predictions_per_user (list of list of tuples): Lista de trayectorias generadas por cada usuario.
        validation_per_user (list of list of tuples): Lista de trayectorias reales por cada usuario.

    Returns:
        float: La precisión de ubicación como porcentaje de coincidencias exactas.
    """
    correct_predictions = 0
    total_predictions = 0

    for predicted_traj, actual_traj in zip(predictions_per_user, validation_per_user):
        for predicted_point, actual_point in zip(predicted_traj, actual_traj):
            # Asegurar que las coordenadas reales no sean NaN
            if not any(np.isnan(coord) for coord in actual_point[2:]):
                # Comparar puntos completos: (d, t, x, y)
                if predicted_point == actual_point:
                    correct_predictions += 1
                total_predictions += 1

    # Calcular el porcentaje de precisión
    lpp = (
        (correct_predictions / total_predictions) * 100
        if total_predictions > 0
        else 0.0
    )
    return lpp


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
