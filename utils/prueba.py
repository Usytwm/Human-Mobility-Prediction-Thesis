def temporal_train_test_split(data, test_size=0.2):
    """
    Divide los datos temporalmente en train y test según el porcentaje test_size.
    Asume que el eje de los días es el segundo eje de data: (N_usuarios, N_dias, ...)

    Args:
        data: Array con forma (N_usuarios, N_dias, ...)
        test_size: float, porcentaje del total de días que se destinará a test.

    Returns:
        train_data, test_data
    """
    N_dias = data.shape[1]
    N_test = int(N_dias * test_size)
    N_train = N_dias - N_test

    train_data = data[:, :N_train, ...]
    test_data = data[:, N_train:, ...]
    return train_data, test_data


# Ejemplo de uso:
# Supongamos data con shape (N_usuarios, 75, N_intervalos, 2)
# Primero separamos entre train y test
# train_data_full, test_data = temporal_train_test_split(data, test_size=0.2)

# # Ahora, separas el train_data_full en train y validation
# train_data, val_data = temporal_train_test_split(train_data_full, test_size=0.25)
# Aquí 0.25 es el 25% del train_data_full para validación,
# lo que equivale al 20% (0.25 * 0.8 = 0.2) del total original si sumas proporciones.
