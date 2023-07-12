from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standardize(data, scaler_type='standard'):
    """
    Standardize or normalize the data using a specified scaler type.

    Parameters:
    data (ndarray): The array containing the data to be standardized.
    scaler_type (str): The type of scaler to be used. Options: 'standard' (default), 'minmax'.

    Returns:
    standardized_data (ndarray): The standardized or normalized data.
    """
    if scaler_type == 'standard':
        for i in range(data.shape[0]):
            data[i] = StandardScaler().fit_transform(data[i])
    if scaler_type == 'minmax':
        for i in range(data.shape[0]):
            data[i] = MinMaxScaler().fit_transform(data[i])
    return data
