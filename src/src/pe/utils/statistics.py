import numpy  as np
import json 
def get_statistics(arr:np.array, output_file:str) -> dict:
    """
    Calculate various statistics for a NumPy array and store them as a JSON file.
    
    Parameters:
    - arr: NumPy array
    - output_file: The name of the JSON file to save the statistics to
    
    Returns a dictionary containing the calculated statistics.
    """
    statistics = {}
    
    statistics['Mean']               = float(np.mean(arr))
    statistics['Standard Deviation'] = float(np.std(arr))
    statistics['Variance']           = float(np.var(arr))
    
    # Calculate IQR
    quartiles = np.percentile(arr, [25, 75])
    statistics['IQR'] = float(quartiles[1] - quartiles[0])
    
    statistics['Minimum'] = float(np.min(arr))
    statistics['Maximum'] = float(np.max(arr))
    
    # Save the statistics as a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(statistics, json_file, indent=4)
    
    return statistics