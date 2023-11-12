#!/usr/bin/env python

from functools import partial
from glob import glob
from itertools import combinations
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.preprocessing import RobustScaler, PowerTransformer
import string
from sys import argv
from tqdm import tqdm
from time import time

def read_and_extract_results(file_path, fp_path, predict=False):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    fps = pd.read_csv(fp_path)

    #Standardize df and fps, fps may have some NaNs in it
    fps.dropna(axis=0, how='any', inplace=True)
    idx_col = df.columns[0]

    # Merge the two DataFrames based on their SMILES columns
    merged_df = df.merge(fps, on=idx_col, how='inner')

    # Reorder the rows in df to match the order of rows in the merged DataFrame
    df = df[df[idx_col].isin(merged_df[idx_col])]
    fps = fps[fps[idx_col].isin(merged_df[idx_col])]

    # Sort both DataFrames based on the common column ('SMILES')
    df.sort_values(by=idx_col, inplace=True)
    fps.sort_values(by=idx_col, inplace=True)

    # Reset the index for both DataFrames to ensure it's sequential
    df.reset_index(drop=True, inplace=True)
    fps.reset_index(drop=True, inplace=True)
    if predict:
        return df, fps

    # Initialize a dictionary to store endpoint results
    endpoint_results = {}

    # Iterate through the columns to find endpoint names and their results
    for col in df.columns:
        if col.endswith(' True'):
            endpoint_name = col.replace(' True', '')
            observed_results = df[col]
            observed_results = observed_results.dropna()
            # Find the position of the observed column
            observed_position = df.columns.get_loc(col)
            
            # Find the position of the next 'True' column
            next_true_columns = [col for col in df.columns[observed_position + 1:] if col.endswith(' True')]

            if next_true_columns:
                next_true_position = df.columns.get_loc(next_true_columns[0])
                
                # Get the predicted columns in between
                predicted_columns = df.iloc[:, observed_position + 1:next_true_position]
            else:
                # If there are no more 'True' columns, include all remaining columns as predicted
                predicted_columns = df.iloc[:, observed_position + 1:]

            predicted_results = predicted_columns.loc[observed_results.index]
            endpoint_results[endpoint_name] = {
                'observed': observed_results,
                'predicted': predicted_results
            }
    return endpoint_results, fps


def calculate_metrics(observed, predicted, fps):

    use_fps = fps.loc[observed.index]
    fas, fa, feats = perform_mixed_data_fa(use_fps)

    model_metrics = {}  # Dictionary to store metrics for each model
    model_metrics['FA_info'] = [fa, feats, fas]

    for model_column in predicted.columns:
        model_predicted = predicted[model_column]

        # Check for NaN values in the current model's predicted results
        nan_mask = model_predicted.isna()

        # Calculate coverage as the fraction of rows that are not NaN
        coverage = 1.0 - (nan_mask.sum() / len(nan_mask))

        # Filter observed and predicted results to exclude NaN values
        observed_filtered = observed[~nan_mask]
        predicted_filtered = model_predicted.dropna()

        balanced_accuracy = balanced_accuracy_score(observed_filtered, predicted_filtered)

        sensitivity = recall_score(observed_filtered, predicted_filtered, pos_label=1)
        specificity = recall_score(observed_filtered, predicted_filtered, pos_label=0)

        predictivity = 0.7 * balanced_accuracy + 0.3 * (1 - abs(sensitivity - specificity))

        # Calculate modified predictivity as 70% balanced accuracy + 30% sensitivity
        modified_predictivity = 0.7 * balanced_accuracy + 0.3 * sensitivity

        # Calculate Score1 from the CERAPP work, which is Bal Acc * coverage
        score = balanced_accuracy * coverage

        # Calculate the kNN average distances for the FAs
        model_fas = fas.loc[observed_filtered.index]
        knn_dists = knn_distances(model_fas)
        knn_dists[0][knn_dists[0] > knn_dists[0].quantile(0.95)] = np.nan
        threshold = knn_dists[0].quantile(0.95)

        model_metrics[model_column] = {
            'BA': balanced_accuracy,
            'Pred': predictivity,
            'ModP': modified_predictivity,
            'AD': coverage,
            'Score1': score,
            'kNN': knn_dists,
            'FA': threshold
        }
    return model_metrics


def perform_mixed_data_fa(base_fps, fa=None, n_components=7):
    fps = base_fps[base_fps.columns[2:]]

    #Remove non-variant and highly correlated columns
    fps = fps.loc[:, fps.var() > 0]
    fps = remove_highly_correlated_columns(fps)
    feats = fps.columns

    # Perform FA with the specified explained variance threshold
    tot_var = fps.var().sum()
    for components in range(3, 101):
        fa = FactorAnalysis(n_components=components)
        fa_components = fa.fit_transform(fps)
        explained_var = (tot_var - sum(fa.noise_variance_)) / tot_var
        if explained_var > 0.98:
            break
    # Create a DataFrame with the FA components
    fa_df = pd.DataFrame(data=fa_components, index=base_fps.index, columns=[f'FA_{i+1}' for i in range(components)])

    return fa_df, fa, feats


def remove_highly_correlated_columns(df, threshold=0.95):
    # Convert DataFrame to Numpy array
    data = df.to_numpy()

    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)

    # Create a mask to identify highly correlated features
    mask = (corr_matrix >= threshold) & (corr_matrix < 1.0)

    # List of columns to keep
    columns_to_keep = set()

    # Iterate through the columns
    for i, col in enumerate(df.columns):
        if col not in columns_to_keep:
            # Get the highly correlated columns with col
            correlated_indices = np.where(mask[i])[0]

            # Keep col and remove others from the set
            columns_to_keep.add(col)
            columns_to_keep.difference_update(df.columns[correlated_indices])

    # Create a new DataFrame with only the selected columns
    df_filtered = df[list(columns_to_keep)]
    return df_filtered


def knn_distances(fa_data, other_data=None, k=12):
    """
    Implement a k-nearest neighbors (kNN) approach for consensus modeling.

    Parameters:
        fa_data (pd.DataFrame): DataFrame containing FactorAnalysis components.
        k (int): Number of nearest neighbors to consider (default is 7).

    Returns:
        pd.Series: knn average distances.
    """
    # Convert DataFrame to Numpy array
    fa_array = fa_data.to_numpy()

    # Calculate pairwise distances between data points
    nn = knn(n_neighbors=k)
    nn.fit(fa_array)

    if other_data is None:
        distances, indices = nn.kneighbors()
        test_array = fa_array
        test_index = fa_data.index
    else:
        other_array = other_data.to_numpy()
        distances, indices = nn.kneighbors(X=other_array)
        test_array = other_array
        test_index = other_data.index

    average_distances = []

    for i, data_point in enumerate(test_array):
        # Find k-nearest neighbors and their distances
        neighbor_distances = distances[i]

        # Calculate weights based on distances (inverse of distances)
        average_distances.append(np.mean(neighbor_distances))


    # Convert the weighted consensus list to a DataFrame
    return pd.DataFrame(data=average_distances, index=test_index)


def test_endpoint_combinations(observed, predicted, metrics, num_processes=10):
    observed = observed.astype(int)
    models = list(predicted.columns)
    indices = ['Combination'] + models + ['# of models', 'AD', 'AD(%)', 'ModP',
                                          'BA', 'Pred', 'BA*AD', 'TP', 'FN',
                                          'TN', 'FP', 'Sens', 'Spec', 'Base',
                                          'Max', 'Pareto']
    pool = multiprocessing.Pool(processes=num_processes)
    models = list(predicted.columns)
    all_combinations = [list(comb) for i in range(len(models)) for comb in combinations(models, i + 1)]

    partial_test_combination = partial(test_combinations, observed=observed,
                                       predicted=predicted, metrics=metrics,
                                       indices=indices)
    # Use parallel processing to test all combinations
    results = pool.map(partial_test_combination, all_combinations)
    results = pd.concat(results, axis=1).T.reset_index(drop=True)
    return results


def test_combinations(combination, observed, predicted, metrics, indices):
    use = pd.DataFrame(index=indices, columns=[0])
    combinations = ['Majority', 'BA', 'Pred', 'ModP', 'Score1', 'kNN']
    pred_results = predicted[combination]
    if len(combination) == 1:
        combinations = ['']
        use[0]['Base'] = 'Y'
    if len(combination) == len(use) - 17:
        use[0]['Max'] = 'Y'
    for model in combination:
        use[0][model] = 'Y'
    use[0]['# of models'] = len(combination)
    all_combos = []
    for combo in combinations:
        base = use.copy()
        if combo == '':
            results = pred_results[model]
        else:
            results = combine_pred_results(pred_results, combo, metrics)
            base[0]['Combination'] = combo
        results = results.dropna().astype(int)
        obs = observed.loc[results.index]
        base[0]['AD'] = results.shape[0]
        base[0]['AD(%)'] = base[0]['AD'] / observed.shape[0]
        
        base[0]['Sens'] = recall_score(obs, results, pos_label=1)
        base[0]['Spec'] = recall_score(obs, results, pos_label=0)
        base[0]['BA'] = balanced_accuracy_score(obs, results)
        base[0]['Pred'] = 0.7 * base[0]['BA'] + 0.3 * (1 - abs(base[0]['Sens'] -  base[0]['Spec']))
        base[0]['ModP'] = 0.7 * base[0]['BA'] + 0.3 * base[0]['Sens']
        base[0]['BA*AD'] = base[0]['BA'] * base[0]['AD(%)']

        tn, fp, fn, tp = confusion_matrix(obs, results).ravel()

        for label, val in zip(['TN', 'FP', 'FN', 'TP'], [tn, fp, fn, tp]):
            base[0][label] = val
        all_combos.append(base)
    results = pd.concat(all_combos, axis=1)
    return results


def combine_pred_results(pred, combo, metrics, knn=None):
    if combo in ['Majority', 'BA', 'Pred', 'ModP', 'Score1']:
        try:
            weights = [metrics[x][combo] for x in pred.columns]
        except KeyError:
            weights = [1] * pred.shape[1]
        max_val = sum(weights)
        results = pred.apply(lambda x: calculate_weighted_average(x, weights, max_val), axis=1)
    if combo == 'kNN':
        if knn is None:
            mean_knn = pd.concat([metrics[x][combo] for x in pred.columns], axis=1)
        else:
            mean_knn = knn
        if mean_knn.shape[0] < pred.shape[0]:
            missing_idx = sorted(set(pred.index).difference(set(mean_knn.index)))
            new_rows = pd.DataFrame(index=missing_idx)
            mean_knn = pd.concat([mean_knn, new_rows])
        results = pred.apply(lambda x: knn_weighted(x, mean_knn.loc[x.name]), axis=1)
    return results


def calculate_weighted_average(predictions, weights, max_value):
    """
    Calculate a weighted average while handling NaN values.

    Args:
        predictions (list): List of predicted values.
        weights (list): List of weights corresponding to predictions.
        max_value (float): Maximum possible value.

    Returns:
        int or np.nan: Weighted average or np.nan if the sum of weights is less than half of max_value.
    """
    non_nan_pairs = [(pred, weight) for pred, weight in zip(predictions, weights) if not np.isnan(pred)]
    ad = np.fromiter((weight for _, weight in non_nan_pairs), float)

    if ad.sum() / max_value < 0.5:
        return np.nan

    pred = np.fromiter((pred * weight for pred, weight in non_nan_pairs), float)
    return int(pred.sum() / max_value + 0.5)


def knn_weighted(predictions, mean_knn):
    if all(np.isnan(mean_knn)):
        return np.nan
    if all(np.isnan(predictions)):
        return np.nan
    non_nan_pairs = [(pred, weight) for pred, weight in zip(predictions, mean_knn) if not (np.isnan(pred) or np.isnan(weight))]

    # Inverse the weights
    inv_weights = [1.0 / (weight + 1) for _, weight in non_nan_pairs]

    # Normalize the inverses to add up to 1
    total_inv_weight = sum(inv_weights)
    normalized_weights = [inv_weight / total_inv_weight for inv_weight in inv_weights]

    # Calculate the weighted average
    weighted_sum = sum(pred * inv_weight for (pred, _), inv_weight in zip(non_nan_pairs, normalized_weights))

    return int(weighted_sum + 0.5)


def pareto_front(costs, return_mask=True):
    """
    Find the Pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of Pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise, it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]

    next_point_index = 0  # Next index in the is_efficient array to search for

    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs > costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    if return_mask:
        is_efficient_mask = np.full(n_points, np.nan, dtype=object)
        is_efficient_mask[is_efficient] = "Y"
        return is_efficient_mask
    else:
        return is_efficient


def chart_maker(df, workbook, sheet_name):
    use_df = df.reset_index()
    chart = workbook.add_chart({'type': 'scatter'})
    ad_column = get_column_letter(df.columns.get_loc('AD(%)') + 2)
    modp_column = get_column_letter(df.columns.get_loc('ModP') + 2)
    ad_val = '$' + ad_column + '${}'
    modp_val = '$' + modp_column + '${}'
    std_categories = f"='{sheet_name}'!" + ad_val + ':' + ad_val
    std_values = f"='{sheet_name}'!" + modp_val + ':' + modp_val
    lone_category = f",'{sheet_name}'!" + ad_val
    lone_value  = f",'{sheet_name}'!" + modp_val
    #Set non-noted values
    chart_end = df.shape[0]
    pareto_df = use_df[use_df['Pareto'] == 'Y']
    pareto_end = pareto_df.index[-1] + 2
    chart.add_series({'name': 'Other Models',
                      'categories': std_categories.format(pareto_end + 1, chart_end),
                      'values': std_values.format(pareto_end + 1, chart_end),
                      'marker': {'type': 'circle',
                                 'size': 8,
                                 'border': {'color': 'black'},
                                 'fill': {'color': '#5B9BD5'}},
                      })
    #Set base values
    base_df = use_df[~use_df['Base'].isnull()]
    base_end = base_df.index[-1] + 2
    chart.add_series({'name': 'Base Models',
                      'categories': std_categories.format(2, base_end),
                      'values': std_values.format(2, base_end),
                      'marker': {'type': 'diamond',
                                 'size': 11,
                                 'border': {'color': 'black'},
                                 'fill': {'color': '#ED7D31'}},
                      })
    #Set max values
    max_df = use_df[~use_df['Max'].isnull()]
    max_end = max_df.index[-1] + 2
    chart.add_series({'name': 'All Models',
                      'categories': std_categories.format(base_end + 1, max_end),
                      'values': std_values.format(base_end + 1, max_end),
                      'marker': {'type': 'triangle',
                                 'size': 11,
                                 'border': {'color': 'black'},
                                 'fill': {'color': '#FFC000'}},
                      })
    #Set pareto values
    missing_pareto = pareto_df[~pareto_df.index.isin(np.arange(max_end - 1, pareto_end - 1))]
    pareto_cat = std_categories.format(max_end + 1, pareto_end)
    pareto_val = std_values.format(max_end + 1, pareto_end)
    for i in missing_pareto.index:
        pareto_cat = pareto_cat + lone_category.format(i + 2)
        pareto_val = pareto_val + lone_value.format(i + 2)
    if missing_pareto.shape[0] != 0:
        pareto_cat = pareto_cat[:1] + '(' + pareto_cat[1:] + ')'
        pareto_val = pareto_val[:1] + '(' + pareto_val[1:] + ')'
    chart.add_series({'name': 'Pareto Front Models',
                      'categories': pareto_cat,
                      'values': pareto_val,
                      'marker': {'type': 'square',
                                 'size': 11,
                                 'border': {'color': 'black'},
                                 'fill': {'color': 'gray'}},
                      })
    x_min = math.floor(min(df['AD(%)']) * 10) / 10.0
    y_min = math.floor(min(df['ModP']) * 10) / 10.0
    chart.set_size({'x_scale': 0.983, 'y_scale': 1.172})
    chart.set_legend({'none': True})
    chart.set_x_axis({'name': 'Coverage',
                      'name_font': {'size': 18, 'name': 'Calibri', 'bold': False},
                      'num_font': {'size': 14, 'name': 'Calibri'},
                      'num_format': '0%',
                      'min': x_min,
                      'max': 1,
                      'line': {'color': '#D9D9D9'},
                      'major_gridlines': {'visible': True, 'line': {'color': '#D9D9D9'}},
                      })
    chart.set_y_axis({'name': 'Modified Predictivity',
                      'name_font': {'size': 18, 'name': 'Calibri', 'bold': False},
                      'num_font': {'size': 14, 'name': 'Calibri'},
                      'min': y_min,
                      'max': 1,
                      'line': {'color': '#D9D9D9'},
                      'major_gridlines': {'visible': True, 'line': {'color': '#D9D9D9'}},
                      })
    return chart


def get_column_letter(n):
    result = ""
    while n > 0:
        # Calculate the remainder when divided by 26 (0-25)
        remainder = (n - 1) % 26
        # Convert the remainder to a letter (A=0, B=1, ..., Z=25)
        letter = chr(ord('A') + remainder)
        # Prepend the letter to the result
        result = letter + result
        # Reduce n by 1 and divide by 26 to get the next digit
        n = (n - 1) // 26
    return result


def consensus_predictions(endpoint, results, vals, fps):
    models, metrics = results
    pareto = models[models['Pareto'] == 'Y']
    num_models_idx = pareto.columns.get_loc('# of models')
    all_models = list(pareto.columns[1:num_models_idx])
    if 'kNN' in list(pareto['Combination']):
        fa_info = metrics['FA_info']
        knn_dists = knn_calculate(fps, fa_info)
    endpoint_results = []
    for index, row in pareto.iterrows():
        method = row['Combination']
        method_models = [x for x in all_models if row[x] == 'Y']
        pred_results = vals[method_models]
        if isinstance(method, (int, float)) and np.isnan(method):
            results = vals[method_models[0]]
            results.name = method_models[0]
        elif method not in ['kNN']:
            results = combine_pred_results(pred_results, method, metrics)
            model_names = ', '.join(method_models)
            results.name = f'{method} combination of {model_names}'
        elif method == 'kNN':
            full_knn_dists = pd.concat([knn_dists for _ in method_models], axis=1)
            full_knn_dists.columns = method_models
            for model in method_models:
                full_knn_dists[model][full_knn_dists[model] > metrics[model]['FA']] = np.nan
            results = combine_pred_results(pred_results, method, metrics, knn=full_knn_dists)
        endpoint_results.append(results)
    return True


def knn_calculate(fps, fa_info):
    fps = fps[fa_info[1]]
    fas = fa_info[0].transform(fps)
    cols = [f'FA_{i+1}' for i in range(fa_info[0].n_components_)]
    fa_df = pd.DataFrame(data=fas, index=fps.index, columns=cols)
    return knn_distances(fa_info[2], fa_df)


def main():
    # Define the filename and read in data
    name = argv[1]
    file_path = f'{name}.csv'
    fp_path = f'{name}_FP.csv'

    mode = 'train'
    if 'predict' in argv:
        mode = 'predict'

    if mode == 'train':

        pickle_file_path = f'{name}_consensus.pkl'
        if os.path.isfile(pickle_file_path):
            #If the Excel file exists read it into all_results
            with open(pickle_file_path, 'rb') as pklfile:
               all_results = pickle.load(pklfile)
        else:
            #If the Excel file doesn't exist create an empty dictionary
            # Call the function to read and extract results
            endpoint_results, fps = read_and_extract_results(file_path, fp_path)
    
            # Now you have a dictionary where each endpoint is related to its observed and predicted results
            all_results = {}
            for endpoint, results in endpoint_results.items():
                start = time()
                observed_results = results['observed']
                predicted_results = results['predicted']
                metrics = calculate_metrics(observed_results, predicted_results, fps)
                combination_results = test_endpoint_combinations(observed_results, predicted_results, metrics)
                combination_results['Pareto'] = pareto_front(combination_results[['AD(%)', 'ModP']].values)
                all_results[endpoint] = [combination_results, metrics]
                print(f'Finished {endpoint} in {time() - start}')
            with open(pickle_file_path, 'wb') as pklfile:
                pickle.dump(all_results, pklfile)
        excel_file_path = f'{name}_consensus_models.xlsx'
        with pd.ExcelWriter(excel_file_path) as writer:
            for sheet_name, results in all_results.items():
                df = results[0]
                df = df.sort_values(['Base', 'Max', 'Pareto', 'AD(%)'], ascending=[False, False, False, True])
                df.to_excel(writer, sheet_name=sheet_name)
                workbook = writer.book
                chart = chart_maker(df, workbook, sheet_name)
                worksheet = writer.sheets[sheet_name]
                worksheet.insert_chart('D23', chart)
    if mode == 'predict':
        vals, fps =read_and_extract_results(file_path, fp_path, predict=True)
        for models in glob('*_consensus.pkl'):
            with open(models, 'rb') as pklfile:
                all_results = pickle.load(pklfile)
            for endpoint, results in all_results.items():
                predictions = consensus_predictions(endpoint, results, vals, fps)


if __name__ == "__main__":
    main()
