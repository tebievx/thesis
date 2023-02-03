import time
import itertools
import random
import concurrent.futures
import ordpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from dbscan1d.core import DBSCAN1D
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error


########################################################################################################################


def timer(orig_func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time()
        print('{} ran in: {} sec'.format(orig_func.__name__, t2-t1))
        return result
    return wrapper


def run_concurrently(task, inputs):
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(task, inputs):
            results.append(result)
    return results


def get_weighted_average(cluster, eps=1):
    distances = np.abs(cluster - np.mean(cluster))
    weights = (eps - distances) / eps
    weighted_average = np.sum(weights * cluster) / np.sum(weights)
    return weighted_average


########################################################################################################################


def get_patterns(pattern_size=3, max_distance=10, patterns_percent=100.0):
    patterns = [c for c in itertools.product(range(1, max_distance+1), repeat=pattern_size)]
    patterns = random.sample(patterns, int(len(patterns) * patterns_percent / 100.0))
    return patterns


def get_motifs(X_train, patterns):

    def _get_z_vectors(pattern):
        rows_num = len(X_train) - sum(pattern)
        first_row = np.int_(np.append([0], np.cumsum(pattern)))
        _filter = np.tile(first_row, (rows_num, 1)) + np.arange(rows_num).reshape(rows_num,1)
        z_vectors = X_train[_filter]
        return z_vectors

    motifs = {p: _get_z_vectors(p) for p in patterns}
    return motifs


########################################################################################################################


def get_possible_predictions(X_train, motifs, distance_eps=0.05):
    possible_predictions = []
    for pattern in motifs.keys():
        prediction_vector = X_train[len(X_train) - np.cumsum(pattern[::-1])][::-1]
        if np.isnan(prediction_vector).any():
            continue
        distances = distance.cdist(motifs[pattern][:,:-1], prediction_vector.reshape(1, -1), 'euclidean')
        close_vectors_idx = np.where(distances < distance_eps)[0]
        possible_predictions.append(motifs[pattern][close_vectors_idx, -1])
    return np.hstack(possible_predictions) if possible_predictions else np.array([])


def get_new_possible_predictions(
    X,  # X_train and Y_pred joined
    idx,  # the index of y_pred in X
    motifs,
    distance_eps = 0.05,
    last = False  # whether y_pred can take the last position in z_vector
):

    def _get_prediction_vector(pattern, pos):
        _indices_backward = idx - np.cumsum(pattern[0: pos][::-1]).astype(int)[::-1]
        z_backward = X[_indices_backward]
        _indices_forward = idx + np.cumsum(pattern[pos: len(pattern)]).astype(int)
        z_forward = np.take(np.append(X, np.nan), indices=_indices_forward, mode='clip')
        return np.append(z_backward, z_forward)

    possible_predictions = []
    for pattern in motifs.keys():
        positions = range(len(pattern)+1 if last else len(pattern)) # possible positions of y_pred inside z_vector
        for pos in positions:
            prediction_vector = _get_prediction_vector(pattern, pos)
            if np.isnan(prediction_vector).any():
                continue
            distances = distance.cdist(np.delete(motifs[pattern], pos, axis=1), prediction_vector.reshape(1, -1), 'euclidean')
            close_vectors_idx = np.where(distances < distance_eps)[0]
            possible_predictions.append(motifs[pattern][close_vectors_idx, pos])

    return np.hstack(possible_predictions) if possible_predictions else np.array([])


def get_cluster_labels(values, cluster_eps=0.01, min_samples=5):
    if len(values) == 0:
        return np.array([])
    dbscan = DBSCAN1D(eps=cluster_eps, min_samples=min_samples)
    labels = dbscan.fit_predict(values)
    return labels


def get_features(x):

    # features
    possible_predictions_std = 0
    cluster_1_std = 0
    cluster_1_proportion = 0
    clusters_proportion_diff = 0
    clusters_center_diff = 0

    def _features():
        return [
            possible_predictions_std,
            cluster_1_std,
            cluster_1_proportion,
            clusters_proportion_diff,
            clusters_center_diff
        ]

    if not len(x):
        return _features()

    cluster_labels = get_cluster_labels(
        values = x,
        cluster_eps = 0.01,
        min_samples = 5
    )

    cluster_labels_no_noise = cluster_labels[cluster_labels != -1]  # remove the cluster with noise
    if not len(cluster_labels_no_noise):
        return _features()

    cluster_counts = np.array(np.unique(cluster_labels_no_noise, return_counts=True)).T  # get the number of elements in each cluster
    cluster_counts = sorted(cluster_counts, key=lambda x: x[1], reverse=True)  # sort so that the largest cluster is first

    cluster_1_label = cluster_counts[0][0]

    # possible_predictions_std
    possible_predictions_std = x.std()

    # cluster_1_std
    cluster_1_std = x[cluster_labels == cluster_1_label].std()

    # cluster_1_proportion
    cluster_1_size = cluster_counts[0][1]
    cluster_1_proportion = cluster_1_size / len(x)

    # clusters_proportion_diff
    cluster_2_size = 0 if len(cluster_counts) < 2 else cluster_counts[1][1]
    cluster_2_proportion = cluster_2_size / len(x)
    clusters_proportion_diff = cluster_1_proportion - cluster_2_proportion

    # clusters_center_diff
    if len(cluster_counts) > 1:
        cluster_2_label = cluster_counts[1][0]
        center_1 = x[cluster_labels == cluster_1_label].mean()
        center_2 = x[cluster_labels == cluster_2_label].mean()
        clusters_center_diff = abs(center_1 - center_2)

    return _features()


def get_single_prediction(possible_predictions,
                          cluster_eps=0.01,
                          min_samples=5,
                          cluster_1_proportion_threshold=-1,
                          clusters_diff_threshold=-1,
                          sigma=0,
                          classifier=None,
                          classifier_prob=False,
                          weighted_average=False):

    # print(f'sigma = {sigma}')

    if not len(possible_predictions):
        return np.nan

    if cluster_eps == min_samples == cluster_1_proportion_threshold == clusters_diff_threshold == -1:
        prediction = np.mean(possible_predictions) + np.random.normal(0, sigma, 1)[0]
        return prediction

    if classifier is not None:
        features = get_features(possible_predictions)
        if classifier_prob:
            prob = classifier.predict_proba([features])[0][-1]
            if prob < 0.5:
                return np.nan
        elif not classifier.predict([features])[0]:
            return np.nan

    cluster_labels = get_cluster_labels(
        values = possible_predictions,
        cluster_eps = cluster_eps,
        min_samples = min_samples
    )

    cluster_labels_no_noise = cluster_labels[cluster_labels != -1]  # remove the cluster with noise

    if not len(cluster_labels_no_noise):
        return np.nan

    cluster_counts = np.array(np.unique(cluster_labels_no_noise, return_counts=True)).T  # get the number of elements in each cluster
    cluster_counts = sorted(cluster_counts, key=lambda x: x[1], reverse=True)  # sort so that the largest cluster is first

    # check cluster 1 proportion criteria
    cluster_1_size = cluster_counts[0][1]
    cluster_1_proportion = cluster_1_size / len(possible_predictions)
    if cluster_1_proportion < cluster_1_proportion_threshold:
        return np.nan

    # check first 2 clusters proportion difference criteria
    cluster_2_size = 0 if len(cluster_counts) < 2 else cluster_counts[1][1]
    cluster_2_proportion = cluster_2_size / len(possible_predictions)
    clusters_diff = cluster_1_proportion - cluster_2_proportion
    if (clusters_diff < clusters_diff_threshold):
        return np.nan

    # get prediction
    cluster_1_label = cluster_counts[0][0]
    cluster_1 = possible_predictions[cluster_labels == cluster_1_label]

    if weighted_average:
        prediction = get_weighted_average(cluster_1)
    else:
        prediction =  np.mean(cluster_1)

    if sigma:
        prediction += np.random.normal(0, sigma, 1)[0]  # add perturbation to the prediction

    return prediction


def get_single_trajectory(trajectory_length, X_train, motifs, distance_eps,  cluster_eps, min_samples,
                          cluster_1_proportion_threshold, clusters_diff_threshold, sigma, return_possible_predictions=False):
    _X_train = X_train.copy()
    possible_predictions = []
    for i in range(trajectory_length):
        _sigma = sigma + (i // 10) * 0.004
        _possible_predictions = get_possible_predictions(_X_train, motifs, distance_eps)
        prediction = get_single_prediction(
            _possible_predictions,
            cluster_eps,
            min_samples,
            cluster_1_proportion_threshold,
            clusters_diff_threshold,
            _sigma
        )
        _X_train = np.append(_X_train, prediction)
        possible_predictions.append(_possible_predictions)
    trajectory = _X_train[-trajectory_length:]
    if return_possible_predictions:
        return trajectory, possible_predictions
    return trajectory


def daemon(trajectory_length,
           X_train,
           Y_true,
           motifs,
           distance_eps,
           cluster_eps,
           min_samples,
           cluster_1_proportion_threshold,
           clusters_diff_threshold,
           sigma,
           return_possible_predictions=False,
           daemon_eps=0.05):
    _X_train = X_train.copy()
    possible_predictions = []
    for step in range(trajectory_length):
        _possible_predictions = get_possible_predictions(_X_train, motifs, distance_eps)
        prediction = get_single_prediction(
            _possible_predictions,
            cluster_eps,
            min_samples,
            cluster_1_proportion_threshold,
            clusters_diff_threshold,
            sigma
        )
        if prediction is not np.nan and abs(prediction - Y_true[step]) > daemon_eps:
            prediction = np.nan
        _X_train = np.append(_X_train, prediction)
        possible_predictions.append(_possible_predictions)
    trajectory = _X_train[-trajectory_length:]
    if return_possible_predictions:
        return trajectory, possible_predictions
    return trajectory


def get_new_predictions(
    X_train,
    old_predictions,
    motifs,
    distance_eps,
    cluster_eps,
    min_samples,
    cluster_1_proportion_threshold,
    clusters_diff_threshold,
    sigma,
    keep_previous_iter_predictions = True,
    last = False,  # whether y_pred can take the last position in z_vector
    possible_predictions_min_size = 80,
    return_possible_predictions = False,
    classifier = None,
    classifier_prob = False,
    weighted_average = False

):

    X = np.hstack([X_train, old_predictions])
    new_predictions = old_predictions.copy()
    possible_predictions = []

    for step in range(len(new_predictions)):
        if keep_previous_iter_predictions and not np.isnan(new_predictions)[step]:
            if return_possible_predictions:
                possible_predictions.append(np.array([]))
            continue

        idx = step + len(X_train)  # the index of y_pred in X

        _possible_predictions = get_new_possible_predictions(
            X = X,
            idx = idx,
            motifs = motifs,
            distance_eps = distance_eps,
            last = last
        )

        if return_possible_predictions:
            possible_predictions.append(_possible_predictions)

        if len(_possible_predictions) < possible_predictions_min_size:
            continue

        prediction = get_single_prediction(
            possible_predictions = _possible_predictions,
            cluster_eps = cluster_eps,
            min_samples = min_samples,
            cluster_1_proportion_threshold = cluster_1_proportion_threshold,
            clusters_diff_threshold = clusters_diff_threshold,
            sigma = sigma,
            classifier = classifier,
            classifier_prob = classifier_prob,
            weighted_average = weighted_average
        )
        new_predictions[step] = prediction

    if return_possible_predictions:
        return new_predictions, possible_predictions

    return new_predictions


########################################################################################################################


def get_multiple_trajectories(trajectories_num, trajectory_length, X_train, motifs, distance_eps,  cluster_eps,
                              min_samples, cluster_1_proportion_threshold, clusters_diff_threshold, sigma):
    args = {k: v for k, v in locals().items() if k != 'trajectories_num'}
    trajectories = []
    for i in range(trajectories_num):
            # print(i)
            trajectories.append(
                get_single_trajectory(**args)
            )
    return np.array(trajectories)


def get_max_cluster_predictions(predictions, cluster_eps, min_samples):
    total = len(predictions)

    predictions = predictions[~np.isnan(predictions)]
    if len(predictions) == 0:
        return np.array([]), 0

    cluster_labels = DBSCAN1D(eps=cluster_eps, min_samples=min_samples).fit_predict(predictions)

    predictions = predictions[np.where(cluster_labels != -1)]
    cluster_labels = cluster_labels[np.where(cluster_labels != -1)]

    _cluster_labels, _counts =  np.unique(cluster_labels, return_counts=True)
    if len(_counts) == 0:
        return np.array([]), 0

    _cluster_labels_max = _cluster_labels[np.where(_counts == np.max(_counts))]
    if _cluster_labels_max.size != 1:
        return np.array([]), 0

    idx = np.where(cluster_labels == _cluster_labels_max)
    max_cluster_predictions = predictions[idx]
    max_cluster_proportion = len(max_cluster_predictions) /  total

    return max_cluster_predictions,  max_cluster_proportion


def get_final_trajectory(trajectories, max_cluster_proportion_threshold, cluster_eps, min_samples):
    final_predictions = []
    for predictions in trajectories.T:  # each iteration will get predictions from all trajectories for a particular step
        max_cluster_predictions, max_cluster_proportion = get_max_cluster_predictions(predictions, cluster_eps, min_samples)
        final_predictions.append(
            np.mean(max_cluster_predictions) if max_cluster_proportion > max_cluster_proportion_threshold else np.nan
        )
    return final_predictions


########################################################################################################################


def get_non_predictable_points_percent(predicted):
    return np.sum(np.isnan(predicted), axis=0) / len(predicted) * 100


def get_rmse(true, predicted):
    rmse = []
    for step in range(true.shape[-1]):
        _filter = ~np.isnan(predicted[:, step])
        y_true = true[:, step][_filter]
        y_pred = predicted[:, step][_filter]
        rmse.append(
            mean_squared_error(y_true, y_pred, squared=False) if len(y_pred) else None
        )
    return rmse


def get_mse(true, predicted):
    mse = []
    for step in range(true.shape[-1]):
        _filter = ~np.isnan(predicted[:, step])
        y_true = true[:, step][_filter]
        y_pred = predicted[:, step][_filter]
        mse.append(
            mean_squared_error(y_true, y_pred, squared=True) if len(y_pred) else None
        )
    return mse


def get_mape(true, predicted):
    mape = []
    for step in range(true.shape[-1]):
        _filter = ~np.isnan(predicted[:, step])
        y_true = true[:, step][_filter]
        y_pred = predicted[:, step][_filter]
        mape.append(
            mean_absolute_percentage_error(y_true, y_pred) * 100 if len(y_pred) else None
        )
    return mape


def get_mae(true, predicted):
    mae = []
    for step in range(true.shape[-1]):
        _filter = ~np.isnan(predicted[:, step])
        y_true = true[:, step][_filter]
        y_pred = predicted[:, step][_filter]
        mae.append(
            mean_absolute_error(y_true, y_pred) if len(y_pred) else None
        )
    return mae


########################################################################################################################


def plot_entropy_complexity(time_series, label):
    # theoretical curves
    hc_max_curve = ordpy.maximum_complexity_entropy(dx=6).T
    hc_min_curve = ordpy.minimum_complexity_entropy(dx=6, size=719).T

    # entropy-complexity
    hc = ordpy.complexity_entropy(time_series, dx=6)

    # plot
    plt.plot(*hc_min_curve, color='green')
    plt.plot(*hc_max_curve, color='red')
    plt.plot(*hc, color='blue', marker='D', label=label)
    plt.title('Compexity-Entropy Plane')
    plt.xlabel('Entropy')
    plt.ylabel('MPR Complexity')
    plt.legend()
    plt.show()

    # print
    print(f'Entropy:          {hc[0]}')
    print(f'MPR-Complexity:   {hc[1]}')


########################################################################################################################


def plot_Y_true(Y_true):
    plt.plot(Y_true, label='True', zorder=3)


def plot_predictions(predictions):
    plt.scatter(range(len(predictions)), predictions, color='red', alpha=0.8, label='Predicted (by base algorithm)', zorder=4)


def plot_new_predictions(base_predictions, current_predictions):
    old_idx = np.argwhere(np.isnan(base_predictions)).reshape(1, -1)[0]
    new_idx = np.argwhere(np.isnan(current_predictions)).reshape(1, -1)[0]
    x = np.array(list(set(old_idx) - set(new_idx))).astype(int)
    y = current_predictions[x]
    plt.scatter(x, y, color='limegreen', label='Predicted (by new algorithm)', zorder=4)



def plot_non_predictable_points(predictions):
    non_pred_idx = np.argwhere(np.isnan(predictions)).reshape(1, -1)[0]
    non_pred = np.zeros_like(non_pred_idx)
    plt.scatter(non_pred_idx, non_pred, marker='x', color='grey', alpha=0.5, label='Non-predictable', zorder=2)


def plot_former_non_predictable_points(base_predictions, current_predictions):
    old_idx = np.argwhere(np.isnan(base_predictions)).reshape(1, -1)[0]
    new_idx = np.argwhere(np.isnan(current_predictions)).reshape(1, -1)[0]
    x = np.array(list(set(old_idx) - set(new_idx)))
    y = np.zeros_like(x)
    plt.scatter(x, y, marker='x', color='limegreen', label='Predictable again', zorder=2)


def plot_possible_predictions(possible_predictions):
    for step in range(len(possible_predictions)):
        y = possible_predictions[step]
        x = np.ones_like(y) * step
        plt.plot(
            x,
            y,
            marker = '_',
            # markersize = 8,
            linestyle = '',
            color='gold',
            label = '_nolegend_' if step > 0 else 'Possible predictions',
            zorder = 0
        )


def plot_possible_predictions_clustered(values, cluster_labels, step=0):
    if not len(values):
        return
    y = values
    x = np.ones_like(y) * step
    df = pd.DataFrame(dict(x=x, y=y, labels=cluster_labels))
    for label, group in df.groupby('labels'):
        if label == -1:
            continue         # don't plot noise
        plt.plot(
            group.x,
            group.y,
            # group.x.mean(),  # to plot cluster centers
            # group.y.mean(),  # to plot cluster centers
            marker = '_',
            markersize = 8,
            linestyle = '',
            color='orange',
            label = '_nolegend_' if step > 0 else 'Possible predictions',
            zorder = 0
        )


########################################################################################################################


def plot_non_predictable_points_percent(values):
    # plt.figure(figsize=[20, 10])
    plt.plot(values, marker='.',  linestyle='')
    plt.ylim(bottom=0)
    plt.title('Non-predictable Points')
    plt.xlabel('Steps')
    plt.ylabel('Non-predictable Points %')
    plt.show()


def plot_rmse(rmse):
    # plt.figure(figsize=[20, 10])
    plt.plot(rmse, marker='.', linestyle='')
    # plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.title('Root Mean Squared Error')
    plt.xlabel('Steps')
    plt.ylabel('RMSE')
    plt.show()


def plot_mse(mse):
    # plt.figure(figsize=[20, 10])
    plt.plot(mse, marker='.', linestyle='')
    # plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.title('Mean Squared Error')
    plt.xlabel('Steps')
    plt.ylabel('MSE')
    plt.show()


def plot_mape(mape):
    # plt.figure(figsize=[20, 10])
    plt.plot(mape, marker='.', linestyle='')
    plt.ylim(bottom=0)
    plt.title('Mean Absolute Percentage Error')
    plt.xlabel('Steps')
    plt.ylabel('MAPE')
    plt.show()


def plot_mae(mae):
    # plt.figure(figsize=[20, 10])
    plt.plot(mae, marker='.', linestyle='')
    # plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.title('Mean Absolute Error')
    plt.xlabel('Steps')
    plt.ylabel('MAE')
    plt.show()


########################################################################################################################


def compare_non_predictable_points_percent(base, new):
    plt.figure(figsize=[8, 5])
    plt.plot(base, marker='o',  linestyle='', color='red', mfc='none', label='Base algorithm', alpha=0.9)
    plt.plot(new, marker='^',  linestyle='', color='green', mfc='none', label='New algorithm', alpha=0.9)
    # plt.ylim(top=105)
    plt.ylim(bottom=0)
    plt.title('Non-predictable Points')
    plt.xlabel('Steps')
    plt.ylabel('Non-predictable Points %')
    plt.legend()
    plt.grid()
    plt.show()


def compare_rmse(base, new):
    plt.figure(figsize=[8, 5])
    plt.plot(base, marker='o',  linestyle='', color='red', mfc='none', label='Base algorithm', alpha=0.9)
    plt.plot(new, marker='^',  linestyle='', color='green', mfc='none', label='New algorithm', alpha=0.9)
    # plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.title('Root Mean Squared Error')
    plt.xlabel('Steps')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()


def compare_mse(base, new):
    plt.figure(figsize=[8, 5])
    plt.plot(base, marker='o',  linestyle='', color='red', mfc='none', label='Base algorithm', alpha=0.9)
    plt.plot(new, marker='^',  linestyle='', color='green', mfc='none', label='New algorithm', alpha=0.9)
    # plt.ylim(top=0.5)
    plt.ylim(bottom=0)
    plt.title('Mean Squared Error')
    plt.xlabel('Steps')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plt.show()


def compare_mape(base, new):
    plt.figure(figsize=[8, 5])
    plt.plot(base, marker='o',  linestyle='', color='red', mfc='none', label='Base algorithm', alpha=0.9)
    plt.plot(new, marker='^',  linestyle='', color='green', mfc='none', label='New algorithm', alpha=0.9)
    # plt.ylim(top=105)
    plt.ylim(bottom=0)
    plt.title('Mean Absolute Percentage Error')
    plt.xlabel('Steps')
    plt.ylabel('MAPE')
    plt.legend()
    plt.grid()
    plt.show()


def compare_mae(base, new):
    plt.figure(figsize=[8, 5])
    plt.plot(base, marker='o',  linestyle='', color='red', mfc='none', label='Base algorithm', alpha=0.9)
    plt.plot(new, marker='^',  linestyle='', color='green', mfc='none', label='New algorithm', alpha=0.9)
    # plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.title('Mean Absolute Error')
    plt.xlabel('Steps')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid()
    plt.show()


########################################################################################################################

def print_configurations():
    motifs = ['n', 'w']
    single_prediction = ['db', 'wa']
    non_predictable = ['sv', 'kn', 'lr', 'gb', 'mp']

    for m in motifs:
        for s in single_prediction:
            for n in non_predictable:
                print(f'{m}_{s}_{n}')
            print()
