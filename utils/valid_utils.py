import statistics

from sklearn.metrics import roc_auc_score as auroc_scorer
from sksurv.metrics import concordance_index_censored

from utils.print_utils import print_info_message


def get_pooling_scores(wsi_score_dict, wsi_labels_dict, pooling_k_list):
    pooling_scores = []
    high_score_auc = 0
    wsi_names = []
    wsi_labels = []
    for pooling_k in pooling_k_list:
        wsi_scores = []
        wsi_names = []
        wsi_labels = []
        for wsi_name, wsi_score_list in wsi_score_dict.items():
            sorted_scores = sorted(wsi_score_list, reverse=True)

            top_scores = []
            for j in range(pooling_k):
                if j == len(sorted_scores):
                    break
                top_scores.append(sorted_scores[j])

            wsi_names.append(wsi_name)
            wsi_scores.append(statistics.mean(top_scores))
            wsi_labels.append(wsi_labels_dict[wsi_name])

        val_wsi_auc = auroc_scorer(wsi_labels, wsi_scores)

        if val_wsi_auc > high_score_auc:
            high_score_auc = val_wsi_auc

        pooling_scores.append(val_wsi_auc)
    return pooling_scores, high_score_auc


def calc_c_index(wsi_label_dict, wsi_score_dict, wsi_time_to_event_dict, pooling_k_list):
    event = [wsi_label == 1 for _, wsi_label in wsi_label_dict.items()]
    pooling_scores = []
    high_score_c_index = 0
    for pooling_k in pooling_k_list:
        wsi_scores = []
        time_to_events = []
        for wsi_name, wsi_score_list in wsi_score_dict.items():
            sorted_scores = sorted(wsi_score_list, reverse=True)

            top_scores = []
            for j in range(pooling_k):
                if j == len(sorted_scores):
                    break
                top_scores.append(sorted_scores[j])

            wsi_scores.append(statistics.mean(top_scores))
            time_to_events.append(wsi_time_to_event_dict[wsi_name])

        val_wsi_c_index = concordance_index_censored(event, time_to_events, wsi_scores, tied_tol=1e-08)[0]

        if val_wsi_c_index > high_score_c_index:
            high_score_c_index = val_wsi_c_index

        pooling_scores.append(val_wsi_c_index)
    return pooling_scores, high_score_c_index


def print_val_info_str(
    loss_list, auc_score, c_index_score, set_name=""
):
    info_str = f"{'' if set_name == '' else set_name + ' '}Loss: {loss_list.mean():.3f}"
    info_str += f" - AUC: {auc_score:.2f}"
    info_str += f" - C-Index: {c_index_score:.2f}"

    print_info_message(info_str)
