"""Metrics for classification.
"""

import numpy as np


def make_onehot(y, labels):
    """Convert y into a one hot format

    For example, given:
        y = [1,2,3,2,2,3]
        labels = [1,2,3]
    It will return:
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0, 1]]

    NOTE: You do NOT have to use this function. You MAY use it, if you find it
    helpful, especially for calculating precision and recall.

    Arguments
    ---------
        y: np.ndarray of shape (n_samples,)
        labels: list-like

    Returns
    -------
        np.ndarray: of shape (n_samples, len(labels))
    """

    labels = set(labels)
    if len(y.shape) != 1:
        raise Exception("Currently support only 1d input to make_onehot")

    label_indices = {label: i for i, label in enumerate(labels)}

    row_selector = [i for i, label in enumerate(y) if label in labels]
    column_selector = [label_indices[label] for label in y if label in label_indices]

    onehot = np.zeros((len(y), len(labels)), dtype=int)
    onehot[row_selector, column_selector] = 1
    return onehot


def check_metric_args(y_true, y_pred, average, labels):
    """Will check that y_true and y_pred are of compatible and correct shapes.
    
    Arguments
    ---------
        y_true: list-like 
        y_pred: list-like, of same shape as y_true
        average: One of "micro", "macro", or None
        labels: The labels for which we will calculate metrics

    Returns
    -------
        y_true: np.ndarray
        y_pred: np.ndarray
    """

    if average not in ["macro", "micro", None]:
        raise Exception("average param must be one of 'macro' or 'micro', or None.")

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise Exception("shape of y_true and y_pred is not the same")

    return y_true, y_pred


def precision(y_true, y_pred, average, labels):
    """Calculate precision.

    `labels` will be used to 
    
    Arguments
    ---------
        y_true: list-like 
        y_pred: list-like, of same shape as y_true
        average: One of "micro", "macro", or None
        labels: The labels for which we will calculate metrics

    Returns
    -------
        np.ndarray or float:
            If `average` is None, it returns a numpy array of shape
            (len(labels), ), where the precision values for each class are 
            calculated.
            Otherwise, it returns either the macro, or micro precision value as 
            float.
    """
    y_true, y_pred = check_metric_args(y_true, y_pred, average, labels)
    
    y_true_oh = make_onehot(y_true, labels)
    y_pred_oh = make_onehot(y_pred, labels)
    
    
    m = y_true_oh.shape[0]
    n = y_true_oh.shape[1]
        

    total_pos_count = 0
    total_neg_count = 0
    
    
    class_pos_counts = []
    class_neg_counts = []
    class_precisions = []
    
    for i in range(n):
        pos_count = 0
        neg_count = 0
        for j in range(m):
            if y_pred_oh[j][i] == 1:
                if y_true_oh[j][i] == 1:
                    pos_count += 1
                else :
                    neg_count += 1 
        class_pos_counts.append(pos_count)
        class_neg_counts.append(neg_count)
        total_pos_count += pos_count
        total_neg_count += neg_count
        
        
        
    
    for k in range(len(class_pos_counts)):
        class_precisions.append(class_pos_counts[k]/(class_pos_counts[k] + class_neg_counts[k]))
            
                
                
    if average is None:
        return np.array(class_precisions)
    
    if average == "macro":
        return (sum(class_precisions)/len(class_precisions))
    else:
        return(total_pos_count/(total_pos_count + total_neg_count))
            
    # At this point, you can be sure that y_true and y_pred
    # are one hot encoded.
    result = None
    ##### Write code here #######

    ##### End of your work ######
    return result


def recall(y_true, y_pred, average, labels):
    """Calculate precision.

    `labels` will be used to 
    
    Arguments
    ---------
        y_true: list-like 
        y_pred: list-like, of same shape as y_true
        average: One of "micro", "macro", or None
        labels: The labels for which we will calculate metrics

    Returns
    -------
        np.ndarray or float:
            If `average` is None, it returns a numpy array of shape
            (len(labels), ), where the recall values for each class are 
            calculated.
            Otherwise, it returns either the macro, or micro recall value as 
            float.
    """
    
    
    
    
    y_true, y_pred = check_metric_args(y_true, y_pred, average, labels)
    
    y_true_oh = make_onehot(y_true, labels)
    y_pred_oh = make_onehot(y_pred, labels)
    
    
    m = y_true_oh.shape[0]
    n = y_true_oh.shape[1]
    
    counts = []
    

    total_pos_count = 0
    total_neg_count = 0
    
    
    class_pos_counts = []
    class_neg_counts = []
    class_recalls = []
    
    for i in range(n):
        pos_count = 0
        neg_count = 0
        for j in range(m):
            if y_true_oh[j][i] == 1:
                if y_pred_oh[j][i] == 1:
                    pos_count += 1
                else :
                    neg_count += 1 
        class_pos_counts.append(pos_count)
        class_neg_counts.append(neg_count)
        total_pos_count += pos_count
        total_neg_count += neg_count
        
        
        
    
    for k in range(len(class_pos_counts)):
        class_recalls.append(class_pos_counts[k]/(class_pos_counts[k] + class_neg_counts[k]))
            
                
                
    if average is None:
        return np.array(class_recalls)
    
    if average == "macro":
        return (sum(class_recalls)/len(class_recalls))
    else:
        return(total_pos_count/(total_pos_count + total_neg_count))
        

    result = None
    ##### Write code here #######

    ##### End of your work ######

    return result


def test():
    """Test precision and recall
    """

    labels = [
        "blue",
        "red",
        "yellow",
    ]

    true1 = ["blue", "red", "blue", "blue", "blue", "blue", "yellow"]
    pred1 = ["blue", "red", "yellow", "yellow", "red", "red", "red"]

    for (correct_precision, correct_recall, averaging) in [
        [0.4166666666666667, 0.39999999999999997, "macro"],
        [0.2857142857142857, 0.2857142857142857, "micro"],
    ]:
        our_recall = recall(true1, pred1, labels=labels, average=averaging)
        our_precision = precision(true1, pred1, labels=labels, average=averaging)

        print("\nAveraging: {}\n============".format(averaging))

        print("Recall\n-------")
        print("Correct: ", correct_recall)
        print("Ours: ", our_recall)
        print("")

        print("Precision\n---------")
        print("Correct: ", correct_precision)
        print("Ours: ", our_precision)
        print("")

        if correct_recall == our_recall and correct_precision == our_precision:
            print("All good!")
        else:
            print("Hmm, check implementation.")


if __name__ == "__main__":
    test()
