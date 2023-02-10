
def get_metrics(X, qnn, ws):
    count = 0
    tp = 0
    fn = 0
    fp = 0
    y_predict = []
    y_tensor = []

    for x in X:
        out = qnn.forward(x[:-1], ws)
        label = 0
        if out > 0:
            label = 1
        else:
            label = -1
        if label == x[-1]:
            count += 1
            if label == 1:
                tp += 1
        else:
            if label == 1:
                fn += 1
            else:
                fp += 1
        y_predict.append(label)
        y_tensor.append(x[-1])

    accuracy = count / len(X)
    tn = count - tp
    print("Accuracy: ", accuracy)
    print("tp: ", tp, "tn: ", tn, " fp: ", fp, " fn: ", fn)

    return y_predict, y_tensor, accuracy, [tp, tn, fp, fn]