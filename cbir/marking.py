import numpy as np
from termcolor import colored
from IPython.display import HTML, display
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, silhouette_samples

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import warnings
import traceback


warnings.filterwarnings('ignore')


RED = "#FFCDD2"
GREEN = "#C8E6C9"
BLUE = "#e4f8fb"
YELLOW = "#ffe0b2"


def red(text):
    return colored(text, "red")


def green(text):
    return colored(text, "green")


def yellow(text):
    return colored(text, "yellow")


def set_background(color):
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)
    display(HTML('<img src onerror="{}" style="display:none">'.format(script)))


def this_is_an_action_cell():
    set_background(BLUE)


def report(passed, y, y_hat):
    if passed:
        result = "PASSED"
        log = green
        set_background(GREEN)
    else:
        result = "FAILED"
        log = red
        set_background(RED)
    print(log("Test {}".format(result)))
    print(log("Expecting:\n{}".format(y)))
    print(log("Received:\n{}\n".format(y_hat)))
    return


def verify(f):
    def wrapper(*args, **kwargs):
        set_background(YELLOW)
        do = input("Are you sure you want to submit and verify the code above?").lower() not in ["n", "no"]
        if not do:
            return
        print(yellow("Verifing...\t\t\t\t"), end="\r")
        try:
            passed, y, y_hat = f(*args, **kwargs)
            report(passed, y, y_hat)
            return
        except Exception as e:
            e = traceback.format_exc()
            report(False, None, repr(e))
            return
    return wrapper


@verify
def verify_class_imbalance(dataset, counts_fn):
    y_hat = counts_fn(dataset)
    _, y = np.unique(dataset.labels, return_counts=True)
    passed = np.array_equal(y_hat, y)
    return passed, y, y_hat


@verify
def verify_create_model(create_fn):
    model = create_fn()
    passed = True
    passed &= hasattr(model, "fit")
    passed &= hasattr(model, "predict")
    return passed, model, model


@verify
def verify_train_model(model):
    passed = True
    try:
        check_is_fitted(model)
    except:
        passed = False
    return passed, model, model


@verify
def verify_infer_model(create, infer):
    model = create()
    x = np.random.random((4, 28 * 28))
    y = np.random.randint(0, 2, (4, 1))
    model.fit(x, y)
    y = model.predict(x)
    y_hat = infer(model, x)
    passed = np.array_equal(y, y_hat)
    return passed, y, y_hat


@verify
def verify_kfold(create, user_skfold_fn, X, Y):
    k = 20
    X = X[:120]
    Y = Y[:120]
    def solution(x, y, k=k):
        """Here we perform stratified k-fold cross validation, with a random (not-yet-fitted) model, and a dataset.
        The function should return a list of scores for each fold iteration"""
        ### REMOVE THE BELOW CODE TO DEPLOY
        skfolds = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
        scores = []
        for train_index, val_index in skfolds.split(x, y):
            model = create(random_state=42, shuffle=True)

            x_train = x[train_index]
            y_train = y[train_index]
            x_val = x[val_index]
            y_val = y[val_index]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_val).reshape(-1, 1)
            n_correct = np.count_nonzero(y_pred == y_val)
            scores.append(n_correct)
        return np.array(scores)

    y = np.mean(solution(X, Y, k=k))
    y_hat = np.mean(user_skfold_fn(X, Y, k=k))

    passed = np.all((y_hat - y) < 2)
    return passed, y_hat, y


@verify
def verify_precision(accuracy_fn):
    x0 = (np.random.random((50,)) > 0.5).astype("int")
    x1 = (np.random.random((50,)) > 0.5).astype("int")
    y = precision_score(x0, x1)
    y_hat = accuracy_fn(x0, x1)
    passed = np.array_equal(y, y_hat)
    return passed, y_hat, y


@verify
def verify_recall(recall_fn):
    x0 = (np.random.random((50,)) > 0.5).astype("int")
    x1 = (np.random.random((50,)) > 0.5).astype("int")
    y = recall_score(x0, x1)
    y_hat = recall_fn(x0, x1)
    passed = np.array_equal(y, y_hat)
    return passed, y_hat, y


@verify
def verify_accuracy(accuracy_fn):
    x0 = (np.random.random((50,)) > 0.5).astype("int")
    x1 = (np.random.random((50,)) > 0.5).astype("int")
    y = accuracy_score(x0, x1)
    y_hat = accuracy_fn(x0, x1)
    passed = np.array_equal(y, y_hat)
    return passed, y_hat, y


@verify
def verify_f1(f1_fn):
    x0 = (np.random.random((50,)) > 0.5).astype("int")
    x1 = (np.random.random((50,)) > 0.5).astype("int")
    y = f1_score(x0, x1)
    y_hat = f1_fn(x0, x1)
    passed = np.array_equal(y_hat, y)
    return passed, y_hat, y


@verify
def verify_silhouette_coeff(f_sc):
    X = np.random.random((30,2))
    Y = np.random.randint(3, size=(30,))
    y = silhouette_samples(X, Y)
    y_hat = f_sc(X,Y)
    passed = np.allclose(y, y_hat)
    return passed, y_hat, y


@verify
def verify_kmeans(kmeans_fn):
    x, _ = make_blobs(10000, 100)
    k = 2

    kmeans = KMeans(n_clusters=k, random_state=42, init="random")
    kmeans.fit(x)

    y = kmeans.cluster_centers_
    _, y_hat = kmeans_fn(x, k)
    y = np.sort(y, axis=0)
    y_hat = np.sort(y_hat, axis=0)
    passed = np.linalg.norm(y - y_hat, ord=2).mean() < 1e-1
    return passed, y_hat, y
