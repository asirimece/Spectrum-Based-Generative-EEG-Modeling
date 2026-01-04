from sklearn.metrics import balanced_accuracy_score
from skorch.callbacks import EpochScoring, WandbLogger


def balanced_accuracy_scoring(net, dataset, y):
    y_pred = net.predict(dataset)
    return balanced_accuracy_score(y, y_pred)


class BalancedAccuracyScoring(EpochScoring):
    def __init__(self, lower_is_better=False, on_train=False, **kwargs):
        name = 'train_balanced_accuracy' if on_train else 'valid_balanced_accuracy'
        super().__init__(
            name=name,
            scoring=balanced_accuracy_scoring,
            lower_is_better=lower_is_better,
            on_train=on_train
        )
