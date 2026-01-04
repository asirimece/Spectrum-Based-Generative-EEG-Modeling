from lib.experiment.evaluation.result import Result, VisualResult


class Evaluator:
    name: str

    def __init__(self, name: str):
        self.name: str = name

    def evaluate(self, *args, **kwargs) -> dict[str, any]:
        pass


class Visualizer(Evaluator):

    def evaluate(self, *args, **kwargs) -> VisualResult:
        pass
