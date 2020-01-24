from abc import ABC, abstractmethod


class Transformation(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _transform(self):
        pass

    @abstractmethod
    def _project(self):
        pass
