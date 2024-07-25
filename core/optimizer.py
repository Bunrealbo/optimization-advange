from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")