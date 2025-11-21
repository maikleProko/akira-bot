from abc import ABC, abstractmethod


class ValidationPattern(ABC):
    @abstractmethod
    def validate(self, *args):
        pass
