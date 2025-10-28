from abc import ABC, abstractmethod

from network.types import NDIndex


class ConnectLayer(ABC):
    @abstractmethod
    def get_indexes(self, index: NDIndex) -> list[NDIndex]: ...

    def __getitem__(self, index: NDIndex) -> list[NDIndex]:
        return self.get_indexes(index)

    @abstractmethod
    def get_weight(
            self,
            prev_layer_index: NDIndex,
            next_layer_index: NDIndex
    ) -> float: ...