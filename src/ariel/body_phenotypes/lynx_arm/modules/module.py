# src/ariel/body_phenotypes/lynx_arm/modules/module.py
import typing
from abc import ABC, abstractmethod

class Module(ABC):
    """Base class for all modules."""
    required_attributes: typing.ClassVar[list[str]] = ["index", "module_type"]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for attr in cls.required_attributes:
            if not hasattr(cls, attr):
                raise NotImplementedError(
                    f"Class '{cls.__name__}' must define attribute '{attr}'"
                )

    @abstractmethod
    def rotate(self, angle: float) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement 'rotate' method."
        )
