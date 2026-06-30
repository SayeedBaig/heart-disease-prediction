from typing import Any, Dict, Optional


class SharedMemoryService:
    """
    Stores shared backend data during runtime.

    Future modules like RAG, Reports and Digital Twin
    will access prediction results from here instead
    of calling the prediction pipeline again.
    """

    _memory: Dict[str, Any] = {}   # class-level: shared across ALL instances

    def set(self, key: str, value: Any) -> None:
        self._memory[key] = value

    def get(self, key: str) -> Optional[Any]:
        return self._memory.get(key)

    def clear(self) -> None:
        self._memory.clear()