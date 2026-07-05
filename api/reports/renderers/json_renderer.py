from typing import Any, Dict


class JsonRenderer:
    def render(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        return report_data