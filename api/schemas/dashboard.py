from typing import List, Optional

from pydantic import BaseModel

class ActivityItem(BaseModel):
    type: str
    title: str
    description: str
    timestamp: str
    resource_id: Optional[str] = None

class RecentActivityResponse(BaseModel):
    total: int
    activities: List[ActivityItem]
