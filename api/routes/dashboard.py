from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.repositories.dashboard_repository import DashboardRepository
from api.schemas.dashboard import RecentActivityResponse
from api.services.dashboard_service import DashboardService
from api.utils.auth import get_current_doctor

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get(
    "/recent-activity",
    response_model=RecentActivityResponse,
    summary="Get recent activity feed",
    description="Returns the latest 20 activities from across the system, protected by Doctor JWT."
)
def get_recent_activity(
    db: Session = Depends(get_db),
    current_doctor=Depends(get_current_doctor)
):
    repository = DashboardRepository(db)
    service = DashboardService(repository)
    activities = service.get_recent_activity(limit=20)
    return RecentActivityResponse(
        total=len(activities),
        activities=activities
    )
