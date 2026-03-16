from fastapi import APIRouter, Depends
from app.auth import require_role

router = APIRouter()

@router.get("/dashboard-data")
def dashboard_data(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    return {
        "message": "Dashboard endpoint working"
    }