from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query, Response

from api.dependencies import get_db
from database.database_manager import DatabaseManager

router = APIRouter(prefix="/api/reports", tags=["Reports & Exports"])


@router.get("/export")
def export_violations_csv(
    event_type: Optional[str] = Query(None, description="Optional filter by event type"),
    review_status: Optional[str] = Query(None, description="Optional filter by review status"),
    include_counts: bool = Query(False, description="Whether to include traffic flow COUNT records in the export"),
    db: DatabaseManager = Depends(get_db)
) -> Response:
    """
    Exports filtered database events and violations as a standard downloadable CSV file.
    By default, traffic counting events are excluded to focus on safety violations.
    """
    exclude_types = None if include_counts else ["COUNT"]

    csv_data = db.export_to_csv(
        event_type=event_type,
        exclude_event_types=exclude_types,
        review_status=review_status
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"traffic_violations_{ts}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache"
        }
    )
