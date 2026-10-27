from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas import ViolationsListResponse, ViolationItem, ReviewStatusUpdate, ReviewStatusResponse
from api.dependencies import get_db
from api.utils import to_media_url
from database.database_manager import DatabaseManager

router = APIRouter(prefix="/api/violations", tags=["Violations & Safety Management"])

VALID_REVIEW_STATUSES = {"pending", "approved", "rejected"}


@router.get("", response_model=ViolationsListResponse)
def get_violations(
    event_type: Optional[str] = Query(None, description="Filter by event type (e.g. NO_HELMET, WRONG_WAY, RED_LIGHT)"),
    review_status: Optional[str] = Query(None, description="Filter by review status: pending, approved, rejected"),
    search: Optional[str] = Query(None, description="Search term for plate_text, vehicle_class, or event_type"),
    include_counts: bool = Query(False, description="Whether to include traffic flow COUNT records"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    db: DatabaseManager = Depends(get_db)
) -> ViolationsListResponse:
    """
    Searchable, filterable, and paginated violation records.
    Strictly excludes road 'COUNT' events from safety violations display unless include_counts=True.
    Exposes safe static media URLs rather than internal local filesystem paths.
    """
    offset = (page - 1) * limit

    if event_type:
        raw_events = db.get_events(
            event_type=event_type,
            review_status=review_status,
            search=search,
            limit=limit,
            offset=offset
        )
        total_records = db.count_events(
            event_type=event_type,
            review_status=review_status,
            search=search
        )
    else:
        # Exclude COUNT events from general violation views unless include_counts=True
        exclude_types = None if include_counts else ["COUNT"]
        raw_events = db.get_events(
            exclude_event_types=exclude_types,
            review_status=review_status,
            search=search,
            limit=limit,
            offset=offset
        )
        total_records = db.count_events(
            exclude_event_types=exclude_types,
            review_status=review_status,
            search=search
        )

    items = []
    for r in raw_events:
        items.append(
            ViolationItem(
                id=r["id"],
                timestamp=r["timestamp"],
                track_id=r.get("track_id"),
                vehicle_class=r.get("vehicle_class"),
                event_type=r["event_type"],
                confidence=r.get("confidence"),
                plate_text=r.get("plate_text"),
                evidence_url=to_media_url(r.get("evidence_path")),
                review_status=r.get("review_status", "pending"),
                count_details=r.get("count_details"),
                total_count=r.get("total_count"),
            )
        )

    pages = (total_records + limit - 1) // limit if total_records > 0 else 1

    return ViolationsListResponse(
        total=total_records,
        page=page,
        limit=limit,
        pages=pages,
        items=items
    )


@router.patch("/{event_id}/review", response_model=ReviewStatusResponse)
def update_violation_review(
    event_id: int,
    body: ReviewStatusUpdate,
    db: DatabaseManager = Depends(get_db)
) -> ReviewStatusResponse:
    """
    Updates the review status ('pending', 'approved', 'rejected') for an existing record.
    Rejects review operations on COUNT summary events.
    """
    status_lower = body.status.lower().strip()
    if status_lower not in VALID_REVIEW_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid review status '{body.status}'. Must be one of: {sorted(list(VALID_REVIEW_STATUSES))}"
        )

    existing = db.get_event_by_id(event_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Violation event with ID {event_id} not found."
        )

    if existing.get("event_type") == "COUNT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Traffic flow COUNT events are statistical summaries and cannot be reviewed as safety violations."
        )

    ok = db.update_review_status(event_id, status_lower)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update review status in database."
        )

    return ReviewStatusResponse(
        id=event_id,
        review_status=status_lower,
        updated=True
    )
