import ast
import json
from typing import Dict, Optional
from fastapi import APIRouter, Depends

from api.schemas import OverviewStatsResponse, RecentEventItem
from api.dependencies import get_db
from api.utils import to_media_url
from database.database_manager import DatabaseManager

router = APIRouter(prefix="/api/stats", tags=["Statistics & Overview"])


@router.get("/overview", response_model=OverviewStatsResponse)
def get_overview_stats(db: DatabaseManager = Depends(get_db)) -> OverviewStatsResponse:
    """
    Returns high-level traffic intelligence metrics:
    - Cumulative vehicle crossings counted across all sessions
    - Total safety violations (strictly excluding COUNT events)
    - Breakdown of violations by rule type
    - Aggregated vehicle counts across standard classifications
    - List of recent traffic events formatted with safe static media URLs
    """
    # 1. Aggregate vehicle crossings and category distribution from COUNT events
    count_records = db.get_events(event_type="COUNT")
    categories: Dict[str, int] = {
        "bus": 0,
        "car": 0,
        "three-wheeler": 0,
        "truck": 0,
        "two-wheeler": 0,
        "van": 0,
    }
    total_vehicles = 0

    for rec in count_records:
        cdetails = rec.get("count_details")
        tcount = rec.get("total_count")
        plate_str = rec.get("plate_text") or ""
        vclass = rec.get("vehicle_class") or ""

        # Parse counts from structured count_details or fallback legacy plate_text
        parsed_counts = None
        raw_source = cdetails or (plate_str if (plate_str.startswith("{") or plate_str.startswith("[")) else None)
        if raw_source:
            try:
                parsed_counts = json.loads(raw_source) if raw_source.startswith("{") else ast.literal_eval(raw_source)
            except Exception:
                try:
                    parsed_counts = ast.literal_eval(raw_source)
                except Exception:
                    parsed_counts = None

        if isinstance(parsed_counts, dict):
            classes_dict = parsed_counts.get("classes", parsed_counts) if isinstance(parsed_counts.get("classes"), dict) else parsed_counts
            sum_from_dict = 0
            for k, v in classes_dict.items():
                if k == "total":
                    continue
                clean_k = str(k).lower().strip()
                try:
                    val = int(v)
                    categories[clean_k] = categories.get(clean_k, 0) + val
                    sum_from_dict += val
                except (ValueError, TypeError):
                    pass

            if tcount is not None:
                total_vehicles += int(tcount)
            elif "total" in parsed_counts:
                try:
                    total_vehicles += int(parsed_counts["total"])
                except (ValueError, TypeError):
                    total_vehicles += sum_from_dict
            else:
                total_vehicles += sum_from_dict
        else:
            # Fallback to total_count column or legacy vehicle_class "Total: X"
            if tcount is not None:
                total_vehicles += int(tcount)
            elif "Total:" in vclass:
                try:
                    total_vehicles += int(vclass.split(":")[-1].strip())
                except ValueError:
                    pass

    # 2. Safety violations (strictly excluding COUNT events)
    total_violations = db.count_events(exclude_event_types=["COUNT", "PLATE_READ"])
    violations_by_type = {
        "NO_HELMET": db.count_events(event_type="NO_HELMET"),
        "WRONG_WAY": db.count_events(event_type="WRONG_WAY"),
        "RED_LIGHT": db.count_events(event_type="RED_LIGHT"),
    }

    # 3. Recent activity with safe media URLs (no arbitrary local filesystem paths)
    recent_raw = db.get_events(limit=10)
    recent_items = []
    for r in recent_raw:
        media_url = to_media_url(r.get("evidence_path"))
        recent_items.append(
            RecentEventItem(
                id=r["id"],
                timestamp=r["timestamp"],
                track_id=r.get("track_id"),
                vehicle_class=r.get("vehicle_class"),
                event_type=r["event_type"],
                confidence=r.get("confidence"),
                plate_text=r.get("plate_text"),
                evidence_url=media_url,
                review_status=r.get("review_status", "pending"),
                count_details=r.get("count_details"),
                total_count=r.get("total_count"),
            )
        )

    return OverviewStatsResponse(
        total_vehicles=total_vehicles,
        total_violations=total_violations,
        violations_by_type=violations_by_type,
        category_breakdown=categories,
        recent_events=recent_items,
    )
