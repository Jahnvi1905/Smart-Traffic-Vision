import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any


DEFAULT_DB_PATH = os.path.join("data", "traffic_violations.db")


class DatabaseManager:
    """
    SQLite Database Manager for Smart Traffic Vision.
    Manages traffic event logging, violation records, evidence references, and review workflows.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the database manager.
        Creates parent directories and initializes tables if they do not exist.
        """
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a SQLite connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """
        Creates the 'events' table and associated indexes if they do not exist.
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            track_id INTEGER,
            vehicle_class TEXT,
            event_type TEXT NOT NULL,
            confidence REAL,
            plate_text TEXT,
            evidence_path TEXT,
            review_status TEXT DEFAULT 'pending'
        );
        """
        create_idx_track = "CREATE INDEX IF NOT EXISTS idx_events_track_id ON events (track_id);"
        create_idx_type = "CREATE INDEX IF NOT EXISTS idx_events_event_type ON events (event_type);"
        create_idx_status = "CREATE INDEX IF NOT EXISTS idx_events_review_status ON events (review_status);"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_idx_track)
            cursor.execute(create_idx_type)
            cursor.execute(create_idx_status)
            conn.commit()

    def insert_event(
        self,
        event_type: str,
        track_id: Optional[int] = None,
        vehicle_class: Optional[str] = None,
        confidence: Optional[float] = None,
        plate_text: Optional[str] = None,
        evidence_path: Optional[str] = None,
        review_status: str = "pending",
        timestamp: Optional[str] = None,
        prevent_duplicates: bool = True
    ) -> int:
        """
        Inserts a new traffic event / violation into the database using parameterized queries.
        Includes evidence path validation and duplicate event prevention.

        Args:
            event_type: Type of event (e.g. 'COUNT', 'PLATE_READ', 'NO_HELMET', 'WRONG_WAY').
            track_id: Vehicle tracking ID.
            vehicle_class: Vehicle classification category ('car', 'motorcycle', etc.).
            confidence: Detection or model confidence score.
            plate_text: License plate string or summary text.
            evidence_path: Filepath to saved crop/snapshot/video evidence.
            review_status: Status ('pending', 'approved', 'rejected').
            timestamp: ISO datetime string or None (defaults to current local time).
            prevent_duplicates: If True, skips duplicate insertion if identical record exists.

        Returns:
            int: The primary key ID of the event record (newly created or existing duplicate).
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Evidence path validation and normalization
        norm_evidence_path = None
        if evidence_path:
            norm_evidence_path = os.path.abspath(evidence_path) if os.path.exists(evidence_path) else str(evidence_path)

        # Duplicate check: prevent rapid re-insertions of identical event data
        if prevent_duplicates:
            check_sql = """
            SELECT id FROM events 
            WHERE event_type = ? 
              AND (evidence_path = ? OR (? IS NULL AND evidence_path IS NULL))
              AND (plate_text = ? OR (? IS NULL AND plate_text IS NULL))
              AND (vehicle_class = ? OR (? IS NULL AND vehicle_class IS NULL))
            ORDER BY id DESC LIMIT 1;
            """
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(check_sql, (
                    event_type,
                    norm_evidence_path, norm_evidence_path,
                    plate_text, plate_text,
                    vehicle_class, vehicle_class
                ))
                existing = cursor.fetchone()
                if existing:
                    print(f"[DB] Skipped duplicate event insertion. Returning existing ID #{existing['id']}.")
                    return existing["id"]

        query = """
        INSERT INTO events (
            timestamp, track_id, vehicle_class, event_type, confidence,
            plate_text, evidence_path, review_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            timestamp,
            track_id,
            vehicle_class,
            event_type,
            confidence,
            plate_text,
            norm_evidence_path,
            review_status
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def check_evidence_exists(self, event_id: int) -> bool:
        """
        Checks whether the evidence media file for an event actually exists on disk.
        """
        event = self.get_event_by_id(event_id)
        if not event or not event.get("evidence_path"):
            return False
        return os.path.exists(event["evidence_path"])

    def get_events(
        self,
        event_type: Optional[str] = None,
        review_status: Optional[str] = None,
        track_id: Optional[int] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieves events matching optional filter criteria.

        Returns:
            List of dictionaries representing matched event rows.
        """
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)

        if review_status is not None:
            query += " AND review_status = ?"
            params.append(review_status)

        if track_id is not None:
            query += " AND track_id = ?"
            params.append(track_id)

        query += " ORDER BY id DESC"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single event by its primary key ID.
        """
        query = "SELECT * FROM events WHERE id = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (event_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_review_status(self, event_id: int, new_status: str) -> bool:
        """
        Updates the review status ('pending', 'approved', 'rejected') of an event record.

        Returns:
            bool: True if a row was updated, False otherwise.
        """
        query = "UPDATE events SET review_status = ? WHERE id = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (new_status, event_id))
            conn.commit()
            return cursor.rowcount > 0

    def delete_event(self, event_id: int) -> bool:
        """
        Deletes an event record by ID (useful for testing and data cleanup).

        Returns:
            bool: True if a row was deleted, False otherwise.
        """
        query = "DELETE FROM events WHERE id = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (event_id,))
            conn.commit()
            return cursor.rowcount > 0

    def export_to_csv(self, event_type: Optional[str] = None) -> str:
        """
        Exports database events as a CSV string.

        Args:
            event_type: Optional filter by event type.

        Returns:
            str: CSV formatted string.
        """
        import io
        import csv

        events = self.get_events(event_type=event_type)
        if not events:
            return "id,timestamp,track_id,vehicle_class,event_type,confidence,plate_text,evidence_path,review_status\n"

        output = io.StringIO()
        fieldnames = ["id", "timestamp", "track_id", "vehicle_class", "event_type", "confidence", "plate_text", "evidence_path", "review_status"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for evt in events:
            writer.writerow(evt)

        return output.getvalue()

