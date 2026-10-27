import ast
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional, Any, Union


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

    @contextmanager
    def _get_connection(self):
        """Context manager providing a SQLite connection with Row factory and automatic closing."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        """
        Creates the 'events' table and associated indexes if they do not exist.
        Performs safe, backward-compatible column migrations and non-destructive
        data cleanup for legacy COUNT records.
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
            review_status TEXT DEFAULT 'pending',
            count_details TEXT,
            total_count INTEGER
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

            # Ensure columns exist on legacy tables without dropping or altering schema destructively
            cursor.execute("PRAGMA table_info(events);")
            existing_cols = {col["name"] for col in cursor.fetchall()}

            if "count_details" not in existing_cols:
                cursor.execute("ALTER TABLE events ADD COLUMN count_details TEXT;")
            if "total_count" not in existing_cols:
                cursor.execute("ALTER TABLE events ADD COLUMN total_count INTEGER;")

            # Non-destructive data migration for legacy COUNT records:
            # Move count dictionaries out of plate_text, extract total_count, and clean vehicle_class
            cursor.execute("""
                SELECT id, vehicle_class, plate_text, count_details, total_count
                FROM events
                WHERE event_type = 'COUNT'
                  AND (
                      (plate_text IS NOT NULL AND (plate_text LIKE '{%' OR plate_text LIKE '[%'))
                      OR count_details IS NULL
                  );
            """)
            legacy_count_rows = cursor.fetchall()
            for row in legacy_count_rows:
                rid = row["id"]
                vclass = row["vehicle_class"] or ""
                ptext = row["plate_text"] or ""
                cdetails = row["count_details"]
                tcount = row["total_count"]

                new_cdetails = cdetails
                if ptext and (ptext.startswith("{") or ptext.startswith("[")):
                    new_cdetails = ptext

                new_tcount = tcount
                if new_tcount is None:
                    if "Total:" in vclass:
                        try:
                            new_tcount = int(vclass.split(":")[-1].strip())
                        except ValueError:
                            pass
                    elif new_cdetails:
                        try:
                            parsed = json.loads(new_cdetails) if new_cdetails.startswith("{") else ast.literal_eval(new_cdetails)
                            if isinstance(parsed, dict):
                                if "total" in parsed:
                                    new_tcount = int(parsed["total"])
                                else:
                                    new_tcount = sum(int(v) for v in parsed.values() if isinstance(v, (int, float)))
                        except Exception:
                            pass

                new_vclass = "Traffic Stream" if ("Total:" in vclass or not vclass) else vclass

                cursor.execute("""
                    UPDATE events
                    SET vehicle_class = ?,
                        plate_text = NULL,
                        count_details = ?,
                        total_count = ?
                    WHERE id = ?;
                """, (new_vclass, new_cdetails, new_tcount, rid))

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
        prevent_duplicates: bool = True,
        count_details: Optional[Union[Dict, str]] = None,
        total_count: Optional[int] = None
    ) -> int:
        """
        Inserts a new traffic event / violation into the database using parameterized queries.
        Includes evidence path validation, structured count persistence, and duplicate event prevention.

        Args:
            event_type: Type of event (e.g. 'COUNT', 'PLATE_READ', 'NO_HELMET', 'WRONG_WAY').
            track_id: Vehicle tracking ID.
            vehicle_class: Vehicle classification category ('car', 'motorcycle', etc.) or 'Traffic Stream'.
            confidence: Detection or model confidence score.
            plate_text: License plate string. NEVER used for count dictionaries.
            evidence_path: Filepath to saved crop/snapshot/video evidence.
            review_status: Status ('pending', 'approved', 'rejected').
            timestamp: ISO datetime string or None (defaults to current local time).
            prevent_duplicates: If True, skips duplicate insertion if identical record exists.
            count_details: Structured dictionary or JSON string of category counts for COUNT events.
            total_count: Integer total vehicles counted for COUNT events.

        Returns:
            int: The primary key ID of the event record (newly created or existing duplicate).
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Enforce COUNT event persistence standards:
        # Never store count dictionaries in plate_text; store in count_details and total_count.
        if event_type == "COUNT":
            # If caller passed count dictionary or summary in plate_text, re-route to count_details
            if plate_text and (plate_text.startswith("{") or plate_text.startswith("[") or "{" in plate_text):
                if count_details is None:
                    count_details = plate_text
            # Never keep count dict in plate_text
            plate_text = None

            # Process count_details
            if isinstance(count_details, dict):
                dict_classes = {k: v for k, v in count_details.items() if k != "total"}
                if total_count is None and "total" in count_details:
                    try:
                        total_count = int(count_details["total"])
                    except (ValueError, TypeError):
                        pass
                if total_count is None:
                    total_count = sum(int(v) for v in dict_classes.values() if isinstance(v, (int, float)))
                count_details = json.dumps(count_details)
            elif isinstance(count_details, str):
                if total_count is None:
                    try:
                        parsed = json.loads(count_details) if count_details.startswith("{") else ast.literal_eval(count_details)
                        if isinstance(parsed, dict):
                            if "total" in parsed:
                                total_count = int(parsed["total"])
                            else:
                                total_count = sum(int(v) for v in parsed.values() if isinstance(v, (int, float)))
                    except Exception:
                        pass

            # If vehicle_class is "Total: X" or empty, normalize and extract total_count if still needed
            if vehicle_class and "Total:" in vehicle_class:
                if total_count is None:
                    try:
                        total_count = int(vehicle_class.split(":")[-1].strip())
                    except ValueError:
                        pass
                vehicle_class = "Traffic Stream"
            elif not vehicle_class:
                vehicle_class = "Traffic Stream"

        # Evidence path validation and normalization
        norm_evidence_path = None
        if evidence_path:
            norm_evidence_path = os.path.abspath(evidence_path) if os.path.exists(evidence_path) else str(evidence_path)

        # Duplicate check: prevent rapid re-insertions of identical event data
        if prevent_duplicates:
            if track_id is not None:
                check_sql = """
                SELECT id FROM events 
                WHERE event_type = ? 
                  AND track_id = ?
                ORDER BY id DESC LIMIT 1;
                """
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(check_sql, (event_type, track_id))
                    existing = cursor.fetchone()
                    if existing:
                        print(f"[DB] Skipped duplicate event insertion for track #{track_id}. Returning existing ID #{existing['id']}.")
                        return existing["id"]
            else:
                check_sql = """
                SELECT id FROM events 
                WHERE event_type = ? 
                  AND (evidence_path = ? OR (? IS NULL AND evidence_path IS NULL))
                  AND (plate_text = ? OR (? IS NULL AND plate_text IS NULL))
                  AND (vehicle_class = ? OR (? IS NULL AND vehicle_class IS NULL))
                  AND (count_details = ? OR (? IS NULL AND count_details IS NULL))
                ORDER BY id DESC LIMIT 1;
                """
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(check_sql, (
                        event_type,
                        norm_evidence_path, norm_evidence_path,
                        plate_text, plate_text,
                        vehicle_class, vehicle_class,
                        count_details, count_details
                    ))
                    existing = cursor.fetchone()
                    if existing:
                        print(f"[DB] Skipped duplicate event insertion. Returning existing ID #{existing['id']}.")
                        return existing["id"]

        query = """
        INSERT INTO events (
            timestamp, track_id, vehicle_class, event_type, confidence,
            plate_text, evidence_path, review_status, count_details, total_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            timestamp,
            track_id,
            vehicle_class,
            event_type,
            confidence,
            plate_text,
            norm_evidence_path,
            review_status,
            count_details,
            total_count
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def insert_count_event(
        self,
        total_count: int,
        class_counts: Dict[str, int],
        evidence_path: Optional[str] = None,
        confidence: Optional[float] = None,
        review_status: str = "approved",
        timestamp: Optional[str] = None
    ) -> int:
        """
        Structured insertion helper specifically for traffic counting sessions.
        Guarantees that vehicle counts are stored in structured fields (count_details, total_count)
        and never pollutes plate_text.
        """
        return self.insert_event(
            event_type="COUNT",
            vehicle_class="Traffic Stream",
            confidence=confidence,
            plate_text=None,
            evidence_path=evidence_path,
            review_status=review_status,
            timestamp=timestamp,
            count_details=json.dumps(dict(class_counts)),
            total_count=total_count
        )

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
        event_types: Optional[List[str]] = None,
        exclude_event_types: Optional[List[str]] = None,
        review_status: Optional[str] = None,
        track_id: Optional[int] = None,
        search: Optional[str] = None,
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

        if event_types:
            placeholders = ",".join(["?"] * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)

        if exclude_event_types:
            placeholders = ",".join(["?"] * len(exclude_event_types))
            query += f" AND event_type NOT IN ({placeholders})"
            params.extend(exclude_event_types)

        if review_status is not None:
            query += " AND review_status = ?"
            params.append(review_status)

        if track_id is not None:
            query += " AND track_id = ?"
            params.append(track_id)

        if search:
            query += " AND (plate_text LIKE ? OR vehicle_class LIKE ? OR event_type LIKE ?)"
            s_term = f"%{search}%"
            params.extend([s_term, s_term, s_term])

        query += " ORDER BY id DESC"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def count_events(
        self,
        event_type: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        exclude_event_types: Optional[List[str]] = None,
        review_status: Optional[str] = None,
        track_id: Optional[int] = None,
        search: Optional[str] = None
    ) -> int:
        """Counts total events matching the given filters."""
        query = "SELECT COUNT(*) as cnt FROM events WHERE 1=1"
        params = []

        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)

        if event_types:
            placeholders = ",".join(["?"] * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)

        if exclude_event_types:
            placeholders = ",".join(["?"] * len(exclude_event_types))
            query += f" AND event_type NOT IN ({placeholders})"
            params.extend(exclude_event_types)

        if review_status is not None:
            query += " AND review_status = ?"
            params.append(review_status)

        if track_id is not None:
            query += " AND track_id = ?"
            params.append(track_id)

        if search:
            query += " AND (plate_text LIKE ? OR vehicle_class LIKE ? OR event_type LIKE ?)"
            s_term = f"%{search}%"
            params.extend([s_term, s_term, s_term])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0

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

    def export_to_csv(
        self,
        event_type: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        exclude_event_types: Optional[List[str]] = None,
        review_status: Optional[str] = None
    ) -> str:
        """
        Exports database events as a CSV string.

        Args:
            event_type: Optional filter by event type.
            event_types: Optional list of event types to include.
            exclude_event_types: Optional list of event types to exclude.
            review_status: Optional filter by review status.

        Returns:
            str: CSV formatted string.
        """
        import io
        import csv

        events = self.get_events(
            event_type=event_type,
            event_types=event_types,
            exclude_event_types=exclude_event_types,
            review_status=review_status
        )
        if not events:
            return "id,timestamp,track_id,vehicle_class,event_type,confidence,plate_text,evidence_path,review_status,count_details,total_count\n"

        output = io.StringIO()
        fieldnames = [
            "id", "timestamp", "track_id", "vehicle_class", "event_type",
            "confidence", "plate_text", "evidence_path", "review_status",
            "count_details", "total_count"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for evt in events:
            writer.writerow(evt)

        return output.getvalue()

