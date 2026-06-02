# attendance-pi-client

Standalone Raspberry Pi attendance client for the AI-Based Class Attendance Monitoring System.

## What this client does

- Polls the backend for active attendance sessions
- Interprets session lifecycle using `session_date`, `start_time`, and `end_time`
- Captures camera frames locally on the Pi
- Runs face detection and recognition with InsightFace
- Tracks faces to avoid redundant recognition calls
- Aggregates attendance records in memory
- Exports local CSV and XLSX backups
- Caches batches when the network is unavailable
- Uploads attendance only through `POST /attendance/batch`
- Restarts automatically under systemd

## Session lifecycle

The Pi uses a state machine instead of treating every active session the same.

States:

- `IDLE` no active session is available
- `READY` a session exists, but the scheduled window has not started yet
- `RECORDING` the current time is inside the session window
- `STOPPED` the session window ended or the session became inactive

Rules:

- Recognition only runs while the state is `RECORDING`
- The Pi waits in `READY` until the session window opens
- The Pi stops recognition automatically when `end_time` is reached
- Attendance is flushed when the session transitions to `STOPPED`

## Repository layout

- `app.py` main process entrypoint
- `config.py` hardcoded runtime configuration
- `services/` backend client, session manager, collector, cache, export, upload, logging
- `services/embedding_client.py` backend-synced face embedding cache
- `recognition/` camera, recognizer adapter, in-memory embedding index, tracker
- `storage/` persistent state and pending upload queue
- `attendance_exports/` local backup exports
- `logs/` rotating runtime logs
- `systemd/attendance.service` systemd unit example

## Backend contract used

### `GET /pi/config`

Expected response:

```json
{
  "active_sessions": [
    {
      "session_id": "abc123",
      "course_code": "CPE501",
      "status": "active",
      "session_date": "2026-06-02",
      "start_time": "09:00:00",
      "end_time": "10:00:00"
    }
  ],
  "refresh_interval": 10
}
```

### `POST /attendance/batch`

Expected request:

```json
{
  "session_id": "abc123",
  "device_id": "PI-001",
  "records": [
    {
      "matric_no": "20CJ027513",
      "confidence": 0.94,
      "first_seen": "timestamp",
      "last_seen": "timestamp"
    }
  ]
}
```

## Setup

1. Create and activate a Python environment on the Pi.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Edit `config.py` and set:
   - `BACKEND_URL`
   - `DEVICE_ID`
   - any camera or timing values you need
4. Ensure the service user can write to:
   - `storage/`
   - `attendance_exports/`
   - `logs/`

## Recognition data

The Pi now syncs embeddings from the backend at startup and on a refresh interval.

Backend response format:

```json
{
  "version": "2026-06-02T10:00:00",
  "embeddings": [
    {
      "student_id": "20CJ027513",
      "embedding": [0.1, 0.2, 0.3]
    }
  ]
}
```

The Pi keeps the embedding index in memory and no longer reads static embedding files.

## Run manually

From the repository root:

```bash
python app.py
```

## systemd installation

1. Copy the repository to a stable path, for example `/opt/attendance-pi-client`.
2. Update `systemd/attendance.service` if your path or username differs.
3. Install the unit:

```bash
sudo cp systemd/attendance.service /etc/systemd/system/attendance.service
sudo systemctl daemon-reload
sudo systemctl enable attendance.service
sudo systemctl start attendance.service
```

## Operational notes

- The Pi never writes directly to the database.
- Offline batches are preserved in `storage/pending_batches.json`.
- Exports are generated only on session stop, before final upload, and during graceful shutdown.
- Export filenames include both `session_id` and `course_code`.
- The upload order is cached batches first, then the current batch.
- Network failures are retried with exponential backoff and logged; the process should keep running.

## Logging

The client writes structured events to `logs/pi.log`, including:

- `SESSION_STATE_CHANGE`
- `SESSION_WINDOW_ACTIVE`
- `SESSION_WINDOW_INACTIVE`
- `TRACK_LOCKED`
- `RECOGNITION_SKIPPED`
- `UPLOAD_RETRY`
- `ATTENDANCE_EXPORTED`
- `EMBEDDINGS_REFRESHED`
