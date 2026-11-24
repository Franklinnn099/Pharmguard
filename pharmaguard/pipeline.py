"""
End-to-end computer vision pipeline for PharmaGuard.
Orchestrates frame ingestion, inference, tracking, behavioral analysis,
alert generation, fine-grained telemetry measurement, and visual annotation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging
import os
import shutil
import threading
import time
import numpy as np
import cv2
import torch

from pharmaguard.config import PharmaGuardConfig, DetectorConfig, ShelfZone
from pharmaguard.detector import BaseDetector, YOLODetector, DetectionResult
from pharmaguard.tracker import ByteTrackerWrapper, TrackingResult, TrackedPerson
from pharmaguard.behavior import PharmacyBehaviorAnalyzer, BehaviorEvent
from pharmaguard.alerts import AlertManager, SecurityAlert
from pharmaguard.alarm import AlarmController
from pharmaguard.storage import EventStore
from pharmaguard.evidence import EvidenceRecorder
from pharmaguard.events import EventManager, SystemEvent
from pharmaguard.annotator import SecurityAnnotator
from pharmaguard.telemetry import StreamTelemetry, TelemetrySnapshot

logger = logging.getLogger(__name__)


def normalize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Coerce any browser frame to 3-channel RGB uint8 no wider than ``max_width``.

    Downscaling here bounds detection, annotation, and (most importantly) return-encoding cost,
    independent of whatever resolution the browser decided to capture.
    """
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    h, w = frame.shape[:2]
    if w > max_width:
        new_h = max(1, int(round(h * max_width / w)))
        frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return frame


@dataclass
class PipelineOutput:
    """Output package produced for each processed video frame."""
    annotated_frame_rgb: np.ndarray
    fps: float
    latency_ms: float
    active_tracks_count: int
    active_ids: List[int]
    recent_alerts: List[SecurityAlert]
    recent_events: List[SystemEvent]
    connection_status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PharmaGuardPipeline:
    """Core real-time processing pipeline."""

    def __init__(
        self,
        config: Optional[PharmaGuardConfig] = None,
        detector: Optional[BaseDetector] = None
    ):
        self.config = config or PharmaGuardConfig()

        # Component Initialization
        self.detector = detector or YOLODetector(self.config.detector)
        self.tracker = ByteTrackerWrapper(self.config.tracker)
        self.behavior_analyzer = PharmacyBehaviorAnalyzer(self.config.zones, self.config.behavior)
        self.object_detector: Optional[BaseDetector] = None
        if self.config.objects.enabled:
            oc = self.config.objects
            self.object_detector = YOLODetector(DetectorConfig(
                model_name=oc.model_name, confidence_threshold=oc.confidence_threshold,
                target_classes=oc.target_classes, device=self.config.detector.device,
                imgsz=self.config.detector.imgsz, half_precision=self.config.detector.half_precision,
            ))
        self.bag_boxes: List[np.ndarray] = []
        self.last_raw_frame: Optional[np.ndarray] = None   # un-annotated RGB, for the zone editor snapshot
        self.alert_manager = AlertManager(self.config.alert)
        self.event_manager = EventManager()
        self.alarm = AlarmController(self.config.alarm, on_action=self._on_alarm_action)
        self.store: Optional[EventStore] = None
        self.evidence: Optional[EvidenceRecorder] = None
        rc = self.config.recording
        if rc.enabled:
            self.store = EventStore(rc.data_dir, camera_id=rc.camera_id, mirror_dir=rc.mirror_dir)
            purged = self.store.purge_older_than(rc.retention_days)
            if purged:
                logger.info("Retention: deleted %d event(s) older than %.0f days", purged, rc.retention_days)
            self.evidence = EvidenceRecorder(rc, os.path.join(rc.data_dir, "clips"), on_clip_done=self._on_clip_done)
        self.annotator = SecurityAnnotator(self.config.display)

        # Performance & Timing Metrics
        self.telemetry = StreamTelemetry(
            window=self.config.stream.telemetry_window,
            signal_timeout_sec=self.config.stream.signal_timeout_sec,
        )
        self._lock = threading.Lock()
        self.last_output: Optional["PipelineOutput"] = None
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.frame_counter: int = 0

        # Fine-grained stage latencies (ms)
        self.last_stage_timings: Dict[str, float] = {
            "detect_ms": 0.0,
            "track_ms": 0.0,
            "behavior_ms": 0.0,
            "annotate_ms": 0.0,
            "total_backend_ms": 0.0
        }

        model_info = self.detector.get_model_info()
        self.hand_analysis = bool(model_info.get("has_keypoints", False))
        if not self.hand_analysis:
            logger.warning("Detector '%s' provides no pose keypoints: hand/shelf/bag analysis is DISABLED; "
                           "use a *-pose model (default yolov8n-pose.pt).", model_info.get("model_name"))
        self.device = model_info.get("device", "cpu")
        self.model_name = model_info.get("model_name", "unknown")

        self.event_manager.log(
            category="SYSTEM",
            message=f"Pipeline initialized with model '{self.model_name}' on device '{self.device}'"
        )

    def process_frame(self, frame_rgb: np.ndarray, timestamp: Optional[float] = None) -> PipelineOutput:
        """Process a single frame captured from the webcam (RGB format).

        Args:
            frame_rgb: RGB image numpy array (H, W, 3) or (H, W, 4).
            timestamp: optional frame time in seconds (e.g. video time for recorded footage).
                Defaults to the live monotonic clock.

        Returns:
            PipelineOutput with annotated frame and monitoring metrics.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return self._create_empty_output(time.perf_counter(), "NO_SIGNAL")

        with self._lock:
            return self._process_locked(frame_rgb, timestamp)

    def _process_locked(self, frame_rgb: np.ndarray, timestamp: Optional[float]) -> PipelineOutput:
        start_time = self.telemetry.mark_arrival()
        self.frame_counter += 1
        input_h, input_w = frame_rgb.shape[:2]

        frame_rgb = normalize_frame(frame_rgb, self.config.stream.max_process_width)
        self.last_raw_frame = frame_rgb
        h, w = frame_rgb.shape[:2]

        # Convert to BGR for OpenCV processing & standard model input
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 1. Person Detection (YOLO / RT-DETR)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_det_start = time.perf_counter()
        detection_result = self.detector.detect(frame_bgr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_det_end = time.perf_counter()
        detect_ms = (t_det_end - t_det_start) * 1000.0

        # 2. Multi-Object Tracking (ByteTrack)
        t_track_start = time.perf_counter()
        ts = time.monotonic() if timestamp is None else float(timestamp)
        tracking_result = self.tracker.update(detection_result, timestamp=ts)
        t_track_end = time.perf_counter()
        track_ms = (t_track_end - t_track_start) * 1000.0

        # 2b. Carried containers (bags), every Nth frame; boxes reused in between. On demand (default), only
        #     while a hand is at / just left a shelf - the only time bag positions can affect the score.
        objects_ms = 0.0
        if self.object_detector is not None:
            wanted = (not self.config.objects.on_demand) or self.behavior_analyzer.needs_bag_context(ts)
            if not wanted:
                self.bag_boxes = []
                self._bag_frames = 0
            else:
                if getattr(self, "_bag_frames", 0) % self.config.objects.frame_stride == 0:
                    t_obj = time.perf_counter()
                    self.bag_boxes = list(self.object_detector.detect(frame_bgr).xyxy)
                    objects_ms = (time.perf_counter() - t_obj) * 1000.0
                self._bag_frames = getattr(self, "_bag_frames", 0) + 1

        # 3. Behavioural analysis: hand-shelf-bag/pocket sequences and suspicion scoring
        t_beh_start = time.perf_counter()
        behavior_events = self.behavior_analyzer.analyze(
            tracking_result, frame_width=w, frame_height=h, timestamp=ts, bag_boxes=self.bag_boxes)
        t_beh_end = time.perf_counter()
        behavior_ms = (t_beh_end - t_beh_start) * 1000.0

        # 4. Alert Management -> operator alarm (ALERT sounds once per incident; WARNING is visual only)
        new_alerts = self.alert_manager.process_behavior_events(behavior_events)
        if new_alerts:
            self._record_alerts(new_alerts, ts)
            self.alarm.on_alerts(new_alerts)

        # Log new events to event timeline
        for b_evt in behavior_events:
            self.event_manager.log(
                category="SECURITY" if b_evt.severity in ("WARNING", "ALERT") else "INTERACTION",
                message=b_evt.description,
                metadata={"zone": b_evt.zone_name, "tracker_id": b_evt.tracker_id,
                          "type": b_evt.event_type.value, "score": b_evt.score}
            )

        # Per-zone activity: CONTACT (hand in zone) / PRESENCE / IDLE - computed per zone (audit B5).
        zone_activity = self.behavior_analyzer.zone_activity()

        # 5. Visual Annotation
        t_ann_start = time.perf_counter()
        annotated_bgr = frame_bgr.copy()
        annotated_bgr = self.annotator.draw_shelf_zones(annotated_bgr, self.config.zones, zone_activity)
        annotated_bgr = self.annotator.draw_bags(annotated_bgr, self.bag_boxes)
        annotated_bgr = self.annotator.draw_tracked_persons(
            annotated_bgr, tracking_result, views=self.behavior_analyzer.person_view)

        # HUD shows the previous frame's measured figures (this frame's are not final yet).
        tele = self.telemetry.snapshot()
        self.fps = tele.delivered_fps

        annotated_bgr = self.annotator.draw_hud(
            frame=annotated_bgr,
            fps=self.fps,
            latency_ms=self.latency_ms,
            active_tracks=len(tracking_result.active_ids),
            alert_count=len(self.alert_manager.alert_history),
            connection_status="LIVE"
        )
        # Alarm border last, so nothing (HUD bar) paints over it.
        annotated_bgr = self.annotator.draw_alarm_border(annotated_bgr, self.alarm.snapshot(), self.frame_counter)
        t_ann_end = time.perf_counter()
        annotate_ms = (t_ann_end - t_ann_start) * 1000.0

        # Convert back to RGB for browser/Gradio presentation
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        if self.evidence is not None:
            self.evidence.add_frame(annotated_rgb, ts)   # reference only; encoding happens off-thread

        self.latency_ms = self.telemetry.mark_return(start_time)
        tele = self.telemetry.snapshot()

        self.last_stage_timings = {
            "detect_ms": round(detect_ms, 2),
            "track_ms": round(track_ms, 2),
            "objects_ms": round(objects_ms, 2),
            "behavior_ms": round(behavior_ms, 2),
            "annotate_ms": round(annotate_ms, 2),
            "total_backend_ms": round(self.latency_ms, 2)
        }

        metadata = {
            "input_resolution": f"{input_w}x{input_h}",
            "resolution": f"{w}x{h}",
            "telemetry": tele,
            "device": self.device,
            "model_name": self.model_name,
            "stage_timings": self.last_stage_timings,
            "detections_count": detection_result.count,
            "bags_count": len(self.bag_boxes),
            "new_alerts": new_alerts,
            "behavior_events": behavior_events,
            "zone_activity": zone_activity,
            "frame_counter": self.frame_counter,
            "overlay": self._overlay_payload(tracking_result, zone_activity, w, h),
        }

        output = PipelineOutput(
            annotated_frame_rgb=annotated_rgb,
            fps=round(self.fps, 1),
            latency_ms=round(self.latency_ms, 1),
            active_tracks_count=len(tracking_result.active_ids),
            active_ids=list(tracking_result.active_ids),
            recent_alerts=self.alert_manager.get_recent_alerts(limit=10),
            recent_events=self.event_manager.get_recent(limit=15),
            connection_status="ONLINE",
            metadata=metadata
        )
        self.last_output = output
        return output

    def _overlay_payload(self, tracking_result, zone_activity: Dict[str, Any], w: int, h: int) -> Dict[str, Any]:
        """What the annotator draws, as plain data in normalized [0, 1] coordinates. The fast stream's
        smooth mode draws this over the browser's live video instead of downloading annotated frames."""
        def nx(v: float) -> float:
            return round(float(v) / w, 4)

        def ny(v: float) -> float:
            return round(float(v) / h, 4)

        persons = []
        for tid, person in tracking_result.tracked_persons.items():
            x1, y1, x2, y2 = [float(v) for v in person.current_box]
            entry: Dict[str, Any] = {"id": int(tid), "box": [nx(x1), ny(y1), nx(x2), ny(y2)],
                                     "conf": round(float(person.confidence), 3),
                                     "occluded": person.frames_lost > 0, "in_shelf": bool(person.in_shelf_zone)}
            view = self.behavior_analyzer.person_view(tid)
            if view is not None and person.frames_lost == 0:
                entry["level"], entry["score"] = int(view.level), float(view.score)
                hands = []
                for hand in view.hands:
                    pt = hand.point or hand.reach_point
                    if pt is None:
                        continue
                    state = "pocket" if (hand.in_pocket or hand.in_bag) else ("shelf" if hand.shelf_zone else "free")
                    hands.append({"side": hand.side, "x": nx(pt[0]), "y": ny(pt[1]), "state": state,
                                  "weak": hand.point is None})
                entry["hands"] = hands
                obs = view.observation
                if obs is not None and obs.has_pose:
                    entry["hips_visible"] = bool(obs.hips_visible)
                    entry["pockets"] = [[nx(b[0]), ny(b[1]), nx(b[2]), ny(b[3])] for b in obs.pocket_boxes]
            persons.append(entry)
        alarm = self.alarm.snapshot()
        return {
            "zones": [{"name": z.name, "polygon": z.polygon_normalized,
                       "state": zone_activity.get(z.name, "IDLE")} for z in self.config.zones],
            "bags": [[nx(b[0]), ny(b[1]), nx(b[2]), ny(b[3])] for b in self.bag_boxes],
            "persons": persons,
            "alarm": alarm.state,
            "sound": {"sounding": bool(alarm.sounding), "seq": int(alarm.sound_seq), "kind": alarm.kind,
                      "volume": float(self.config.alarm.volume)},
            "alerts": len(self.alert_manager.alert_history),
            "aspect": round(w / h, 4),
        }

    # ------------------------------------------------------------------ event recording
    def _record_alerts(self, alerts: List[SecurityAlert], ts: float) -> None:
        if self.store is None:
            return
        for a in alerts:
            if a.severity not in self.config.recording.record_severities:
                continue
            if self.store.record_alert(a) and self.evidence is not None:
                self.store.set_clip(a.alert_id, "recording")
                self.evidence.trigger(a.alert_id, ts)
        self._mirror_async()

    def _on_clip_done(self, event_uid: str, status: str, path: Optional[str], codec: Optional[str]) -> None:
        """Runs on the evidence worker thread."""
        if self.store is None:
            return
        mirror = self.config.recording.mirror_dir
        if path and mirror:
            try:
                os.makedirs(os.path.join(mirror, "clips"), exist_ok=True)
                shutil.copy2(path, os.path.join(mirror, "clips", os.path.basename(path)))
            except OSError as exc:
                logger.warning("Copying clip %s to %s failed: %s", path, mirror, exc)
        self.store.set_clip(event_uid, status, path, codec)
        self.store.mirror()

    def _mirror_async(self) -> None:
        if self.store is not None and self.store.mirror_dir and self.evidence is not None:
            self.evidence.submit(self.store.mirror)

    def close(self) -> None:
        """Flush evidence clips and make the database durable (call on shutdown / end of video)."""
        if self.evidence is not None:
            self.evidence.flush()
            self.evidence.wait_idle(timeout=180)
            self.evidence.close()
            self.evidence = None
        if self.store is not None:
            self.store.mirror()
            self.store.close()
            self.store = None

    def _on_alarm_action(self, action: str, details: Dict[str, Any]) -> None:
        """Audit every alarm transition: timeline + persistent log (who acknowledged what, when)."""
        operator = details.get("operator")
        msg = {
            "ALARM_RAISED": f"ALARM sounded for person #{details.get('tracker_id')} "
                            f"(incident {details.get('incident_id')}, score {details.get('score', 0):.2f})",
            "ALARM_ACKNOWLEDGED": f"Alarm acknowledged by {operator} ({len(details.get('alert_ids', []))} alert(s))",
            "ALARM_RESET": f"Alarms reset by {operator}; monitoring resumed",
            "ALARM_TEST": f"Manual alarm test started by {operator}",
        }.get(action, action)
        self.event_manager.log(category="ALARM", message=msg, metadata=dict(details))
        if self.store is not None:
            self.store.audit(action, operator, details)
            if action == "ALARM_ACKNOWLEDGED":
                self.store.mark_acknowledged(details.get("alert_ids", []), operator or "operator")
            elif action == "ALARM_RESET":
                self.store.mark_reset(details.get("alert_ids", []))
            self._mirror_async()
        if action != "ALARM_RAISED":  # raised alarms are already persisted as the alert record itself
            self.alert_manager.persist_record({"record": action, **details})

    def set_zones(self, zones: List["ShelfZone"]) -> None:
        """Replace monitored zones on the running system (zone editor). Takes effect on the next frame."""
        with self._lock:
            new = list(zones)
            self.config.zones = new
            self.behavior_analyzer.zones = new
        self.event_manager.log(category="SYSTEM", message=f"Monitoring zones updated: {[z.name for z in zones]}")
        if self.store is not None:
            self.store.audit("ZONES_UPDATED", None, {"zones": [z.model_dump() for z in zones]})

    def telemetry_snapshot(self) -> TelemetrySnapshot:
        """Thread-safe read of stream health for the dashboard poller."""
        return self.telemetry.snapshot()

    def _create_empty_output(self, start_time: float, status: str) -> PipelineOutput:
        """Fallback when frame is invalid or dropped."""
        blank_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_rgb, f"SIGNAL LOSS: {status}", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)
        latency = (time.perf_counter() - start_time) * 1000.0
        return PipelineOutput(
            annotated_frame_rgb=blank_rgb,
            fps=0.0,
            latency_ms=round(latency, 1),
            active_tracks_count=0,
            active_ids=[],
            recent_alerts=self.alert_manager.get_recent_alerts(limit=10),
            recent_events=self.event_manager.get_recent(limit=15),
            connection_status=status,
            metadata={"device": getattr(self, "device", "unknown"), "status": status}
        )

    def reset_state(self) -> None:
        """Reset trackers and alerts for fresh session."""
        with self._lock:
            self.tracker.reset()
            self.behavior_analyzer.reset()
            self.bag_boxes = []
            self.alert_manager.clear()
            self.alarm.reset(operator="system (tracking reset)")
            self.event_manager.clear()
            self.telemetry.reset()
            self.last_output = None
            self.frame_counter = 0
            self.fps = 0.0
            self.latency_ms = 0.0
