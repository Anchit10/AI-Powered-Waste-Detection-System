import io
import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from ultralytics import YOLO
from passlib.context import CryptContext
import jwt
from mssa_layers import inject_mssa
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np

from models import init_db, get_db, User, WasteReport

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
SECRET_KEY = "supersecretkey_change_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dehradun localities mapped to responsible government authorities
DEHRADUN_AUTHORITIES = {
    "Rajpur Road":          "Nagar Nigam Dehradun — Ward 1",
    "Clock Tower":          "Nagar Nigam Dehradun — Ward 5",
    "Paltan Bazaar":        "Nagar Nigam Dehradun — Ward 5",
    "ISBT Dehradun":        "Nagar Nigam Dehradun — Ward 12",
    "Clement Town":         "Cantonment Board Clement Town",
    "Sahastradhara Road":   "Nagar Nigam Dehradun — Ward 2",
    "Mussoorie Road":       "Nagar Nigam Dehradun — Ward 3",
    "Race Course":          "Nagar Nigam Dehradun — Ward 6",
    "Ballupur":             "Nagar Nigam Dehradun — Ward 7",
    "Dalanwala":            "Nagar Nigam Dehradun — Ward 8",
    "Karanpur":             "Nagar Nigam Dehradun — Ward 9",
    "Rispana":              "Nagar Nigam Dehradun — Ward 10",
    "Dharampur":            "Nagar Nigam Dehradun — Ward 11",
    "Nehru Colony":         "Nagar Nigam Dehradun — Ward 13",
    "Shimla Bypass":        "Nagar Nigam Dehradun — Ward 14",
    "Jakhan":               "Nagar Nigam Dehradun — Ward 15",
    "Chandrabani":          "Nagar Nigam Dehradun — Ward 16",
    "Prem Nagar":           "Nagar Nigam Dehradun — Ward 17",
    "Doiwala":              "Nagar Palika Doiwala",
    "Raipur":               "Nagar Nigam Dehradun — Ward 18",
    "Vasant Vihar":         "Nagar Nigam Dehradun — Ward 19",
    "Turner Road":          "Nagar Nigam Dehradun — Ward 4",
    "Subhash Road":         "Nagar Nigam Dehradun — Ward 5",
    "GMS Road":             "Nagar Nigam Dehradun — Ward 20",
    "IT Park":              "Nagar Nigam Dehradun — Ward 21",
    "Doon University Area": "Nagar Nigam Dehradun — Ward 22",
    "Hathibarkala":         "Nagar Nigam Dehradun — Ward 23",
    "Kanwali":              "Nagar Nigam Dehradun — Ward 24",
    "Niranjanpur":          "Nagar Nigam Dehradun — Ward 25",
    "Other":                "Nagar Nigam Dehradun — General"
}

DEHRADUN_REPRESENTATIVES = {
    "Rajpur Road":          "Deepak Rawat (Councillor)",
    "Clock Tower":          "Sanjeev Sharma (Councillor)",
    "Paltan Bazaar":        "Sanjeev Sharma (Councillor)",
    "ISBT Dehradun":        "Manoj Kumar (Councillor)",
    "Clement Town":         "Sunil Rawat (Board Member)",
    "Sahastradhara Road":   "Priyanka Negi (Councillor)",
    "Mussoorie Road":       "Ravi Pathak (Councillor)",
    "Race Course":          "Vicky Khurana (Councillor)",
    "Ballupur":             "Anita Singh (Councillor)",
    "Dalanwala":            "Meenakshi Maurya (Councillor)",
    "Karanpur":             "Sanjay Nautiyal (Councillor)",
    "Rispana":              "Kamleshwar Singh (Councillor)",
    "Dharampur":            "Vinod Kumar (Councillor)",
    "Nehru Colony":         "Rekha Mehra (Councillor)",
    "Shimla Bypass":        "Mahesh Sharma (Councillor)",
    "Jakhan":               "Pradeep Nautiyal (Councillor)",
    "Chandrabani":          "Seema Bhardwaj (Councillor)",
    "Prem Nagar":           "Surya Mani Tyagi (Councillor)",
    "Doiwala":              "Sumit Bhardwaj (President)",
    "Raipur":               "Rakesh Joshi (Councillor)",
    "Vasant Vihar":         "Archana Purohit (Councillor)",
    "Turner Road":          "Nirmala Bisht (Councillor)",
    "Subhash Road":         "Sanjeev Sharma (Councillor)",
    "GMS Road":             "Rahul Rana (Councillor)",
    "IT Park":              "Swati Chandel (Councillor)",
    "Doon University Area": "Mohini Devi (Councillor)",
    "Hathibarkala":         "Vipul Kumar (Councillor)",
    "Kanwali":              "Ankit Saini (Councillor)",
    "Niranjanpur":          "Devendra Singh (Councillor)",
    "Other":                "Area Representative"
}

DEHRADUN_LOCALITIES = list(DEHRADUN_AUTHORITIES.keys())
DEFAULT_LAT = 30.3165
DEFAULT_LNG = 78.0322

# Report status stages
REPORT_STAGES = ["Reported", "Taken Up", "Being Solved", "Solved"]

# Auth setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    cred_exc = HTTPException(status_code=401, detail="Could not validate credentials",
                             headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise cred_exc
    except jwt.PyJWTError:
        raise cred_exc
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise cred_exc
    return user

def require_admin(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

def require_authority(current_user: User = Depends(get_current_user)):
    if current_user.role not in ("authority", "admin"):
        raise HTTPException(status_code=403, detail="Authority access required")
    return current_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Seed default admin account
    from models import SessionLocal
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == "admin@wastewatch.in").first()
        if not admin:
            admin = User(
                email="admin@wastewatch.in",
                password_hash=get_password_hash("admin123"),
                role="admin",
                name="System Admin"
            )
            db.add(admin)
            db.commit()
            logger.info("Seeded default admin account: admin@wastewatch.in / admin123")
    finally:
        db.close()
    yield

app = FastAPI(title="Dehradun Waste Watch API", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# Load YOLO model
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
try:
    model = YOLO(MODEL_PATH)
    model = inject_mssa(model, device='cpu')
    logger.info(f"Successfully loaded and modified YOLO model with MSSA from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    model = YOLO("yolov8n.pt")
    model = inject_mssa(model, device='cpu')
    logger.warning("Fell back to standard yolov8n.pt model with MSSA")

# COCO gating model for reducing false positives
try:
    COCO_MODEL = YOLO('yolov8m.pt')
    NON_WASTE_COCO_IDS = {
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
        0,                                       # person
        1, 2, 3, 4, 5, 6, 7, 8,                  # vehicles
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65,  # furniture
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # food
    }
    logger.info("Successfully re-enabled COCO gating model (yolov8m.pt)")
except Exception as e:
    logger.warning(f"Failed to load COCO gating model: {e}")
    COCO_MODEL = None

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)
    async def broadcast(self, msg: dict):
        for c in self.active_connections:
            try:
                await c.send_json(msg)
            except:
                pass

# ═══════════ EMAIL NOTIFICATION SYSTEM ═══════════

def send_email_notification(to_email: str, subject: str, body: str):
    """
    Simulates sending an email.
    In a real-world scenario, you would use smtplib or fastapi-mail here.
    """
    logger.info(f"📧 SENDING EMAIL TO: {to_email}")
    logger.info(f"📝 SUBJECT: {subject}")
    logger.info(f"📄 BODY:\n{body}\n{'-'*30}")
    
    # Real implementation example (commented out):
    # import smtplib
    # from email.mime.text import MIMEText
    # msg = MIMEText(body)
    # msg['Subject'] = subject
    # msg['From'] = "alerts@wastewatch.in"
    # msg['To'] = to_email
    # with smtplib.SMTP("smtp.gmail.com", 587) as server:
    #     server.starttls()
    #     server.login("your-email@gmail.com", "your-app-password")
    #     server.send_message(msg)

def notify_new_report(reporter_email: str, locality: str, waste_type: str, authorities: List[str]):
    # Notify Reporter
    send_email_notification(
        reporter_email,
        "Waste Report Received - Dehradun Waste Watch",
        f"Hello,\n\nYour report for '{waste_type}' in '{locality}' has been received. "
        f"Our team is looking into it.\n\nThank you for helping keep Dehradun clean!"
    )
    # Authority notification removed for now

def notify_status_change(reporter_email: str, report_id: int, new_status: str):
    send_email_notification(
        reporter_email,
        f"Update on Your Waste Report #{report_id}",
        f"Hello,\n\nThe status of your waste report #{report_id} has been updated to: '{new_status}'.\n"
        f"You can track the progress on our website."
    )
    async def broadcast(self, msg: dict):
        for c in self.active_connections:
            try:
                await c.send_json(msg)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/api/ws/detections")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ═══════════ AUTH ROUTES ═══════════

@app.post("/api/auth/register")
def register(
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("citizen"),
    name: str = Form(None),
    organization_name: str = Form(None),
    organization_type: str = Form(None),
    assigned_locality: str = Form(None),
    phone: str = Form(None),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Auto-detect admin: email starting with 'admin' gets admin role
    if email.lower().startswith("admin"):
        role = "admin"
    
    new_user = User(
        email=email,
        password_hash=get_password_hash(password),
        role=role,
        name=name or email.split("@")[0],
        organization_name=organization_name,
        organization_type=organization_type,
        assigned_locality=assigned_locality,
        phone=phone
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    token = create_access_token(data={"sub": new_user.email},
                                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {
        "access_token": token, "token_type": "bearer",
        "user": {"id": new_user.id, "email": new_user.email, "role": new_user.role, "name": new_user.name,
                 "organization_name": new_user.organization_name, "assigned_locality": new_user.assigned_locality}
    }

@app.post("/api/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    token = create_access_token(data={"sub": user.email},
                                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {
        "access_token": token, "token_type": "bearer",
        "user": {
            "id": user.id, "email": user.email, "role": user.role, "name": user.name,
            "organization_name": user.organization_name, "assigned_locality": user.assigned_locality
        }
    }

@app.get("/api/auth/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id, "email": current_user.email, "role": current_user.role,
        "name": current_user.name, "organization_name": current_user.organization_name,
        "assigned_locality": current_user.assigned_locality
    }

# ═══════════ ANALYZE (inference only) ═══════════

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """AI inference only — no DB save."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    try:
        contents = await file.read()
        from PIL import ImageOps
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = ImageOps.exif_transpose(image) # Fix orientation
        # Use SAHI for better detection
        predictions, best_conf, best_class = get_sahi_predictions(image)
        
        # Re-enabled COCO gating for better suppression
        predictions = apply_coco_gating(predictions, np.array(image))
        
        if not predictions:
            return JSONResponse(content={"predictions": [], "message": "No waste detected after COCO gating."})
        
        # Recalculate best after gating
        best_pred = max(predictions, key=lambda x: x["confidence"])
        best_class = best_pred["class"]
        best_conf = best_pred["confidence"]

        # Micro-waste categories
        MICRO_WASTE_CLASSES = ["cigarette_butt", "straw", "food_wrapper", "plastic_bottle_cap"]
        severity = "micro-waste" if best_class in MICRO_WASTE_CLASSES else "macro-waste"
        
        width, height = image.size
        return JSONResponse(content={
            "predictions": predictions, 
            "waste_type": best_class,
            "all_detected": list(set(p["class"] for p in predictions)),
            "confidence": best_conf, 
            "severity": severity,
            "image_width": width,
            "image_height": height
        })
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing image.")

# ═══════════ SAHI WRAPPER ═══════════

def get_sahi_predictions(image_pil):
    """Run Slicing Aided Hyper Inference for better small object detection."""
    # Convert PIL to numpy (RGB)
    image_np = np.array(image_pil)
    width, height = image_pil.size
    
    # Use UltralyticsDetectionModel directly instead of the factory
    detection_model = UltralyticsDetectionModel(
        model=model, # Uses the injected MSSA-YOLO model
        confidence_threshold=0.50, # Set to 0.50 as requested to filter low confidence categories
        device='cpu' # Using CPU as default for compatibility
    )
    
    # Run sliced prediction with NMS to remove duplicates
    result = get_sliced_prediction(
        image_np,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0,
        postprocess_type="NMS",
        postprocess_match_threshold=0.3
    )
    
    predictions = []
    best_conf = 0
    best_class = "Unknown"
    
    for object_prediction in result.object_prediction_list:
        bbox = object_prediction.bbox.to_xyxy()
        conf = object_prediction.score.value
        cls_name = object_prediction.category.name
        
        # Clip bounding boxes to image boundaries
        x1 = max(0, min(float(bbox[0]), width))
        y1 = max(0, min(float(bbox[1]), height))
        x2 = max(0, min(float(bbox[2]), width))
        y2 = max(0, min(float(bbox[3]), height))
        
        predictions.append({
            "class": cls_name,
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2]
        })
        
        if conf > best_conf:
            best_conf = conf
            best_class = cls_name
            
    return predictions, best_conf, best_class

def calculate_iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (float(aA + aB - inter) + 1e-7)

def apply_coco_gating(predictions, image_np, coco_iou_gate=0.5):
    """
    Suppresses waste detections that overlap with COCO non-waste objects.
    Ported from train.py logic.
    """
    if COCO_MODEL is None or not predictions:
        return predictions

    # Run COCO model on the image
    coco_results = COCO_MODEL.predict(source=image_np, conf=0.4, iou=0.45, imgsz=640, verbose=False)[0]
    
    coco_non_waste_boxes = []
    for box in coco_results.boxes:
        if int(box.cls) in NON_WASTE_COCO_IDS:
            coco_non_waste_boxes.append(box.xyxy[0].tolist())
    
    if not coco_non_waste_boxes:
        logger.info(f"COCO gating: No non-waste objects found. Keeping all {len(predictions)} detections.")
        return predictions

    filtered_preds = []
    for pred in predictions:
        waste_box = pred["bbox"] # [x1, y1, x2, y2]
        suppressed = any(calculate_iou(waste_box, coco_box) > coco_iou_gate for coco_box in coco_non_waste_boxes)
        if not suppressed:
            filtered_preds.append(pred)
    
    logger.info(f"COCO gating: Filtered {len(predictions)} -> {len(filtered_preds)} detections (gate={coco_iou_gate})")
    return filtered_preds

# ═══════════ REPORT ROUTES ═══════════

@app.post("/api/reports")
async def create_report(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lng: float = Form(0.0),
    locality: str = Form("Other"),
    token: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image.")
    try:
        contents = await file.read()
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(contents)
        
        from PIL import ImageOps
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = ImageOps.exif_transpose(image) # Fix mobile orientation
        # Use SAHI for better detection
        predictions, best_conf, best_class = get_sahi_predictions(image)
        
        # Re-enabled COCO gating for better suppression
        predictions = apply_coco_gating(predictions, np.array(image))
        
        if not predictions:
            return JSONResponse(content={"predictions": [], "message": "No waste detected after COCO gating."})
        
        # Recalculate best after gating
        best_pred = max(predictions, key=lambda x: x["confidence"])
        best_class = best_pred["class"]
        best_conf = best_pred["confidence"]

        MICRO_WASTE_CLASSES = ["cigarette_butt", "straw", "food_wrapper", "plastic_bottle_cap"]
        severity = "micro-waste" if best_class in MICRO_WASTE_CLASSES else "macro-waste"
        
        # Resolve reporter
        reporter_id = None
        if token:
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                user = db.query(User).filter(User.email == payload.get("sub")).first()
                if user: reporter_id = user.id
            except:
                pass
        
        # Auto-assign authority based on locality
        authority_id = None
        authority = db.query(User).filter(User.role == "authority", User.assigned_locality == locality).first()
        if authority:
            authority_id = authority.id
        
        report = WasteReport(
            lat=lat, lng=lng, locality=locality,
            image_path=f"/uploads/{filename}",
            waste_type=best_class, severity=severity, confidence=best_conf,
            all_detected_objects=", ".join(list(set(p["class"] for p in predictions))),
            reporter_id=reporter_id, authority_id=authority_id,
            status="Reported"
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        
        # Email Notifications via Background Task
        if report.reporter:
            background_tasks.add_task(
                notify_new_report, 
                report.reporter.email, 
                locality, 
                best_class, 
                [] # Authority notification removed
            )
        
        await manager.broadcast({
            "type": "new_report",
            "report": {
                "id": report.id, "lat": report.lat, "lng": report.lng,
                "waste_type": report.waste_type, "severity": report.severity,
                "status": report.status, "locality": report.locality
            }
        })
        
        return {
            "id": report.id, "waste_type": best_class, "confidence": best_conf,
            "all_detected_objects": report.all_detected_objects,
            "responsible_authority": DEHRADUN_AUTHORITIES.get(locality, "Unassigned"),
            "representative_name": DEHRADUN_REPRESENTATIVES.get(locality, "Area Representative"),
            "locality": locality, "status": "Reported"
        }
    except Exception as e:
        logger.error(f"Report creation error: {e}")
        raise HTTPException(status_code=500, detail="Internal error during report creation.")

@app.get("/api/reports")
def get_reports(db: Session = Depends(get_db)):
    reports = db.query(WasteReport).order_by(WasteReport.reported_at.desc()).all()
    return [_serialize_report(r) for r in reports]

@app.get("/api/reports/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)):
    r = db.query(WasteReport).filter(WasteReport.id == report_id).first()
    if not r: raise HTTPException(404, "Report not found")
    return _serialize_report(r)

def _serialize_report(r):
    return {
        "id": r.id, "lat": r.lat, "lng": r.lng, "locality": r.locality,
        "image_path": r.image_path, "waste_type": r.waste_type,
        "all_detected_objects": r.all_detected_objects or r.waste_type,
        "severity": r.severity, "confidence": r.confidence,
        "status": r.status,
        "reported_at": r.reported_at.isoformat() if r.reported_at else None,
        "taken_up_at": r.taken_up_at.isoformat() if r.taken_up_at else None,
        "being_solved_at": r.being_solved_at.isoformat() if r.being_solved_at else None,
        "solved_at": r.solved_at.isoformat() if r.solved_at else None,
        "admin_notes": r.admin_notes,
        "resolution_notes": r.resolution_notes,
        "after_image_path": r.after_image_path,
        "reporter_email": r.reporter.email if r.reporter else "Anonymous",
        "authority_name": r.assigned_authority.organization_name if r.assigned_authority else DEHRADUN_AUTHORITIES.get(r.locality, "Unassigned"),
        "representative_name": DEHRADUN_REPRESENTATIVES.get(r.locality, "Area Representative"),
        "authority_id": r.authority_id,
        "user_verified": r.user_verified,
        "verification_comment": r.verification_comment
    }

# ═══════════ ADMIN ROUTES ═══════════

@app.post("/api/admin/reports/{report_id}/status")
async def update_report_status(
    report_id: int,
    status: str = Form(...),
    notes: str = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: User = Depends(require_admin),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    if status not in REPORT_STAGES:
        raise HTTPException(400, f"Invalid status. Must be one of: {REPORT_STAGES}")
    
    report = db.query(WasteReport).filter(WasteReport.id == report_id).first()
    if not report: raise HTTPException(404, "Report not found")
    
    report.status = status
    if notes: report.admin_notes = notes
    
    now = datetime.utcnow()
    if status == "Taken Up": report.taken_up_at = now
    elif status == "Being Solved": report.being_solved_at = now
    elif status == "Solved":
        report.solved_at = now
        if file and file.content_type and file.content_type.startswith("image/"):
            contents = await file.read()
            fname = f"resolved_{uuid.uuid4()}_{file.filename}"
            fpath = os.path.join(UPLOAD_DIR, fname)
            with open(fpath, "wb") as f:
                f.write(contents)
            report.after_image_path = f"/uploads/{fname}"
    
    db.commit()
    
    # Notify Reporter of Status Change
    if report.reporter:
        background_tasks.add_task(notify_status_change, report.reporter.email, report.id, status)

    return {"message": f"Report #{report_id} status updated to '{status}'"}

@app.post("/api/admin/reports/{report_id}/assign")
def assign_authority(
    report_id: int,
    authority_id: int = Form(...),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    report = db.query(WasteReport).filter(WasteReport.id == report_id).first()
    if not report: raise HTTPException(404, "Report not found")
    
    authority = db.query(User).filter(User.id == authority_id, User.role == "authority").first()
    if not authority: raise HTTPException(404, "Authority user not found")
    
    report.authority_id = authority_id
    db.commit()
    return {"message": f"Assigned to {authority.organization_name}"}

@app.get("/api/admin/authorities")
def list_authorities(current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    auths = db.query(User).filter(User.role == "authority").all()
    return [{
        "id": a.id, "name": a.name, "email": a.email,
        "organization_name": a.organization_name,
        "assigned_locality": a.assigned_locality
    } for a in auths]

@app.post("/api/admin/authorities")
def create_authority(
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    organization_name: str = Form(...),
    assigned_locality: str = Form(...),
    phone: str = Form(None),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered")
    
    auth_user = User(
        email=email, password_hash=get_password_hash(password),
        role="authority", name=name,
        organization_name=organization_name,
        assigned_locality=assigned_locality, phone=phone
    )
    db.add(auth_user)
    db.commit()
    db.refresh(auth_user)
    return {"id": auth_user.id, "message": f"Authority '{name}' created for {assigned_locality}"}

# ═══════════ AUTHORITY ROUTES ═══════════

@app.get("/api/authority/reports")
def get_authority_reports(current_user: User = Depends(require_authority), db: Session = Depends(get_db)):
    """Get reports assigned to this authority."""
    if current_user.role == "admin":
        reports = db.query(WasteReport).order_by(WasteReport.reported_at.desc()).all()
    else:
        reports = db.query(WasteReport).filter(
            WasteReport.authority_id == current_user.id
        ).order_by(WasteReport.reported_at.desc()).all()
    return [_serialize_report(r) for r in reports]

@app.post("/api/authority/reports/{report_id}/update")
async def authority_update_report(
    report_id: int,
    status: str = Form(...),
    notes: str = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: User = Depends(require_authority),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    report = db.query(WasteReport).filter(WasteReport.id == report_id).first()
    if not report: raise HTTPException(404, "Report not found")
    
    # Verify authority owns this report (or is admin)
    if current_user.role != "admin" and report.authority_id != current_user.id:
        raise HTTPException(403, "Not assigned to this report")
    
    if status not in REPORT_STAGES:
        raise HTTPException(400, f"Invalid status. Must be one of: {REPORT_STAGES}")
    
    report.status = status
    if notes: report.resolution_notes = notes
    
    now = datetime.utcnow()
    if status == "Taken Up": report.taken_up_at = now
    elif status == "Being Solved": report.being_solved_at = now
    elif status == "Solved":
        report.solved_at = now
        # Handle after photo
        if file and file.content_type.startswith("image/"):
            contents = await file.read()
            fname = f"solved_{uuid.uuid4()}_{file.filename}"
            fpath = os.path.join(UPLOAD_DIR, fname)
            with open(fpath, "wb") as f:
                f.write(contents)
            report.after_image_path = f"/uploads/{fname}"
    
    db.commit()

    # Notify Reporter of Status Change
    if report.reporter:
        background_tasks.add_task(notify_status_change, report.reporter.email, report.id, status)

    return {"message": f"Report #{report_id} updated to '{status}'"}

# ═══════════ VERIFICATION ROUTES ═══════════

@app.post("/api/reports/{report_id}/verify")
def verify_report(
    report_id: int,
    verified: bool = Form(...),
    comment: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    report = db.query(WasteReport).filter(WasteReport.id == report_id).first()
    if not report: raise HTTPException(404, "Report not found")
    if report.status != "Solved":
        raise HTTPException(400, "Only solved reports can be verified")
    
    report.user_verified = verified
    report.verified_by_id = current_user.id
    report.verification_comment = comment
    db.commit()
    return {"message": f"Report #{report_id} verification: {'Confirmed' if verified else 'Disputed'}"}

# ═══════════ AREA GALLERY ═══════════

@app.get("/api/reports/area/{locality}")
def get_area_reports(locality: str, db: Session = Depends(get_db)):
    """Get all reports for a specific locality — for area gallery dropdown."""
    reports = db.query(WasteReport).filter(
        WasteReport.locality == locality
    ).order_by(WasteReport.reported_at.desc()).all()
    return [_serialize_report(r) for r in reports]

# ═══════════ DATA ENDPOINTS ═══════════

@app.get("/api/localities")
def get_localities():
    return {
        "localities": DEHRADUN_LOCALITIES,
        "authorities": DEHRADUN_AUTHORITIES,
        "representatives": DEHRADUN_REPRESENTATIVES,
        "default_lat": DEFAULT_LAT,
        "default_lng": DEFAULT_LNG
    }

@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(WasteReport).count()
    reported = db.query(WasteReport).filter(WasteReport.status == "Reported").count()
    taken_up = db.query(WasteReport).filter(WasteReport.status == "Taken Up").count()
    being_solved = db.query(WasteReport).filter(WasteReport.status == "Being Solved").count()
    solved = db.query(WasteReport).filter(WasteReport.status == "Solved").count()
    
    area_stats = db.query(WasteReport.locality, func.count(WasteReport.id)
                          ).group_by(WasteReport.locality).all()
    type_stats = db.query(WasteReport.waste_type, func.count(WasteReport.id)
                          ).group_by(WasteReport.waste_type).all()
    
    return {
        "total": total, "reported": reported, "taken_up": taken_up,
        "being_solved": being_solved, "solved": solved,
        "by_area": {a: c for a, c in area_stats if a},
        "by_type": {t: c for t, c in type_stats if t}
    }

# Mount static files last
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
