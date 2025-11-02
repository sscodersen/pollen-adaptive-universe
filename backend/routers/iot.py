from fastapi import APIRouter, Depends, Query, HTTPException
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from backend.database import get_db
from backend.services.iot_service import iot_service


router = APIRouter()


class DeviceRegistration(BaseModel):
    device_id: Optional[str] = None
    device_type: str
    device_name: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    os_version: Optional[str] = None
    firmware_version: Optional[str] = None
    location: Optional[dict] = None
    capabilities: Optional[list] = None
    metadata: Optional[dict] = None


class TelemetryData(BaseModel):
    type: str
    sensor_data: Optional[dict] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    metadata: Optional[dict] = None


@router.post("/devices/register")
async def register_device(
    device: DeviceRegistration,
    db: Session = Depends(get_db)
):
    result = iot_service.register_device(db, device.dict())
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/devices")
async def get_devices(
    device_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db)
):
    devices = iot_service.get_devices(db, device_type, status, limit)
    return {"devices": devices, "count": len(devices)}


@router.post("/devices/{device_id}/telemetry")
async def record_telemetry(
    device_id: str,
    telemetry: TelemetryData,
    db: Session = Depends(get_db)
):
    result = iot_service.record_telemetry(db, device_id, telemetry.dict())
    return result


@router.get("/devices/{device_id}/telemetry")
async def get_device_telemetry(
    device_id: str,
    hours: int = Query(24, le=168),
    limit: int = Query(1000, le=5000),
    db: Session = Depends(get_db)
):
    telemetry = iot_service.get_device_telemetry(db, device_id, hours, limit)
    return {"device_id": device_id, "telemetry": telemetry, "count": len(telemetry)}


@router.patch("/devices/{device_id}/status")
async def update_device_status(
    device_id: str,
    status: str,
    metadata: Optional[dict] = None,
    db: Session = Depends(get_db)
):
    result = iot_service.update_device_status(db, device_id, status, metadata)
    
    if not result:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return result


@router.get("/devices/{device_id}/stream")
async def stream_device_data(
    device_id: str,
    db: Session = Depends(get_db)
):
    return EventSourceResponse(
        iot_service.stream_device_data(db, device_id)
    )
