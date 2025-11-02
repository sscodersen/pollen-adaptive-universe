from typing import Dict, List, Any, Optional, AsyncGenerator
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import hashlib
import json
import uuid

from backend.database import IoTDevice, DeviceTelemetry


class IoTService:
    def __init__(self):
        self.supported_device_types = [
            "smart_fridge", "smart_tv", "smart_thermostat", "robot",
            "sensor", "wearable", "vehicle", "industrial_equipment",
            "medical_device", "home_assistant", "camera", "speaker"
        ]
    
    def register_device(
        self,
        db: Session,
        device_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        device_id = device_data.get("device_id") or str(uuid.uuid4())
        
        existing = db.query(IoTDevice).filter(
            IoTDevice.device_id == device_id
        ).first()
        
        if existing:
            return {"error": "Device already registered", "device_id": device_id}
        
        device = IoTDevice(
            device_id=device_id,
            device_type=device_data.get("device_type", "unknown"),
            device_name=device_data.get("device_name", f"Device {device_id[:8]}"),
            manufacturer=device_data.get("manufacturer"),
            model=device_data.get("model"),
            os_version=device_data.get("os_version"),
            firmware_version=device_data.get("firmware_version"),
            status="online",
            location=device_data.get("location", {}),
            capabilities=device_data.get("capabilities", []),
            device_metadata=device_data.get("metadata", {}),
            last_seen=datetime.utcnow()
        )
        
        db.add(device)
        db.commit()
        db.refresh(device)
        
        return self._device_to_dict(device)
    
    def update_device_status(
        self,
        db: Session,
        device_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        device = db.query(IoTDevice).filter(
            IoTDevice.device_id == device_id
        ).first()
        
        if not device:
            return None
        
        device.status = status
        device.last_seen = datetime.utcnow()
        
        if metadata:
            current_metadata = device.device_metadata if device.device_metadata else {}
            device.device_metadata = {**current_metadata, **metadata}
        
        db.commit()
        db.refresh(device)
        
        return self._device_to_dict(device)
    
    def record_telemetry(
        self,
        db: Session,
        device_id: str,
        telemetry_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        telemetry = DeviceTelemetry(
            device_id=device_id,
            telemetry_type=telemetry_data.get("type", "general"),
            sensor_data=telemetry_data.get("sensor_data", {}),
            value=telemetry_data.get("value"),
            unit=telemetry_data.get("unit"),
            telemetry_metadata=telemetry_data.get("metadata", {})
        )
        
        db.add(telemetry)
        
        device = db.query(IoTDevice).filter(
            IoTDevice.device_id == device_id
        ).first()
        
        if device:
            device.last_seen = datetime.utcnow()
            device.status = "online"
        
        db.commit()
        
        return {
            "device_id": device_id,
            "telemetry_recorded": True,
            "timestamp": telemetry.timestamp.isoformat()
        }
    
    def get_devices(
        self,
        db: Session,
        device_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = db.query(IoTDevice)
        
        if device_type:
            query = query.filter(IoTDevice.device_type == device_type)
        
        if status:
            query = query.filter(IoTDevice.status == status)
        
        devices = query.order_by(IoTDevice.last_seen.desc()).limit(limit).all()
        
        return [self._device_to_dict(d) for d in devices]
    
    def get_device_telemetry(
        self,
        db: Session,
        device_id: str,
        hours: int = 24,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        telemetry = db.query(DeviceTelemetry).filter(
            DeviceTelemetry.device_id == device_id,
            DeviceTelemetry.timestamp >= cutoff
        ).order_by(DeviceTelemetry.timestamp.desc()).limit(limit).all()
        
        return [self._telemetry_to_dict(t) for t in telemetry]
    
    async def stream_device_data(
        self,
        db: Session,
        device_id: str
    ) -> AsyncGenerator[str, None]:
        device = db.query(IoTDevice).filter(
            IoTDevice.device_id == device_id
        ).first()
        
        if not device:
            yield json.dumps({"error": "Device not found"}) + "\n"
            return
        
        yield json.dumps({
            "type": "device_info",
            "data": self._device_to_dict(device)
        }) + "\n"
        
        telemetry = self.get_device_telemetry(db, device_id, hours=24)
        
        yield json.dumps({
            "type": "telemetry_data",
            "count": len(telemetry),
            "data": telemetry[:100]
        }) + "\n"
        
        yield json.dumps({
            "type": "complete",
            "message": "Device data stream complete"
        }) + "\n"
    
    def _device_to_dict(self, device: IoTDevice) -> Dict[str, Any]:
        return {
            "device_id": device.device_id,
            "device_type": device.device_type,
            "device_name": device.device_name,
            "manufacturer": device.manufacturer,
            "model": device.model,
            "os_version": device.os_version,
            "firmware_version": device.firmware_version,
            "status": device.status,
            "location": device.location,
            "capabilities": device.capabilities,
            "metadata": device.device_metadata,
            "last_seen": device.last_seen.isoformat() if device.last_seen else None,
            "registered_at": device.registered_at.isoformat()
        }
    
    def _telemetry_to_dict(self, telemetry: DeviceTelemetry) -> Dict[str, Any]:
        return {
            "device_id": telemetry.device_id,
            "telemetry_type": telemetry.telemetry_type,
            "sensor_data": telemetry.sensor_data,
            "value": telemetry.value,
            "unit": telemetry.unit,
            "timestamp": telemetry.timestamp.isoformat(),
            "metadata": telemetry.telemetry_metadata
        }


iot_service = IoTService()
