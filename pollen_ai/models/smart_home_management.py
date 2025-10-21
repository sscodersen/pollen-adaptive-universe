"""
Smart Home Management for Pollen AI
Handles home automation, device control, and energy management
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import random


class SmartHomeDevice:
    """Represents a smart home device"""
    
    def __init__(self, device_id: str, device_type: str, name: str, room: str):
        self.device_id = device_id
        self.device_type = device_type  # light, thermostat, lock, camera, sensor, plug
        self.name = name
        self.room = room
        self.state = "off"
        self.properties = {}
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert device to dictionary"""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "name": self.name,
            "room": self.room,
            "state": self.state,
            "properties": self.properties,
            "last_updated": self.last_updated
        }


class SmartHomeManager:
    """Manages smart home devices and automation"""
    
    DEVICE_TYPES = {
        "light": {"properties": ["brightness", "color", "temperature"]},
        "thermostat": {"properties": ["temperature", "mode", "humidity"]},
        "lock": {"properties": ["locked", "battery"]},
        "camera": {"properties": ["recording", "motion_detection", "resolution"]},
        "sensor": {"properties": ["value", "unit", "alert_threshold"]},
        "plug": {"properties": ["power_usage", "schedule"]},
        "speaker": {"properties": ["volume", "playing", "source"]},
        "blinds": {"properties": ["position", "tilt"]},
        "fan": {"properties": ["speed", "oscillation"]},
        "vacuum": {"properties": ["cleaning_mode", "battery", "status"]}
    }
    
    def __init__(self):
        self.devices: Dict[str, SmartHomeDevice] = {}
        self.automation_rules: List[Dict[str, Any]] = []
        self.energy_data: List[Dict[str, Any]] = []
        self.rooms = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Office", "Garage"]
    
    def add_device(self, device_type: str, name: str, room: str) -> Dict[str, Any]:
        """Add a new smart home device"""
        if device_type not in self.DEVICE_TYPES:
            return {"success": False, "error": f"Unknown device type: {device_type}"}
        
        device_id = f"{device_type}_{len(self.devices) + 1}_{random.randint(1000, 9999)}"
        device = SmartHomeDevice(device_id, device_type, name, room)
        
        # Initialize default properties
        for prop in self.DEVICE_TYPES[device_type]["properties"]:
            device.properties[prop] = self._get_default_value(device_type, prop)
        
        self.devices[device_id] = device
        
        return {
            "success": True,
            "device": device.to_dict(),
            "message": f"Added {name} to {room}"
        }
    
    def control_device(self, device_id: str, action: str, value: Any = None) -> Dict[str, Any]:
        """Control a smart home device"""
        if device_id not in self.devices:
            return {"success": False, "error": f"Device not found: {device_id}"}
        
        device = self.devices[device_id]
        
        # Handle common actions
        if action == "turn_on":
            device.state = "on"
        elif action == "turn_off":
            device.state = "off"
        elif action == "toggle":
            device.state = "on" if device.state == "off" else "off"
        elif action in device.properties:
            device.properties[action] = value
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        device.last_updated = datetime.now().isoformat()
        
        # Track energy usage
        self._track_energy(device)
        
        return {
            "success": True,
            "device": device.to_dict(),
            "message": f"{action} executed on {device.name}"
        }
    
    def create_automation(self, name: str, trigger: Dict, actions: List[Dict]) -> Dict[str, Any]:
        """Create automation rule"""
        automation = {
            "id": f"auto_{len(self.automation_rules) + 1}",
            "name": name,
            "trigger": trigger,  # {"type": "time", "value": "18:00"} or {"type": "device_state", "device_id": "...", "state": "on"}
            "actions": actions,  # [{"device_id": "...", "action": "turn_on"}]
            "enabled": True,
            "created": datetime.now().isoformat()
        }
        
        self.automation_rules.append(automation)
        
        return {
            "success": True,
            "automation": automation,
            "message": f"Created automation: {name}"
        }
    
    def get_energy_report(self, timeframe: str = "day") -> Dict[str, Any]:
        """Get energy usage report"""
        total_usage = sum(e.get("usage", 0) for e in self.energy_data)
        device_breakdown = {}
        
        for entry in self.energy_data:
            device_id = entry.get("device_id")
            if device_id:
                device_breakdown[device_id] = device_breakdown.get(device_id, 0) + entry.get("usage", 0)
        
        # AI-generated recommendations
        recommendations = self._generate_energy_recommendations(total_usage, device_breakdown)
        
        return {
            "timeframe": timeframe,
            "total_usage_kwh": round(total_usage, 2),
            "cost_estimate": round(total_usage * 0.12, 2),  # $0.12/kWh
            "device_breakdown": device_breakdown,
            "recommendations": recommendations,
            "savings_potential": round(total_usage * 0.15, 2)  # 15% potential savings
        }
    
    def get_room_status(self, room: str) -> Dict[str, Any]:
        """Get status of all devices in a room"""
        room_devices = [d.to_dict() for d in self.devices.values() if d.room == room]
        
        active_devices = sum(1 for d in room_devices if d["state"] == "on")
        
        return {
            "room": room,
            "total_devices": len(room_devices),
            "active_devices": active_devices,
            "devices": room_devices
        }
    
    def suggest_automation(self, context: str) -> Dict[str, Any]:
        """AI-generated automation suggestions"""
        suggestions = []
        
        # Analyze usage patterns and suggest automations
        if "night" in context.lower() or "evening" in context.lower():
            suggestions.append({
                "name": "Evening Routine",
                "description": "Turn off lights and lock doors at 11 PM",
                "trigger": {"type": "time", "value": "23:00"},
                "actions": [
                    {"type": "turn_off_lights", "rooms": ["all"]},
                    {"type": "lock_doors", "devices": ["all_locks"]}
                ]
            })
        
        if "morning" in context.lower():
            suggestions.append({
                "name": "Morning Wake Up",
                "description": "Gradually increase bedroom light and adjust thermostat",
                "trigger": {"type": "time", "value": "07:00"},
                "actions": [
                    {"type": "adjust_light", "brightness": 70, "room": "Bedroom"},
                    {"type": "set_temperature", "value": 72, "device_type": "thermostat"}
                ]
            })
        
        if "energy" in context.lower() or "save" in context.lower():
            suggestions.append({
                "name": "Energy Saver",
                "description": "Turn off unused devices when away",
                "trigger": {"type": "location", "value": "away"},
                "actions": [
                    {"type": "turn_off_all", "except": ["security_camera", "lock"]}
                ]
            })
        
        return {
            "suggestions": suggestions,
            "context": context,
            "total_suggestions": len(suggestions)
        }
    
    def _get_default_value(self, device_type: str, property_name: str) -> Any:
        """Get default value for device property"""
        defaults = {
            "brightness": 100,
            "color": "#FFFFFF",
            "temperature": 72,
            "mode": "auto",
            "humidity": 45,
            "locked": True,
            "battery": 100,
            "recording": False,
            "motion_detection": True,
            "resolution": "1080p",
            "value": 0,
            "unit": "units",
            "alert_threshold": 100,
            "power_usage": 0,
            "schedule": "none",
            "volume": 50,
            "playing": False,
            "source": "none",
            "position": 50,
            "tilt": 0,
            "speed": 2,
            "oscillation": False,
            "cleaning_mode": "auto",
            "status": "idle"
        }
        return defaults.get(property_name, 0)
    
    def _track_energy(self, device: SmartHomeDevice):
        """Track energy usage for device"""
        if device.state == "on":
            # Estimate power usage based on device type
            power_map = {
                "light": 0.01,
                "thermostat": 0.5,
                "camera": 0.02,
                "plug": 0.1,
                "speaker": 0.05,
                "fan": 0.08,
                "vacuum": 0.3
            }
            
            usage = power_map.get(device.device_type, 0.05)
            
            self.energy_data.append({
                "device_id": device.device_id,
                "device_name": device.name,
                "usage": usage,
                "timestamp": datetime.now().isoformat()
            })
    
    def _generate_energy_recommendations(self, total_usage: float, breakdown: Dict) -> List[str]:
        """Generate AI-powered energy saving recommendations"""
        recommendations = []
        
        if total_usage > 10:
            recommendations.append("Consider upgrading to LED lights for 40% energy savings")
        
        if any("thermostat" in dev_id for dev_id in breakdown.keys()):
            recommendations.append("Smart temperature scheduling can reduce heating/cooling costs by 15%")
        
        recommendations.append("Turn off devices when not in use to save up to 20% on energy")
        recommendations.append("Use automation to optimize device usage patterns")
        
        return recommendations
    
    def get_all_devices(self) -> List[Dict[str, Any]]:
        """Get all devices"""
        return [d.to_dict() for d in self.devices.values()]
    
    def get_automation_rules(self) -> List[Dict[str, Any]]:
        """Get all automation rules"""
        return self.automation_rules
