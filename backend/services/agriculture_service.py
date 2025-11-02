from typing import Dict, List, Any, Optional, AsyncGenerator
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json
import uuid

from backend.database import AgricultureData, CropRecommendation


class AgricultureService:
    def __init__(self):
        self.crop_database = {
            "wheat": {"optimal_temp": [15, 25], "water_needs": "moderate", "season": "cool"},
            "corn": {"optimal_temp": [20, 30], "water_needs": "high", "season": "warm"},
            "rice": {"optimal_temp": [25, 35], "water_needs": "very_high", "season": "warm"},
            "soybeans": {"optimal_temp": [20, 30], "water_needs": "moderate", "season": "warm"},
            "cotton": {"optimal_temp": [25, 35], "water_needs": "moderate", "season": "warm"},
            "potatoes": {"optimal_temp": [15, 20], "water_needs": "moderate", "season": "cool"},
            "tomatoes": {"optimal_temp": [20, 30], "water_needs": "moderate", "season": "warm"},
            "lettuce": {"optimal_temp": [10, 20], "water_needs": "moderate", "season": "cool"}
        }
    
    def record_farm_data(
        self,
        db: Session,
        farm_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        agriculture_data = AgricultureData(
            farm_id=farm_id,
            field_id=data.get("field_id"),
            crop_type=data.get("crop_type", "unknown"),
            location=data.get("location", {}),
            soil_data=data.get("soil_data", {}),
            climate_data=data.get("climate_data", {}),
            crop_health_score=data.get("crop_health_score"),
            growth_stage=data.get("growth_stage"),
            estimated_yield=data.get("estimated_yield"),
            irrigation_level=data.get("irrigation_level"),
            fertilizer_data=data.get("fertilizer_data", {}),
            pest_detection=data.get("pest_detection", {}),
            farm_metadata=data.get("metadata", {})
        )
        
        db.add(agriculture_data)
        db.commit()
        db.refresh(agriculture_data)
        
        return self._farm_data_to_dict(agriculture_data)
    
    def get_farm_data(
        self,
        db: Session,
        farm_id: str,
        field_id: Optional[str] = None,
        hours: int = 168,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        query = db.query(AgricultureData).filter(
            AgricultureData.farm_id == farm_id,
            AgricultureData.timestamp >= cutoff
        )
        
        if field_id:
            query = query.filter(AgricultureData.field_id == field_id)
        
        data = query.order_by(AgricultureData.timestamp.desc()).limit(limit).all()
        
        return [self._farm_data_to_dict(d) for d in data]
    
    def generate_crop_recommendation(
        self,
        db: Session,
        farm_id: str,
        soil_data: Dict[str, Any],
        climate_data: Dict[str, Any],
        field_id: Optional[str] = None
    ) -> Dict[str, Any]:
        recommendation_id = str(uuid.uuid4())
        
        best_crop, score, reasoning = self._analyze_conditions(
            soil_data, climate_data
        )
        
        recommendation = CropRecommendation(
            recommendation_id=recommendation_id,
            farm_id=farm_id,
            field_id=field_id,
            recommended_crop=best_crop,
            confidence_score=score,
            expected_yield=self._estimate_yield(best_crop, soil_data, climate_data),
            optimal_conditions=self._get_optimal_conditions(best_crop),
            seasonal_timing=self._get_seasonal_timing(best_crop),
            risk_factors=self._assess_risks(best_crop, soil_data, climate_data),
            reasoning=reasoning,
            recommendation_metadata={"soil_data": soil_data, "climate_data": climate_data}
        )
        
        db.add(recommendation)
        db.commit()
        db.refresh(recommendation)
        
        return self._recommendation_to_dict(recommendation)
    
    async def stream_crop_analysis(
        self,
        db: Session,
        farm_id: str
    ) -> AsyncGenerator[str, None]:
        yield json.dumps({
            "type": "status",
            "message": "🌱 Analyzing farm data..."
        }) + "\n"
        
        farm_data = self.get_farm_data(db, farm_id, hours=168, limit=100)
        
        if not farm_data:
            yield json.dumps({
                "type": "error",
                "message": "No farm data found"
            }) + "\n"
            return
        
        yield json.dumps({
            "type": "farm_data",
            "count": len(farm_data),
            "latest": farm_data[0] if farm_data else None
        }) + "\n"
        
        if farm_data:
            latest = farm_data[0]
            soil_data = latest.get("soil_data", {})
            climate_data = latest.get("climate_data", {})
            
            yield json.dumps({
                "type": "status",
                "message": "🌾 Generating recommendations..."
            }) + "\n"
            
            recommendation = self.generate_crop_recommendation(
                db, farm_id, soil_data, climate_data
            )
            
            yield json.dumps({
                "type": "recommendation",
                "data": recommendation
            }) + "\n"
        
        yield json.dumps({
            "type": "complete",
            "message": "Analysis complete"
        }) + "\n"
    
    def _analyze_conditions(
        self,
        soil_data: Dict[str, Any],
        climate_data: Dict[str, Any]
    ) -> tuple[str, float, str]:
        temp = climate_data.get("temperature", 20)
        moisture = soil_data.get("moisture", 50)
        ph = soil_data.get("ph", 6.5)
        
        best_crop = "wheat"
        best_score = 0.0
        
        for crop, requirements in self.crop_database.items():
            score = 50.0
            
            temp_min, temp_max = requirements["optimal_temp"]
            if temp_min <= temp <= temp_max:
                score += 30
            else:
                score -= abs(temp - (temp_min + temp_max) / 2) * 2
            
            if 6.0 <= ph <= 7.5:
                score += 10
            
            if requirements["water_needs"] == "high" and moisture > 60:
                score += 10
            elif requirements["water_needs"] == "moderate" and 40 <= moisture <= 70:
                score += 10
            elif requirements["water_needs"] == "low" and moisture < 50:
                score += 10
            
            if score > best_score:
                best_score = score
                best_crop = crop
        
        reasoning = f"Based on temperature ({temp}°C), soil moisture ({moisture}%), and pH ({ph}), {best_crop} is recommended."
        
        return best_crop, min(best_score, 100), reasoning
    
    def _estimate_yield(
        self,
        crop: str,
        soil_data: Dict[str, Any],
        climate_data: Dict[str, Any]
    ) -> float:
        base_yields = {
            "wheat": 3.5,
            "corn": 10.0,
            "rice": 7.0,
            "soybeans": 3.0,
            "cotton": 2.5,
            "potatoes": 40.0,
            "tomatoes": 60.0,
            "lettuce": 30.0
        }
        
        base = base_yields.get(crop, 5.0)
        
        moisture = soil_data.get("moisture", 50)
        if moisture < 30 or moisture > 80:
            base *= 0.8
        
        return round(base, 2)
    
    def _get_optimal_conditions(self, crop: str) -> Dict[str, Any]:
        return self.crop_database.get(crop, {})
    
    def _get_seasonal_timing(self, crop: str) -> Dict[str, Any]:
        crop_info = self.crop_database.get(crop, {})
        season = crop_info.get("season", "all_year")
        
        if season == "warm":
            return {"planting": "Spring", "harvest": "Fall"}
        elif season == "cool":
            return {"planting": "Fall", "harvest": "Spring"}
        else:
            return {"planting": "Any season", "harvest": "Year-round"}
    
    def _assess_risks(
        self,
        crop: str,
        soil_data: Dict[str, Any],
        climate_data: Dict[str, Any]
    ) -> List[str]:
        risks = []
        
        moisture = soil_data.get("moisture", 50)
        if moisture < 30:
            risks.append("Low soil moisture - irrigation needed")
        elif moisture > 80:
            risks.append("High soil moisture - drainage needed")
        
        temp = climate_data.get("temperature", 20)
        if temp < 10:
            risks.append("Low temperature - frost risk")
        elif temp > 35:
            risks.append("High temperature - heat stress risk")
        
        ph = soil_data.get("ph", 6.5)
        if ph < 5.5 or ph > 8.0:
            risks.append("Soil pH outside optimal range")
        
        return risks
    
    def _farm_data_to_dict(self, data: AgricultureData) -> Dict[str, Any]:
        return {
            "farm_id": data.farm_id,
            "field_id": data.field_id,
            "crop_type": data.crop_type,
            "location": data.location,
            "soil_data": data.soil_data,
            "climate_data": data.climate_data,
            "crop_health_score": data.crop_health_score,
            "growth_stage": data.growth_stage,
            "estimated_yield": data.estimated_yield,
            "irrigation_level": data.irrigation_level,
            "fertilizer_data": data.fertilizer_data,
            "pest_detection": data.pest_detection,
            "metadata": data.farm_metadata,
            "timestamp": data.timestamp.isoformat()
        }
    
    def _recommendation_to_dict(self, rec: CropRecommendation) -> Dict[str, Any]:
        return {
            "recommendation_id": rec.recommendation_id,
            "farm_id": rec.farm_id,
            "field_id": rec.field_id,
            "recommended_crop": rec.recommended_crop,
            "confidence_score": rec.confidence_score,
            "expected_yield": rec.expected_yield,
            "optimal_conditions": rec.optimal_conditions,
            "seasonal_timing": rec.seasonal_timing,
            "risk_factors": rec.risk_factors,
            "reasoning": rec.reasoning,
            "metadata": rec.recommendation_metadata,
            "created_at": rec.created_at.isoformat()
        }


agriculture_service = AgricultureService()
