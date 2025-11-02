from typing import Dict, List, Any, Optional, AsyncGenerator
from sqlalchemy.orm import Session
from datetime import datetime
import json
import uuid
import math

from backend.database import GeoOptimization


class GeoOptimizer:
    def __init__(self):
        self.supported_industries = [
            "retail", "manufacturing", "logistics", "agriculture",
            "energy", "healthcare", "transportation", "construction",
            "hospitality", "education", "real_estate"
        ]
    
    def optimize_location(
        self,
        db: Session,
        industry: str,
        location: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        optimization_id = str(uuid.uuid4())
        
        optimization_type = parameters.get("optimization_type", "general")
        
        recommendations = self._generate_recommendations(
            industry, location, parameters
        )
        
        efficiency_score = self._calculate_efficiency(
            industry, location, parameters
        )
        
        cost_savings = self._estimate_savings(
            industry, parameters, efficiency_score
        )
        
        environmental_impact = self._assess_environmental_impact(
            industry, parameters
        )
        
        implementation_steps = self._create_implementation_plan(
            industry, recommendations
        )
        
        optimization = GeoOptimization(
            optimization_id=optimization_id,
            industry=industry,
            location=location,
            optimization_type=optimization_type,
            parameters=parameters,
            recommendations=recommendations,
            efficiency_score=efficiency_score,
            cost_savings=cost_savings,
            environmental_impact=environmental_impact,
            implementation_steps=implementation_steps,
            optimization_metadata={}
        )
        
        db.add(optimization)
        db.commit()
        db.refresh(optimization)
        
        return self._optimization_to_dict(optimization)
    
    def get_optimizations(
        self,
        db: Session,
        industry: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = db.query(GeoOptimization)
        
        if industry:
            query = query.filter(GeoOptimization.industry == industry)
        
        optimizations = query.order_by(
            GeoOptimization.created_at.desc()
        ).limit(limit).all()
        
        return [self._optimization_to_dict(o) for o in optimizations]
    
    async def stream_optimization_analysis(
        self,
        db: Session,
        industry: str,
        location: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        yield json.dumps({
            "type": "status",
            "message": f"📍 Analyzing {industry} optimization for location..."
        }) + "\n"
        
        yield json.dumps({
            "type": "status",
            "message": "🔍 Evaluating geographical factors..."
        }) + "\n"
        
        yield json.dumps({
            "type": "status",
            "message": "💰 Calculating cost optimization..."
        }) + "\n"
        
        optimization = self.optimize_location(db, industry, location, parameters)
        
        yield json.dumps({
            "type": "optimization_result",
            "data": optimization
        }) + "\n"
        
        yield json.dumps({
            "type": "complete",
            "message": "Optimization analysis complete"
        }) + "\n"
    
    def _generate_recommendations(
        self,
        industry: str,
        location: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        recommendations = []
        
        lat = location.get("latitude", 0)
        lon = location.get("longitude", 0)
        
        if industry == "retail":
            recommendations.extend([
                {
                    "category": "Location",
                    "recommendation": "Position near high-traffic areas",
                    "priority": "high",
                    "impact": "Increase foot traffic by 30-40%"
                },
                {
                    "category": "Layout",
                    "recommendation": "Optimize product placement based on heat maps",
                    "priority": "medium",
                    "impact": "Improve conversion by 15-20%"
                }
            ])
        
        elif industry == "logistics":
            recommendations.extend([
                {
                    "category": "Route Optimization",
                    "recommendation": "Use AI-powered route planning",
                    "priority": "high",
                    "impact": "Reduce fuel costs by 20-25%"
                },
                {
                    "category": "Warehouse Location",
                    "recommendation": "Position near transport hubs",
                    "priority": "high",
                    "impact": "Decrease delivery time by 30%"
                }
            ])
        
        elif industry == "energy":
            recommendations.extend([
                {
                    "category": "Solar Potential",
                    "recommendation": "Install solar panels on south-facing surfaces",
                    "priority": "high",
                    "impact": "Generate 40-60% of energy needs"
                },
                {
                    "category": "Energy Storage",
                    "recommendation": "Implement battery storage systems",
                    "priority": "medium",
                    "impact": "Reduce grid dependency by 50%"
                }
            ])
        
        else:
            recommendations.append({
                "category": "General",
                "recommendation": "Analyze location-specific factors",
                "priority": "medium",
                "impact": "Varies by implementation"
            })
        
        return recommendations
    
    def _calculate_efficiency(
        self,
        industry: str,
        location: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> float:
        base_score = 50.0
        
        if "infrastructure_quality" in parameters:
            base_score += parameters["infrastructure_quality"] * 0.3
        
        if "market_size" in parameters:
            base_score += min(parameters["market_size"] / 10000, 20)
        
        if "competition" in parameters:
            base_score -= parameters["competition"] * 0.2
        
        if industry in ["logistics", "manufacturing"]:
            if "transport_access" in parameters:
                base_score += parameters["transport_access"] * 0.25
        
        return max(0, min(100, base_score))
    
    def _estimate_savings(
        self,
        industry: str,
        parameters: Dict[str, Any],
        efficiency_score: float
    ) -> float:
        base_budget = parameters.get("annual_budget", 1000000)
        
        savings_rate = (efficiency_score - 50) / 100
        
        estimated_savings = base_budget * savings_rate * 0.15
        
        return max(0, estimated_savings)
    
    def _assess_environmental_impact(
        self,
        industry: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        impact = {
            "carbon_reduction": 0,
            "energy_savings": 0,
            "waste_reduction": 0,
            "sustainability_score": 50
        }
        
        if parameters.get("renewable_energy", False):
            impact["carbon_reduction"] = 40
            impact["energy_savings"] = 30
            impact["sustainability_score"] += 25
        
        if parameters.get("waste_management", False):
            impact["waste_reduction"] = 35
            impact["sustainability_score"] += 15
        
        if parameters.get("local_sourcing", False):
            impact["carbon_reduction"] += 20
            impact["sustainability_score"] += 10
        
        impact["sustainability_score"] = min(100, impact["sustainability_score"])
        
        return impact
    
    def _create_implementation_plan(
        self,
        industry: str,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        steps = []
        
        high_priority = [r for r in recommendations if r.get("priority") == "high"]
        medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
        
        for i, rec in enumerate(high_priority):
            steps.append({
                "step": i + 1,
                "action": rec["recommendation"],
                "timeline": "Immediate (0-3 months)",
                "resources_needed": ["Budget allocation", "Team assignment"]
            })
        
        for i, rec in enumerate(medium_priority):
            steps.append({
                "step": len(high_priority) + i + 1,
                "action": rec["recommendation"],
                "timeline": "Short-term (3-6 months)",
                "resources_needed": ["Planning", "Resource allocation"]
            })
        
        return steps
    
    def _optimization_to_dict(self, opt: GeoOptimization) -> Dict[str, Any]:
        return {
            "optimization_id": opt.optimization_id,
            "industry": opt.industry,
            "location": opt.location,
            "optimization_type": opt.optimization_type,
            "parameters": opt.parameters,
            "recommendations": opt.recommendations,
            "efficiency_score": opt.efficiency_score,
            "cost_savings": opt.cost_savings,
            "environmental_impact": opt.environmental_impact,
            "implementation_steps": opt.implementation_steps,
            "metadata": opt.optimization_metadata,
            "created_at": opt.created_at.isoformat(),
            "updated_at": opt.updated_at.isoformat()
        }


geo_optimizer = GeoOptimizer()
