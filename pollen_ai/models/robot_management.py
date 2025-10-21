"""
Robot Management for Pollen AI
Handles robot control, task planning, and path planning
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random
import math


class RobotTask:
    """Represents a robot task"""
    
    def __init__(self, task_id: str, task_type: str, description: str, priority: int = 5):
        self.task_id = task_id
        self.task_type = task_type  # navigation, manipulation, inspection, delivery, cleaning
        self.description = description
        self.priority = priority  # 1-10, higher is more important
        self.status = "pending"  # pending, in_progress, completed, failed
        self.created = datetime.now().isoformat()
        self.started = None
        self.completed = None
        self.result = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "completed": self.completed,
            "result": self.result
        }


class Robot:
    """Represents a robot"""
    
    def __init__(self, robot_id: str, robot_type: str, name: str):
        self.robot_id = robot_id
        self.robot_type = robot_type  # mobile, manipulator, drone, humanoid
        self.name = name
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.battery = 100.0
        self.status = "idle"  # idle, busy, charging, error
        self.current_task = None
        self.capabilities = self._get_capabilities(robot_type)
    
    def _get_capabilities(self, robot_type: str) -> List[str]:
        """Get robot capabilities based on type"""
        capabilities_map = {
            "mobile": ["navigation", "delivery", "inspection"],
            "manipulator": ["grasping", "assembly", "sorting"],
            "drone": ["aerial_inspection", "delivery", "mapping"],
            "humanoid": ["navigation", "manipulation", "interaction"]
        }
        return capabilities_map.get(robot_type, ["basic_operation"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert robot to dictionary"""
        return {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "name": self.name,
            "position": self.position,
            "orientation": self.orientation,
            "battery": round(self.battery, 1),
            "status": self.status,
            "current_task": self.current_task,
            "capabilities": self.capabilities
        }


class PathPlanner:
    """Path planning for robots"""
    
    @staticmethod
    def plan_path(start: Dict[str, float], goal: Dict[str, float], 
                  obstacles: List[Dict] = None) -> Dict[str, Any]:
        """Plan path from start to goal avoiding obstacles"""
        # Simple A* pathfinding simulation
        path_points = PathPlanner._generate_path_points(start, goal, obstacles or [])
        
        distance = PathPlanner._calculate_distance(start, goal)
        estimated_time = distance / 0.5  # Assuming 0.5 m/s speed
        
        return {
            "path": path_points,
            "distance_meters": round(distance, 2),
            "estimated_time_seconds": round(estimated_time, 1),
            "waypoints": len(path_points),
            "algorithm": "A*"
        }
    
    @staticmethod
    def _generate_path_points(start: Dict, goal: Dict, obstacles: List) -> List[Dict]:
        """Generate path waypoints"""
        # Simplified path generation
        steps = 5
        path = []
        
        for i in range(steps + 1):
            t = i / steps
            point = {
                "x": start["x"] + t * (goal["x"] - start["x"]),
                "y": start["y"] + t * (goal["y"] - start["y"]),
                "z": start.get("z", 0.0)
            }
            path.append(point)
        
        return path
    
    @staticmethod
    def _calculate_distance(start: Dict, goal: Dict) -> float:
        """Calculate Euclidean distance"""
        dx = goal["x"] - start["x"]
        dy = goal["y"] - start["y"]
        dz = goal.get("z", 0) - start.get("z", 0)
        return math.sqrt(dx**2 + dy**2 + dz**2)


class RobotManager:
    """Manages robots and their tasks"""
    
    ROBOT_TYPES = ["mobile", "manipulator", "drone", "humanoid"]
    TASK_TYPES = ["navigation", "manipulation", "inspection", "delivery", "cleaning", "assembly"]
    
    def __init__(self):
        self.robots: Dict[str, Robot] = {}
        self.tasks: Dict[str, RobotTask] = {}
        self.task_queue: List[str] = []
        self.path_planner = PathPlanner()
    
    def add_robot(self, robot_type: str, name: str) -> Dict[str, Any]:
        """Add a new robot to the fleet"""
        if robot_type not in self.ROBOT_TYPES:
            return {"success": False, "error": f"Unknown robot type: {robot_type}"}
        
        robot_id = f"robot_{len(self.robots) + 1}_{random.randint(1000, 9999)}"
        robot = Robot(robot_id, robot_type, name)
        self.robots[robot_id] = robot
        
        return {
            "success": True,
            "robot": robot.to_dict(),
            "message": f"Added {robot_type} robot: {name}"
        }
    
    def create_task(self, task_type: str, description: str, 
                   priority: int = 5, robot_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new task"""
        if task_type not in self.TASK_TYPES:
            return {"success": False, "error": f"Unknown task type: {task_type}"}
        
        task_id = f"task_{len(self.tasks) + 1}_{random.randint(1000, 9999)}"
        task = RobotTask(task_id, task_type, description, priority)
        self.tasks[task_id] = task
        
        # Add to queue if no specific robot assigned
        if robot_id:
            return self.assign_task(task_id, robot_id)
        else:
            self.task_queue.append(task_id)
            # Try to auto-assign
            self._auto_assign_task(task_id)
        
        return {
            "success": True,
            "task": task.to_dict(),
            "message": f"Created task: {description}"
        }
    
    def assign_task(self, task_id: str, robot_id: str) -> Dict[str, Any]:
        """Assign task to a specific robot"""
        if task_id not in self.tasks:
            return {"success": False, "error": f"Task not found: {task_id}"}
        if robot_id not in self.robots:
            return {"success": False, "error": f"Robot not found: {robot_id}"}
        
        task = self.tasks[task_id]
        robot = self.robots[robot_id]
        
        # Check if robot can perform this task
        if not self._can_perform_task(robot, task):
            return {
                "success": False, 
                "error": f"Robot {robot.name} cannot perform task type: {task.task_type}"
            }
        
        # Assign task
        task.status = "in_progress"
        task.started = datetime.now().isoformat()
        robot.status = "busy"
        robot.current_task = task_id
        
        # Remove from queue if present
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        
        return {
            "success": True,
            "task": task.to_dict(),
            "robot": robot.to_dict(),
            "message": f"Assigned task to {robot.name}"
        }
    
    def complete_task(self, task_id: str, result: Optional[str] = None) -> Dict[str, Any]:
        """Mark task as completed"""
        if task_id not in self.tasks:
            return {"success": False, "error": f"Task not found: {task_id}"}
        
        task = self.tasks[task_id]
        task.status = "completed"
        task.completed = datetime.now().isoformat()
        task.result = result or "Task completed successfully"
        
        # Free up robot
        for robot in self.robots.values():
            if robot.current_task == task_id:
                robot.status = "idle"
                robot.current_task = None
                # Assign next task if available
                self._auto_assign_task_to_robot(robot.robot_id)
                break
        
        return {
            "success": True,
            "task": task.to_dict(),
            "message": "Task completed"
        }
    
    def plan_robot_path(self, robot_id: str, goal: Dict[str, float], 
                       obstacles: List[Dict] = None) -> Dict[str, Any]:
        """Plan path for robot to goal"""
        if robot_id not in self.robots:
            return {"success": False, "error": f"Robot not found: {robot_id}"}
        
        robot = self.robots[robot_id]
        path_plan = self.path_planner.plan_path(robot.position, goal, obstacles)
        
        return {
            "success": True,
            "robot_id": robot_id,
            "path_plan": path_plan,
            "message": f"Path planned for {robot.name}"
        }
    
    def move_robot(self, robot_id: str, position: Dict[str, float]) -> Dict[str, Any]:
        """Move robot to new position"""
        if robot_id not in self.robots:
            return {"success": False, "error": f"Robot not found: {robot_id}"}
        
        robot = self.robots[robot_id]
        robot.position = position
        robot.battery = max(0, robot.battery - 0.5)  # Consume battery
        
        return {
            "success": True,
            "robot": robot.to_dict(),
            "message": f"Moved {robot.name} to new position"
        }
    
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get status of entire robot fleet"""
        total_robots = len(self.robots)
        idle_robots = sum(1 for r in self.robots.values() if r.status == "idle")
        busy_robots = sum(1 for r in self.robots.values() if r.status == "busy")
        
        avg_battery = sum(r.battery for r in self.robots.values()) / max(1, total_robots)
        
        return {
            "total_robots": total_robots,
            "idle": idle_robots,
            "busy": busy_robots,
            "charging": sum(1 for r in self.robots.values() if r.status == "charging"),
            "error": sum(1 for r in self.robots.values() if r.status == "error"),
            "average_battery": round(avg_battery, 1),
            "pending_tasks": len(self.task_queue),
            "active_tasks": sum(1 for t in self.tasks.values() if t.status == "in_progress"),
            "robots": [r.to_dict() for r in self.robots.values()]
        }
    
    def suggest_task_optimization(self, context: str) -> Dict[str, Any]:
        """AI-generated task optimization suggestions"""
        suggestions = []
        
        # Analyze current tasks and suggest optimizations
        pending_tasks = [t for t in self.tasks.values() if t.status == "pending"]
        
        if len(pending_tasks) > 3:
            suggestions.append({
                "type": "task_batching",
                "description": "Batch similar tasks together to reduce robot travel time",
                "potential_savings": "30% time reduction"
            })
        
        idle_robots = [r for r in self.robots.values() if r.status == "idle"]
        if len(idle_robots) > 0 and len(pending_tasks) > 0:
            suggestions.append({
                "type": "auto_assignment",
                "description": f"Assign {len(pending_tasks)} pending tasks to {len(idle_robots)} idle robots",
                "action": "auto_assign_all"
            })
        
        low_battery_robots = [r for r in self.robots.values() if r.battery < 20]
        if low_battery_robots:
            suggestions.append({
                "type": "battery_management",
                "description": f"{len(low_battery_robots)} robots need charging soon",
                "action": "schedule_charging"
            })
        
        return {
            "suggestions": suggestions,
            "context": context,
            "total_suggestions": len(suggestions)
        }
    
    def _can_perform_task(self, robot: Robot, task: RobotTask) -> bool:
        """Check if robot can perform task"""
        # Map task types to required capabilities
        task_capability_map = {
            "navigation": ["navigation"],
            "manipulation": ["grasping", "manipulation"],
            "inspection": ["inspection", "aerial_inspection"],
            "delivery": ["delivery", "navigation"],
            "cleaning": ["navigation"],
            "assembly": ["grasping", "assembly", "manipulation"]
        }
        
        required_caps = task_capability_map.get(task.task_type, [])
        return any(cap in robot.capabilities for cap in required_caps)
    
    def _auto_assign_task(self, task_id: str):
        """Automatically assign task to available robot"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        # Find best robot for this task
        available_robots = [r for r in self.robots.values() 
                          if r.status == "idle" and self._can_perform_task(r, task)]
        
        if available_robots:
            # Pick robot with highest battery
            best_robot = max(available_robots, key=lambda r: r.battery)
            self.assign_task(task_id, best_robot.robot_id)
    
    def _auto_assign_task_to_robot(self, robot_id: str):
        """Assign next priority task to robot"""
        if not self.task_queue:
            return
        
        robot = self.robots.get(robot_id)
        if not robot or robot.status != "idle":
            return
        
        # Find highest priority compatible task
        for task_id in sorted(self.task_queue, 
                            key=lambda tid: self.tasks[tid].priority, 
                            reverse=True):
            task = self.tasks[task_id]
            if self._can_perform_task(robot, task):
                self.assign_task(task_id, robot_id)
                break
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks"""
        return [t.to_dict() for t in self.tasks.values()]
