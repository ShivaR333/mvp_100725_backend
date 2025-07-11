import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from app.graph import (
    discover_from_data,
    fuse_graphs,
    fit_scm,
    simulate_interventions,
    optimize_with_surrogate,
    load_expert_edges_from_sqlite
)
from connect.schema_crawler import crawl_schema
from connect.canonical_mapper import map_to_canonical
from app.document_parser import extract_edges_from_docs

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of supported workflows."""
    BUILD_CAUSAL_TWIN = "build_causal_twin"
    WHAT_IF = "what_if"
    OPTIMIZE = "optimize"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    step_id: str
    tool_name: str
    description: str
    parameters: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class WorkflowExecution:
    """Represents a complete workflow execution."""
    workflow_id: str
    workflow_type: WorkflowType
    user_goal: str
    steps: List[WorkflowStep]
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    final_result: Optional[Dict[str, Any]] = None
    project_path: str = ""


class PlannerAgent:
    """
    Agent that orchestrates multi-step causal analysis workflows.
    
    Supports three main workflow types:
    1. Build Causal Twin: Full pipeline from schema to fitted SCM
    2. What-if Analysis: Simulation with interventions
    3. Optimization: Find optimal decision variables
    """
    
    def __init__(self, project_memory_path: str = "project_memory.jsonl"):
        self.project_memory_path = project_memory_path
        self.current_execution: Optional[WorkflowExecution] = None
        
        # Tool mappings
        self.tools = {
            "schema_crawler": self._run_schema_crawler,
            "canonical_mapper": self._run_canonical_mapper,
            "doc_ingest": self._run_doc_ingest,
            "causal_fuse": self._run_causal_fuse,
            "scm_fit": self._run_scm_fit,
            "simulate": self._run_simulate,
            "optimize": self._run_optimize
        }
    
    def parse_user_goal(self, user_goal: str) -> WorkflowType:
        """Parse user goal to determine workflow type."""
        goal_lower = user_goal.lower()
        
        if any(keyword in goal_lower for keyword in ["build", "causal twin", "create twin", "construct"]):
            return WorkflowType.BUILD_CAUSAL_TWIN
        elif any(keyword in goal_lower for keyword in ["what if", "simulate", "intervention", "scenario"]):
            return WorkflowType.WHAT_IF
        elif any(keyword in goal_lower for keyword in ["optimize", "optimise", "maximize", "minimize", "best"]):
            return WorkflowType.OPTIMIZE
        else:
            # Default to what-if for ambiguous cases
            return WorkflowType.WHAT_IF
    
    def create_workflow_template(self, workflow_type: WorkflowType, parameters: Dict[str, Any]) -> List[WorkflowStep]:
        """Create workflow template based on type and parameters."""
        
        if workflow_type == WorkflowType.BUILD_CAUSAL_TWIN:
            return self._create_causal_twin_template(parameters)
        elif workflow_type == WorkflowType.WHAT_IF:
            return self._create_what_if_template(parameters)
        elif workflow_type == WorkflowType.OPTIMIZE:
            return self._create_optimize_template(parameters)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    def _create_causal_twin_template(self, parameters: Dict[str, Any]) -> List[WorkflowStep]:
        """Create causal twin building workflow."""
        steps = [
            WorkflowStep(
                step_id="1_schema_crawler",
                tool_name="schema_crawler",
                description="Crawl database schema to discover tables and columns",
                parameters={
                    "connection_string": parameters.get("connection_string", ""),
                    "output_path": parameters.get("schema_output", "raw_schema.json")
                }
            ),
            WorkflowStep(
                step_id="2_canonical_mapper",
                tool_name="canonical_mapper",
                description="Map raw schema to canonical format",
                parameters={
                    "raw_schema_path": parameters.get("schema_output", "raw_schema.json"),
                    "canonical_dict_path": parameters.get("canonical_dict", "canonical_dict.json"),
                    "raw_data_dir": parameters.get("raw_data_dir", "raw_data"),
                    "clean_data_dir": parameters.get("clean_data_dir", "clean_data")
                }
            ),
            WorkflowStep(
                step_id="3_doc_ingest",
                tool_name="doc_ingest",
                description="Extract causal edges from documentation",
                parameters={
                    "doc_dir": parameters.get("doc_dir", "documents"),
                    "output_db": parameters.get("edge_db", "edge_candidates.db")
                }
            ),
            WorkflowStep(
                step_id="4_causal_fuse",
                tool_name="causal_fuse",
                description="Fuse expert knowledge with data-driven causal discovery",
                parameters={
                    "clean_data_dir": parameters.get("clean_data_dir", "clean_data"),
                    "edge_db_path": parameters.get("edge_db", "edge_candidates.db"),
                    "confidence_threshold": parameters.get("confidence_threshold", 0.6),
                    "output_graph": parameters.get("fused_graph", "fused_graph.json"),
                    "whitelist": parameters.get("whitelist", []),
                    "blacklist": parameters.get("blacklist", [])
                }
            ),
            WorkflowStep(
                step_id="5_scm_fit",
                tool_name="scm_fit",
                description="Fit structural causal model using econML",
                parameters={
                    "graph_path": parameters.get("fused_graph", "fused_graph.json"),
                    "clean_data_dir": parameters.get("clean_data_dir", "clean_data"),
                    "outcome_nodes": parameters.get("outcome_nodes", None),
                    "n_boot": parameters.get("n_boot", 250),
                    "model_path": parameters.get("model_path", "model.pkl")
                }
            )
        ]
        return steps
    
    def _create_what_if_template(self, parameters: Dict[str, Any]) -> List[WorkflowStep]:
        """Create what-if analysis workflow."""
        steps = [
            WorkflowStep(
                step_id="1_simulate",
                tool_name="simulate",
                description="Run Monte Carlo simulation with interventions",
                parameters={
                    "interventions": parameters.get("interventions", {}),
                    "n_samples": parameters.get("n_samples", 1000),
                    "graph_path": parameters.get("graph_path", "fused_graph.json"),
                    "model_path": parameters.get("model_path", "model.pkl"),
                    "kpi_nodes": parameters.get("kpi_nodes", None)
                }
            )
        ]
        return steps
    
    def _create_optimize_template(self, parameters: Dict[str, Any]) -> List[WorkflowStep]:
        """Create optimization workflow."""
        steps = [
            WorkflowStep(
                step_id="1_optimize",
                tool_name="optimize",
                description="Optimize decision variables using quadratic surrogate",
                parameters={
                    "decision_vars": parameters.get("decision_vars", []),
                    "objective": parameters.get("objective", "max"),
                    "target_kpi": parameters.get("kpi", ""),
                    "constraints": parameters.get("constraints", []),
                    "decision_bounds": parameters.get("decision_bounds", {}),
                    "graph_path": parameters.get("graph_path", "fused_graph.json"),
                    "model_path": parameters.get("model_path", "model.pkl"),
                    "n_surrogate_samples": parameters.get("n_surrogate_samples", 200)
                }
            )
        ]
        return steps
    
    async def execute_workflow(self, user_goal: str, parameters: Dict[str, Any]) -> WorkflowExecution:
        """Execute a complete workflow based on user goal."""
        
        # Parse workflow type
        workflow_type = self.parse_user_goal(user_goal)
        
        # Create workflow execution
        workflow_id = f"{workflow_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create workflow steps
        steps = self.create_workflow_template(workflow_type, parameters)
        
        # Initialize execution
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            user_goal=user_goal,
            steps=steps,
            start_time=datetime.now().isoformat(),
            project_path=parameters.get("project_path", "")
        )
        
        self.current_execution = execution
        execution.status = "running"
        
        logger.info(f"Starting workflow execution: {workflow_id}")
        logger.info(f"Workflow type: {workflow_type.value}")
        logger.info(f"Steps: {len(steps)}")
        
        try:
            # Execute steps sequentially
            for i, step in enumerate(steps):
                logger.info(f"Executing step {i+1}/{len(steps)}: {step.description}")
                
                step.status = "running"
                step.start_time = datetime.now().isoformat()
                
                try:
                    # Run the tool
                    tool_func = self.tools[step.tool_name]
                    result = await tool_func(step.parameters)
                    
                    step.result = result
                    step.status = "completed"
                    step.end_time = datetime.now().isoformat()
                    
                    logger.info(f"Step {i+1} completed successfully")
                    
                    # Update parameters for next steps if needed
                    self._update_downstream_parameters(steps, i, result)
                    
                except Exception as e:
                    step.error = str(e)
                    step.status = "failed"
                    step.end_time = datetime.now().isoformat()
                    
                    logger.error(f"Step {i+1} failed: {e}")
                    execution.status = "failed"
                    break
            
            # Set final execution status
            if execution.status != "failed":
                execution.status = "completed"
                
                # Create summary result
                execution.final_result = self._create_workflow_summary(execution)
            
            execution.end_time = datetime.now().isoformat()
            
            # Save to project memory
            self._save_to_project_memory(execution)
            
            logger.info(f"Workflow execution completed: {execution.status}")
            
        except Exception as e:
            execution.status = "failed"
            execution.end_time = datetime.now().isoformat()
            logger.error(f"Workflow execution failed: {e}")
        
        return execution
    
    def _update_downstream_parameters(self, steps: List[WorkflowStep], completed_step_idx: int, result: Dict[str, Any]):
        """Update parameters for downstream steps based on completed step results."""
        
        # Update based on completed step
        completed_step = steps[completed_step_idx]
        
        if completed_step.tool_name == "schema_crawler":
            # Schema crawler output goes to canonical mapper
            for step in steps[completed_step_idx + 1:]:
                if step.tool_name == "canonical_mapper":
                    step.parameters["raw_schema_path"] = result.get("output_path", step.parameters["raw_schema_path"])
        
        elif completed_step.tool_name == "canonical_mapper":
            # Canonical mapper output goes to causal fusion
            for step in steps[completed_step_idx + 1:]:
                if step.tool_name == "causal_fuse":
                    step.parameters["clean_data_dir"] = result.get("clean_data_dir", step.parameters["clean_data_dir"])
        
        elif completed_step.tool_name == "doc_ingest":
            # Document ingest creates edge database for fusion
            for step in steps[completed_step_idx + 1:]:
                if step.tool_name == "causal_fuse":
                    step.parameters["edge_db_path"] = result.get("edge_db_path", step.parameters["edge_db_path"])
                    # Extract edges as whitelist
                    if "edges" in result:
                        whitelist = [(edge["source"], edge["target"]) for edge in result["edges"]]
                        step.parameters["whitelist"] = whitelist
        
        elif completed_step.tool_name == "causal_fuse":
            # Fused graph goes to SCM fitting
            for step in steps[completed_step_idx + 1:]:
                if step.tool_name == "scm_fit":
                    step.parameters["graph_path"] = result.get("output_graph", step.parameters["graph_path"])
    
    def _create_workflow_summary(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Create summary of workflow execution."""
        
        summary = {
            "workflow_id": execution.workflow_id,
            "workflow_type": execution.workflow_type.value,
            "user_goal": execution.user_goal,
            "status": execution.status,
            "start_time": execution.start_time,
            "end_time": execution.end_time,
            "total_steps": len(execution.steps),
            "completed_steps": sum(1 for step in execution.steps if step.status == "completed"),
            "step_results": {}
        }
        
        # Extract key results from each step
        for step in execution.steps:
            if step.status == "completed" and step.result:
                summary["step_results"][step.step_id] = {
                    "tool": step.tool_name,
                    "description": step.description,
                    "key_outputs": self._extract_key_outputs(step.tool_name, step.result)
                }
        
        # Add workflow-specific summaries
        if execution.workflow_type == WorkflowType.BUILD_CAUSAL_TWIN:
            summary.update(self._summarize_causal_twin(execution))
        elif execution.workflow_type == WorkflowType.WHAT_IF:
            summary.update(self._summarize_what_if(execution))
        elif execution.workflow_type == WorkflowType.OPTIMIZE:
            summary.update(self._summarize_optimize(execution))
        
        return summary
    
    def _extract_key_outputs(self, tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key outputs from tool results."""
        
        if tool_name == "schema_crawler":
            return {
                "tables_found": result.get("n_tables", 0),
                "columns_found": result.get("n_columns", 0),
                "output_file": result.get("output_path", "")
            }
        
        elif tool_name == "canonical_mapper":
            return {
                "mappings_created": result.get("n_mappings", 0),
                "canonical_files": result.get("canonical_files", {}),
                "review_items": result.get("n_review_items", 0)
            }
        
        elif tool_name == "doc_ingest":
            return {
                "documents_processed": result.get("n_documents", 0),
                "edges_extracted": result.get("n_edges", 0),
                "confidence_range": result.get("confidence_range", [])
            }
        
        elif tool_name == "causal_fuse":
            return {
                "expert_edges": result.get("expert_edges", 0),
                "data_edges": result.get("data_edges", 0),
                "fused_nodes": result.get("fused_nodes", 0),
                "fused_edges": result.get("fused_edges", 0),
                "is_acyclic": result.get("is_acyclic", False)
            }
        
        elif tool_name == "scm_fit":
            return {
                "fitted_nodes": result.get("fitted_nodes", []),
                "total_models": result.get("total_models", 0),
                "model_types": result.get("model_details", {})
            }
        
        elif tool_name == "simulate":
            return {
                "interventions": result.get("simulation_config", {}).get("interventions", {}),
                "n_samples": result.get("simulation_config", {}).get("n_samples", 0),
                "kpi_results": list(result.get("results", {}).keys())
            }
        
        elif tool_name == "optimize":
            return {
                "optimal_decisions": result.get("optimal_decisions", {}),
                "target_kpi": result.get("optimization_config", {}).get("target_kpi", ""),
                "expected_value": result.get("expected_kpi", {}).get("mean", 0),
                "constraints_satisfied": result.get("constraint_validation", {})
            }
        
        return {}
    
    def _summarize_causal_twin(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Create summary for causal twin building workflow."""
        
        # Find final SCM fitting result
        scm_result = None
        for step in execution.steps:
            if step.tool_name == "scm_fit" and step.status == "completed":
                scm_result = step.result
                break
        
        causal_twin_summary = {
            "causal_twin_built": scm_result is not None,
            "models_fitted": scm_result.get("total_models", 0) if scm_result else 0,
            "outcome_variables": scm_result.get("fitted_nodes", []) if scm_result else [],
            "artifacts_created": []
        }
        
        # Collect artifacts created during workflow
        for step in execution.steps:
            if step.status == "completed" and step.result:
                if "output_path" in step.result:
                    causal_twin_summary["artifacts_created"].append(step.result["output_path"])
                if "output_graph" in step.result:
                    causal_twin_summary["artifacts_created"].append(step.result["output_graph"])
                if "model_output_path" in step.result:
                    causal_twin_summary["artifacts_created"].append(step.result["model_output_path"])
        
        return {"causal_twin_summary": causal_twin_summary}
    
    def _summarize_what_if(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Create summary for what-if analysis workflow."""
        
        sim_step = execution.steps[0] if execution.steps else None
        
        if sim_step and sim_step.status == "completed":
            sim_result = sim_step.result
            
            return {
                "what_if_summary": {
                    "interventions_applied": sim_result.get("simulation_config", {}).get("interventions", {}),
                    "kpis_analyzed": list(sim_result.get("results", {}).keys()),
                    "simulation_method": sim_result.get("simulation_config", {}).get("method", ""),
                    "sample_size": sim_result.get("simulation_config", {}).get("n_samples", 0)
                }
            }
        
        return {"what_if_summary": {"status": "failed"}}
    
    def _summarize_optimize(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Create summary for optimization workflow."""
        
        opt_step = execution.steps[0] if execution.steps else None
        
        if opt_step and opt_step.status == "completed":
            opt_result = opt_step.result
            
            return {
                "optimization_summary": {
                    "decision_variables": opt_result.get("optimization_config", {}).get("decision_vars", []),
                    "objective": opt_result.get("optimization_config", {}).get("objective", ""),
                    "target_kpi": opt_result.get("optimization_config", {}).get("target_kpi", ""),
                    "optimal_decisions": opt_result.get("optimal_decisions", {}),
                    "expected_improvement": opt_result.get("expected_kpi", {}).get("mean", 0),
                    "constraints_satisfied": all(
                        constraint.get("satisfied", False) 
                        for constraint in opt_result.get("constraint_validation", {}).values()
                    )
                }
            }
        
        return {"optimization_summary": {"status": "failed"}}
    
    def _save_to_project_memory(self, execution: WorkflowExecution):
        """Save workflow execution to project memory (JSONL format)."""
        
        try:
            # Convert execution to dict
            execution_dict = asdict(execution)
            
            # Convert enum to string
            execution_dict["workflow_type"] = execution.workflow_type.value
            
            # Create JSONL entry
            memory_entry = {
                "timestamp": datetime.now().isoformat(),
                "execution": execution_dict
            }
            
            # Append to project memory file
            memory_path = Path(execution.project_path) / self.project_memory_path if execution.project_path else Path(self.project_memory_path)
            
            # Ensure directory exists
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(memory_path, "a") as f:
                f.write(json.dumps(memory_entry) + "\n")
            
            logger.info(f"Saved workflow execution to project memory: {memory_path}")
            
        except Exception as e:
            logger.error(f"Failed to save to project memory: {e}")
    
    def load_project_memory(self, project_path: str = "") -> List[Dict[str, Any]]:
        """Load workflow executions from project memory."""
        
        memory_path = Path(project_path) / self.project_memory_path if project_path else Path(self.project_memory_path)
        
        if not memory_path.exists():
            return []
        
        executions = []
        try:
            with open(memory_path, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        executions.append(entry)
            
            logger.info(f"Loaded {len(executions)} workflow executions from project memory")
            
        except Exception as e:
            logger.error(f"Failed to load project memory: {e}")
        
        return executions
    
    # Tool implementations
    async def _run_schema_crawler(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run schema crawler tool."""
        logger.info("Running schema crawler")
        
        # This would call the actual schema crawler
        # For now, return mock result
        return {
            "status": "success",
            "output_path": parameters["output_path"],
            "n_tables": 5,
            "n_columns": 25
        }
    
    async def _run_canonical_mapper(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run canonical mapper tool."""
        logger.info("Running canonical mapper")
        
        # This would call the actual canonical mapper
        return {
            "status": "success",
            "clean_data_dir": parameters["clean_data_dir"],
            "n_mappings": 15,
            "canonical_files": ["temperature.parquet", "pressure.parquet"],
            "n_review_items": 3
        }
    
    async def _run_doc_ingest(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run document ingestion tool."""
        logger.info("Running document ingestion")
        
        try:
            doc_dir = Path(parameters["doc_dir"])
            if doc_dir.exists():
                df = extract_edges_from_docs(doc_dir)
                
                edges = []
                if not df.empty:
                    edges = df.to_dict("records")
                
                return {
                    "status": "success",
                    "edge_db_path": parameters["output_db"],
                    "n_documents": len(list(doc_dir.glob("*.pdf"))),
                    "n_edges": len(edges),
                    "edges": edges,
                    "confidence_range": [df["confidence"].min(), df["confidence"].max()] if not df.empty else [0, 0]
                }
            else:
                return {
                    "status": "success",
                    "edge_db_path": parameters["output_db"],
                    "n_documents": 0,
                    "n_edges": 0,
                    "edges": [],
                    "confidence_range": [0, 0]
                }
                
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_causal_fuse(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run causal graph fusion tool."""
        logger.info("Running causal graph fusion")
        
        try:
            # Load expert edges
            expert_graph = load_expert_edges_from_sqlite(
                parameters["edge_db_path"],
                parameters["confidence_threshold"]
            )
            
            # Run data-driven discovery
            data_graph = discover_from_data(
                parameters["clean_data_dir"],
                whitelist=parameters.get("whitelist"),
                blacklist=parameters.get("blacklist")
            )
            
            # Fuse graphs
            fused_graph = fuse_graphs(expert_graph, data_graph)
            
            # Save fused graph
            fused_graph.save_to_json(parameters["output_graph"])
            
            return {
                "status": "success",
                "expert_edges": len(expert_graph.edges()),
                "data_edges": len(data_graph.edges()),
                "fused_nodes": len(fused_graph.nodes()),
                "fused_edges": len(fused_graph.edges()),
                "is_acyclic": fused_graph.is_acyclic(),
                "output_graph": parameters["output_graph"]
            }
            
        except Exception as e:
            logger.error(f"Causal fusion failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_scm_fit(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run SCM fitting tool."""
        logger.info("Running SCM fitting")
        
        try:
            model_dict = fit_scm(
                graph_path=parameters["graph_path"],
                clean_dir=parameters["clean_data_dir"],
                outcome_nodes=parameters.get("outcome_nodes"),
                n_boot=parameters["n_boot"],
                model_output_path=parameters["model_path"]
            )
            
            return {
                "status": "success",
                "fitted_nodes": model_dict["metadata"]["fitted_nodes"],
                "total_models": len(model_dict["models"]),
                "model_output_path": parameters["model_path"],
                "bootstrap_samples": parameters["n_boot"],
                "model_details": {
                    node: {
                        "parents": info["parents"],
                        "model_type": info["model_type"],
                        "n_samples": info["n_samples"]
                    }
                    for node, info in model_dict["models"].items()
                }
            }
            
        except Exception as e:
            logger.error(f"SCM fitting failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_simulate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run simulation tool."""
        logger.info("Running simulation")
        
        try:
            results = simulate_interventions(
                graph_path=parameters["graph_path"],
                model_path=parameters["model_path"],
                interventions=parameters["interventions"],
                n_samples=parameters["n_samples"],
                kpi_nodes=parameters.get("kpi_nodes")
            )
            
            return {
                "status": "success",
                "simulation_config": {
                    "interventions": parameters["interventions"],
                    "n_samples": parameters["n_samples"],
                    "method": "manual_mc",
                    "kpi_nodes": list(results.keys())
                },
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_optimize(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimization tool."""
        logger.info("Running optimization")
        
        try:
            result = optimize_with_surrogate(
                graph_path=parameters["graph_path"],
                model_path=parameters["model_path"],
                decision_vars=parameters["decision_vars"],
                objective=parameters["objective"],
                target_kpi=parameters["target_kpi"],
                constraints=parameters.get("constraints"),
                decision_bounds=parameters.get("decision_bounds"),
                n_surrogate_samples=parameters["n_surrogate_samples"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }