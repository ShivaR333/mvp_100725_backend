from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import logging
import os
from typing import Optional, List, Tuple, Dict, Any
from app.document_parser import extract_edges_from_docs
from app.graph import (
    CausalGraph, 
    discover_from_data, 
    fuse_graphs, 
    load_expert_edges_from_sqlite,
    fit_scm,
    load_scm_models,
    predict_scm,
    simulate_interventions,
    pyro_simulate_interventions,
    optimize_with_surrogate
)
from app.planner_agent import PlannerAgent, WorkflowType
from app.langgraph_planner import LangGraphPlannerAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OptiFlux Backend", version="0.1.0")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PlannerAgents
planner_agent = PlannerAgent()

# Initialize LangGraph AI Planner (requires OpenAI API key)
try:
    ai_planner = LangGraphPlannerAgent()
    logger.info("LangGraph AI Planner initialized successfully")
except Exception as e:
    logger.warning(f"LangGraph AI Planner initialization failed: {e}")
    ai_planner = None

class DocumentPath(BaseModel):
    path: str


class GraphFusionRequest(BaseModel):
    clean_data_dir: str
    sqlite_db_path: str
    confidence_threshold: float = 0.6
    output_json_path: str = "fused_graph.json"
    whitelist: Optional[List[Tuple[str, str]]] = None
    blacklist: Optional[List[Tuple[str, str]]] = None
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None


class SCMFitRequest(BaseModel):
    graph_path: str
    clean_dir: str
    outcome_nodes: Optional[List[str]] = None
    n_boot: int = 250
    model_output_path: str = "model.pkl"


class SCMPredictRequest(BaseModel):
    model_path: str
    interventions: Optional[Dict[str, float]] = None


class SimulationRequest(BaseModel):
    interventions: Dict[str, float]
    n_samples: int = 1000
    graph_path: str = "fused_graph.json"
    model_path: str = "model.pkl"
    kpi_nodes: Optional[List[str]] = None
    use_pyro: bool = False


class ConstraintModel(BaseModel):
    lhs: str  # Left-hand side variable name
    op: str   # Operator: "<=", ">=", "=="
    rhs: float  # Right-hand side value


class OptimizationRequest(BaseModel):
    decision_vars: List[str]
    objective: str  # "max" or "min"
    kpi: str  # Target KPI to optimize
    constraints: Optional[List[ConstraintModel]] = None
    decision_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    graph_path: str = "fused_graph.json"
    model_path: str = "model.pkl"
    n_surrogate_samples: int = 200


class PlannerRequest(BaseModel):
    user_goal: str
    parameters: Dict[str, Any] = {}
    project_path: str = ""


class ProjectMemoryRequest(BaseModel):
    project_path: str = ""


class AIPlannerRequest(BaseModel):
    user_goal: str
    project_path: str = ""
    conversation_id: str = ""


class AIConversationRequest(BaseModel):
    conversation_id: str
    user_message: str
    project_path: str = ""

@app.get("/")
async def root():
    return {"message": "OptiFlux Backend API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process-documents/")
async def process_documents(doc_path: DocumentPath):
    """
    Processes documents in the specified directory to extract causal edges.
    """
    logger.info(f"Received request to process documents at: {doc_path.path}")
    
    directory = Path(doc_path.path)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path provided.")

    try:
        df = extract_edges_from_docs(directory)
        if df.empty:
            return {"message": "No causal edges found."}
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/fuse")
async def fuse_graph(request: GraphFusionRequest):
    """
    Fuse expert knowledge graph with data-driven causal discovery.
    
    This endpoint:
    1. Loads expert edges from SQLite database (edge_candidates table)
    2. Runs causal discovery on clean data using PC-Stable algorithm
    3. Fuses the two graphs using confidence combination
    4. Saves the result to Neo4j and JSON file
    """
    logger.info("Starting graph fusion process")
    
    try:
        # Validate input paths
        clean_data_dir = Path(request.clean_data_dir)
        if not clean_data_dir.is_dir():
            raise HTTPException(status_code=400, detail="Clean data directory not found")
        
        sqlite_db_path = Path(request.sqlite_db_path)
        if not sqlite_db_path.exists():
            raise HTTPException(status_code=400, detail="SQLite database not found")
        
        # Step 1: Load expert edges from SQLite
        logger.info("Loading expert edges from SQLite database")
        expert_graph = load_expert_edges_from_sqlite(
            str(sqlite_db_path), 
            request.confidence_threshold
        )
        
        if len(expert_graph.edges()) == 0:
            logger.warning("No expert edges found meeting confidence threshold")
        
        # Step 2: Discover causal structure from data
        logger.info("Running causal discovery on clean data")
        data_graph = discover_from_data(
            str(clean_data_dir),
            whitelist=request.whitelist,
            blacklist=request.blacklist
        )
        
        if len(data_graph.edges()) == 0:
            logger.warning("No causal edges discovered from data")
        
        # Step 3: Fuse the graphs
        logger.info("Fusing expert and data-driven graphs")
        fused_graph = fuse_graphs(expert_graph, data_graph)
        
        # Step 4: Save to Neo4j if configured
        if request.neo4j_uri and request.neo4j_user and request.neo4j_password:
            logger.info("Saving fused graph to Neo4j")
            fused_graph.configure_neo4j(
                request.neo4j_uri,
                request.neo4j_user,
                request.neo4j_password
            )
            fused_graph.save_to_neo4j()
        
        # Step 5: Save to JSON file
        output_path = Path(request.output_json_path)
        if not output_path.is_absolute():
            output_path = clean_data_dir / output_path
        
        logger.info(f"Saving fused graph to JSON: {output_path}")
        fused_graph.save_to_json(str(output_path))
        
        # Return summary
        result = {
            "status": "success",
            "expert_edges": len(expert_graph.edges()),
            "data_edges": len(data_graph.edges()),
            "fused_nodes": len(fused_graph.nodes()),
            "fused_edges": len(fused_graph.edges()),
            "is_acyclic": fused_graph.is_acyclic(),
            "output_files": {
                "json": str(output_path)
            }
        }
        
        if request.neo4j_uri:
            result["output_files"]["neo4j"] = request.neo4j_uri
        
        logger.info("Graph fusion completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in graph fusion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scm/fit")
async def fit_scm_endpoint(request: SCMFitRequest):
    """
    Fit a Structural Causal Model (SCM) using econML estimators.
    
    This endpoint:
    1. Loads causal graph from JSON file
    2. Processes nodes in topological order
    3. Fits LinearDML or CausalForestDML for each endogenous node
    4. Calculates bootstrap standard errors
    5. Serializes fitted models to pickle file
    """
    logger.info("Starting SCM fitting via API")
    
    try:
        # Validate input paths
        graph_path = Path(request.graph_path)
        if not graph_path.exists():
            raise HTTPException(status_code=400, detail="Graph file not found")
        
        clean_dir = Path(request.clean_dir)
        if not clean_dir.is_dir():
            raise HTTPException(status_code=400, detail="Clean data directory not found")
        
        # Fit SCM
        model_dict = fit_scm(
            graph_path=str(graph_path),
            clean_dir=str(clean_dir),
            outcome_nodes=request.outcome_nodes,
            n_boot=request.n_boot,
            model_output_path=request.model_output_path
        )
        
        # Return summary
        result = {
            "status": "success",
            "fitted_nodes": model_dict["metadata"]["fitted_nodes"],
            "topological_order": model_dict["metadata"]["topological_order"],
            "total_models": len(model_dict["models"]),
            "bootstrap_samples": request.n_boot,
            "model_output_path": request.model_output_path
        }
        
        # Add model details
        model_details = {}
        for node, model_info in model_dict["models"].items():
            model_details[node] = {
                "parents": model_info["parents"],
                "model_type": model_info["model_type"],
                "n_samples": model_info["n_samples"],
                "bootstrap_se": model_info["boot_se"].get("bootstrap_se", 0.0)
            }
        
        result["model_details"] = model_details
        
        logger.info("SCM fitting completed successfully via API")
        return result
        
    except Exception as e:
        logger.error(f"Error in SCM fitting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scm/predict")
async def predict_scm_endpoint(request: SCMPredictRequest):
    """
    Make predictions using fitted SCM models.
    
    This endpoint:
    1. Loads fitted SCM models from pickle file
    2. Processes nodes in topological order
    3. Applies interventions if specified
    4. Returns predicted values for all nodes
    """
    logger.info("Starting SCM prediction via API")
    
    try:
        # Validate model path
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=400, detail="Model file not found")
        
        # Load models
        models = load_scm_models(str(model_path))
        
        # Make predictions
        predictions = predict_scm(models, request.interventions)
        
        # Return results
        result = {
            "status": "success",
            "predictions": predictions,
            "interventions": request.interventions or {},
            "model_metadata": {
                "fitted_nodes": models["metadata"]["fitted_nodes"],
                "topological_order": models["metadata"]["topological_order"]
            }
        }
        
        logger.info("SCM prediction completed successfully via API")
        return result
        
    except Exception as e:
        logger.error(f"Error in SCM prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate")
async def simulate_endpoint(request: SimulationRequest):
    """
    Perform Monte Carlo simulation with interventions using do-calculus.
    
    This endpoint:
    1. Loads fused causal graph and fitted SCM models
    2. Applies interventions by replacing structural equations
    3. Performs forward-sampling Monte Carlo simulation
    4. Returns mean and 5-95th percentiles for each KPI
    
    Example request:
    {
        "interventions": {"temperature": 85, "flow_rate": 150},
        "n_samples": 1000,
        "kpi_nodes": ["efficiency", "quality", "cost"]
    }
    """
    logger.info("Starting simulation via API")
    
    try:
        # Validate file paths
        graph_path = Path(request.graph_path)
        if not graph_path.exists():
            raise HTTPException(status_code=400, detail=f"Graph file not found: {request.graph_path}")
        
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=400, detail=f"Model file not found: {request.model_path}")
        
        # Validate n_samples
        if request.n_samples <= 0 or request.n_samples > 10000:
            raise HTTPException(status_code=400, detail="n_samples must be between 1 and 10000")
        
        # Run simulation
        if request.use_pyro:
            logger.info("Using Pyro for simulation")
            results = pyro_simulate_interventions(
                graph_path=str(graph_path),
                model_path=str(model_path),
                interventions=request.interventions,
                n_samples=request.n_samples,
                kpi_nodes=request.kpi_nodes
            )
        else:
            logger.info("Using manual Monte Carlo simulation")
            results = simulate_interventions(
                graph_path=str(graph_path),
                model_path=str(model_path),
                interventions=request.interventions,
                n_samples=request.n_samples,
                kpi_nodes=request.kpi_nodes
            )
        
        # Format response
        response = {
            "status": "success",
            "simulation_config": {
                "interventions": request.interventions,
                "n_samples": request.n_samples,
                "method": "pyro" if request.use_pyro else "manual_mc",
                "kpi_nodes": list(results.keys())
            },
            "results": results
        }
        
        logger.info(f"Simulation completed successfully for {len(results)} KPIs")
        return response
        
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_endpoint(request: OptimizationRequest):
    """
    Optimize decision variables to maximize/minimize a target KPI subject to constraints.
    
    This endpoint:
    1. Trains quadratic surrogate models using Latin Hypercube Sampling
    2. Formulates a convex quadratic program using CVXPY
    3. Solves the optimization problem
    4. Validates results using the full simulator
    
    Example request:
    {
        "decision_vars": ["temperature", "flow_rate"],
        "objective": "max",
        "kpi": "yield",
        "constraints": [
            {"lhs": "pressure", "op": "<=", "rhs": 100},
            {"lhs": "co2_intensity", "op": "<=", "rhs": 0.5}
        ],
        "decision_bounds": {
            "temperature": [60, 100],
            "flow_rate": [50, 200]
        }
    }
    """
    logger.info("Starting optimization via API")
    
    try:
        # Validate file paths
        graph_path = Path(request.graph_path)
        if not graph_path.exists():
            raise HTTPException(status_code=400, detail=f"Graph file not found: {request.graph_path}")
        
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=400, detail=f"Model file not found: {request.model_path}")
        
        # Validate objective
        if request.objective not in ["max", "min"]:
            raise HTTPException(status_code=400, detail="objective must be 'max' or 'min'")
        
        # Validate decision variables
        if not request.decision_vars:
            raise HTTPException(status_code=400, detail="decision_vars cannot be empty")
        
        # Convert constraints to dict format
        constraints_list = None
        if request.constraints:
            constraints_list = []
            for constraint in request.constraints:
                # Validate operator
                if constraint.op not in ["<=", ">=", "=="]:
                    raise HTTPException(status_code=400, detail=f"Invalid constraint operator: {constraint.op}")
                
                constraints_list.append({
                    "lhs": constraint.lhs,
                    "op": constraint.op,
                    "rhs": constraint.rhs
                })
        
        # Validate surrogate samples
        if request.n_surrogate_samples <= 0 or request.n_surrogate_samples > 1000:
            raise HTTPException(status_code=400, detail="n_surrogate_samples must be between 1 and 1000")
        
        # Run optimization
        result = optimize_with_surrogate(
            graph_path=str(graph_path),
            model_path=str(model_path),
            decision_vars=request.decision_vars,
            objective=request.objective,
            target_kpi=request.kpi,
            constraints=constraints_list,
            decision_bounds=request.decision_bounds,
            n_surrogate_samples=request.n_surrogate_samples
        )
        
        # Format response
        if result["status"] == "success":
            response = {
                "status": "success",
                "optimal_decisions": result["optimal_decisions"],
                "expected_kpi": result["expected_kpi"],
                "optimization_config": {
                    "decision_vars": request.decision_vars,
                    "objective": request.objective,
                    "target_kpi": request.kpi,
                    "n_constraints": len(request.constraints) if request.constraints else 0,
                    "n_surrogate_samples": request.n_surrogate_samples
                },
                "optimization_info": result["optimization_info"],
                "constraint_validation": result["constraint_validation"]
            }
            
            logger.info(f"Optimization completed successfully. Optimal decisions: {result['optimal_decisions']}")
            return response
            
        else:
            # Optimization failed
            error_response = {
                "status": result["status"],
                "error": result.get("error", "Unknown optimization error"),
                "optimization_config": {
                    "decision_vars": request.decision_vars,
                    "objective": request.objective,
                    "target_kpi": request.kpi
                }
            }
            
            if "cvxpy_status" in result:
                error_response["cvxpy_status"] = result["cvxpy_status"]
            
            return error_response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/planner/execute")
async def execute_workflow(request: PlannerRequest):
    """
    Execute a multi-step causal analysis workflow based on user goal.
    
    Supports three main workflow types:
    1. "build causal twin" - Complete pipeline from schema to fitted SCM
    2. "what if" - Simulation with interventions  
    3. "optimize" - Find optimal decision variables
    
    Example requests:
    
    Build Causal Twin:
    {
        "user_goal": "build causal twin for manufacturing process",
        "parameters": {
            "connection_string": "postgresql://user:pass@localhost/db",
            "doc_dir": "./documents",
            "raw_data_dir": "./raw_data",
            "clean_data_dir": "./clean_data"
        }
    }
    
    What-if Analysis:
    {
        "user_goal": "what if we increase temperature to 85",
        "parameters": {
            "interventions": {"temperature": 85},
            "n_samples": 1000
        }
    }
    
    Optimization:
    {
        "user_goal": "optimize yield while keeping pressure low",
        "parameters": {
            "decision_vars": ["temperature", "flow_rate"],
            "objective": "max",
            "kpi": "yield",
            "constraints": [
                {"lhs": "pressure", "op": "<=", "rhs": 100}
            ]
        }
    }
    """
    logger.info("Starting workflow execution via PlannerAgent")
    
    try:
        # Execute workflow
        execution = await planner_agent.execute_workflow(
            user_goal=request.user_goal,
            parameters={**request.parameters, "project_path": request.project_path}
        )
        
        # Convert execution to response format
        response = {
            "status": execution.status,
            "workflow_id": execution.workflow_id,
            "workflow_type": execution.workflow_type.value,
            "user_goal": execution.user_goal,
            "execution_summary": {
                "total_steps": len(execution.steps),
                "completed_steps": sum(1 for step in execution.steps if step.status == "completed"),
                "failed_steps": sum(1 for step in execution.steps if step.status == "failed"),
                "start_time": execution.start_time,
                "end_time": execution.end_time
            },
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "status": step.status,
                    "error": step.error
                }
                for step in execution.steps
            ]
        }
        
        # Add final result if workflow completed successfully
        if execution.status == "completed" and execution.final_result:
            response["final_result"] = execution.final_result
        
        logger.info(f"Workflow execution completed: {execution.status}")
        return response
        
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/planner/memory")
async def get_project_memory(request: ProjectMemoryRequest = None):
    """
    Retrieve workflow execution history from project memory.
    
    Returns all past workflow executions stored in project_memory.jsonl
    """
    try:
        project_path = request.project_path if request else ""
        
        executions = planner_agent.load_project_memory(project_path)
        
        # Format response
        response = {
            "status": "success",
            "total_executions": len(executions),
            "executions": []
        }
        
        # Extract summary information from each execution
        for entry in executions:
            execution_data = entry.get("execution", {})
            
            summary = {
                "workflow_id": execution_data.get("workflow_id", ""),
                "workflow_type": execution_data.get("workflow_type", ""),
                "user_goal": execution_data.get("user_goal", ""),
                "status": execution_data.get("status", ""),
                "timestamp": entry.get("timestamp", ""),
                "total_steps": len(execution_data.get("steps", [])),
                "completed_steps": sum(
                    1 for step in execution_data.get("steps", []) 
                    if step.get("status") == "completed"
                )
            }
            
            # Add final result summary if available
            if execution_data.get("final_result"):
                final_result = execution_data["final_result"]
                summary["final_result_summary"] = {
                    "workflow_type": final_result.get("workflow_type", ""),
                    "status": final_result.get("status", "")
                }
                
                # Add workflow-specific summaries
                if "causal_twin_summary" in final_result:
                    summary["final_result_summary"]["causal_twin"] = final_result["causal_twin_summary"]
                elif "what_if_summary" in final_result:
                    summary["final_result_summary"]["what_if"] = final_result["what_if_summary"]
                elif "optimization_summary" in final_result:
                    summary["final_result_summary"]["optimization"] = final_result["optimization_summary"]
            
            response["executions"].append(summary)
        
        # Sort by timestamp (most recent first)
        response["executions"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        logger.info(f"Retrieved {len(executions)} workflow executions from project memory")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving project memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/planner/parse-goal")
async def parse_user_goal(request: PlannerRequest):
    """
    Parse user goal to determine workflow type and suggest parameters.
    
    This endpoint helps users understand what workflow will be executed
    for their goal and what parameters are needed.
    """
    try:
        # Parse workflow type
        workflow_type = planner_agent.parse_user_goal(request.user_goal)
        
        # Create template parameters
        template_steps = planner_agent.create_workflow_template(workflow_type, request.parameters)
        
        # Suggest missing parameters
        suggested_parameters = {}
        
        if workflow_type == WorkflowType.BUILD_CAUSAL_TWIN:
            suggested_parameters = {
                "connection_string": "Database connection string (required)",
                "doc_dir": "Directory containing documentation (./documents)",
                "raw_data_dir": "Directory with raw data files (./raw_data)",
                "clean_data_dir": "Output directory for clean data (./clean_data)",
                "canonical_dict": "Path to canonical dictionary (canonical_dict.json)",
                "confidence_threshold": "Minimum confidence for expert edges (0.6)",
                "n_boot": "Bootstrap samples for SCM fitting (250)"
            }
        elif workflow_type == WorkflowType.WHAT_IF:
            suggested_parameters = {
                "interventions": "Dictionary of interventions {variable: value} (required)",
                "n_samples": "Number of simulation samples (1000)",
                "graph_path": "Path to causal graph (fused_graph.json)",
                "model_path": "Path to fitted SCM models (model.pkl)",
                "kpi_nodes": "List of KPI nodes to analyze (null = all)"
            }
        elif workflow_type == WorkflowType.OPTIMIZE:
            suggested_parameters = {
                "decision_vars": "List of decision variable names (required)",
                "objective": "Optimization objective: max or min (required)",
                "kpi": "Target KPI to optimize (required)",
                "constraints": "List of constraints [{'lhs': 'var', 'op': '<=', 'rhs': value}]",
                "decision_bounds": "Variable bounds {'var': [min, max]}",
                "n_surrogate_samples": "Samples for surrogate training (200)"
            }
        
        response = {
            "status": "success",
            "parsed_goal": {
                "user_goal": request.user_goal,
                "workflow_type": workflow_type.value,
                "total_steps": len(template_steps),
                "step_descriptions": [step.description for step in template_steps]
            },
            "suggested_parameters": suggested_parameters,
            "provided_parameters": request.parameters
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error parsing user goal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai-planner/execute")
async def execute_ai_workflow(request: AIPlannerRequest):
    """
    Execute AI-powered workflow using LangGraph and GPT-4o.
    
    This endpoint provides intelligent workflow orchestration with:
    - Natural language goal understanding
    - Dynamic workflow planning
    - Conversational parameter extraction
    - Intelligent error recovery
    - AI-powered insights and recommendations
    
    Example requests:
    
    Build Causal Twin:
    {
        "user_goal": "I want to build a causal twin for our manufacturing process. We have sensor data and process documentation.",
        "project_path": "./manufacturing_project"
    }
    
    What-if Analysis:
    {
        "user_goal": "What would happen if we increase the temperature to 85 degrees and flow rate to 150?",
        "project_path": "./analysis_project"
    }
    
    Optimization:
    {
        "user_goal": "Find the best temperature and flow rate settings to maximize yield while keeping pressure below 100.",
        "project_path": "./optimization_project"
    }
    """
    if not ai_planner:
        raise HTTPException(
            status_code=503, 
            detail="AI Planner not available. Please check OpenAI API key configuration."
        )
    
    logger.info("Starting AI-powered workflow execution")
    
    try:
        # Execute AI workflow
        result = await ai_planner.execute_workflow(
            user_goal=request.user_goal,
            project_path=request.project_path
        )
        
        # Format response
        response = {
            "status": result.get("status", "unknown"),
            "workflow_id": result.get("workflow_id", ""),
            "ai_summary": result.get("ai_summary", ""),
            "conversation_history": result.get("conversation_history", []),
            "completed_steps": result.get("completed_steps", []),
            "failed_steps": result.get("failed_steps", []),
            "artifacts_created": result.get("artifacts_created", []),
            "step_results": result.get("step_results", {})
        }
        
        # Add workflow-specific results
        if "step_results" in result:
            # Extract key insights from step results
            if "simulate" in result["step_results"]:
                response["simulation_results"] = result["step_results"]["simulate"].get("results", {})
            
            if "optimize" in result["step_results"]:
                response["optimization_results"] = {
                    "optimal_decisions": result["step_results"]["optimize"].get("optimal_decisions", {}),
                    "expected_kpi": result["step_results"]["optimize"].get("expected_kpi", {})
                }
            
            if "scm_fit" in result["step_results"]:
                response["causal_twin_info"] = {
                    "fitted_models": result["step_results"]["scm_fit"].get("fitted_nodes", []),
                    "model_count": result["step_results"]["scm_fit"].get("total_models", 0)
                }
        
        logger.info(f"AI workflow completed: {result.get('status')}")
        return response
        
    except Exception as e:
        logger.error(f"Error in AI workflow execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai-planner/status")
async def get_ai_planner_status():
    """
    Get the status of the AI planner service.
    
    Returns information about whether the AI planner is available
    and properly configured.
    """
    try:
        status = {
            "ai_planner_available": ai_planner is not None,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "model": "gpt-4o" if ai_planner else None,
            "capabilities": [
                "Natural language goal understanding",
                "Dynamic workflow planning", 
                "Conversational parameter extraction",
                "Intelligent error recovery",
                "AI-powered insights"
            ] if ai_planner else []
        }
        
        if ai_planner:
            status["message"] = "AI Planner is ready for intelligent workflow orchestration"
        else:
            status["message"] = "AI Planner not available. Please configure OpenAI API key."
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking AI planner status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai-planner/chat")
async def chat_with_ai_planner(request: AIConversationRequest):
    """
    Have a conversation with the AI planner to refine goals and parameters.
    
    This endpoint allows for interactive conversation with the AI planner
    to clarify requirements, extract parameters, and plan workflows.
    
    Example:
    {
        "conversation_id": "conv_123",
        "user_message": "I want to understand how temperature affects our process",
        "project_path": "./analysis"
    }
    """
    if not ai_planner:
        raise HTTPException(
            status_code=503,
            detail="AI Planner not available. Please check OpenAI API key configuration."
        )
    
    logger.info(f"AI chat conversation: {request.conversation_id}")
    
    try:
        # For now, we'll use the main workflow execution with conversation
        # In a full implementation, you'd maintain conversation state
        
        conversational_goal = f"Continue conversation: {request.user_message}"
        
        result = await ai_planner.execute_workflow(
            user_goal=conversational_goal,
            project_path=request.project_path
        )
        
        # Extract conversation response
        conversation_history = result.get("conversation_history", [])
        last_response = ""
        
        if conversation_history:
            # Get the last AI response
            for msg in reversed(conversation_history):
                if msg.get("role") == "assistant":
                    last_response = msg.get("content", "")
                    break
        
        response = {
            "conversation_id": request.conversation_id,
            "ai_response": last_response,
            "conversation_history": conversation_history,
            "suggested_actions": [
                "Continue conversation",
                "Execute workflow",
                "Modify parameters"
            ]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in AI chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai-planner/examples")
async def get_ai_planner_examples():
    """
    Get example user goals and use cases for the AI planner.
    
    This helps users understand how to interact with the AI planner
    and what kinds of goals it can handle.
    """
    examples = {
        "causal_twin_building": [
            "Build a causal twin for our manufacturing process using sensor data and documentation",
            "Create a digital twin that can predict how process changes affect quality",
            "I want to understand the causal relationships in our production data",
            "Help me build a model that shows how different variables affect our KPIs"
        ],
        "what_if_analysis": [
            "What would happen if we increase temperature to 85 degrees?",
            "Simulate the effect of changing flow rate and pressure together",
            "Show me how yield would change if we modify these process parameters",
            "What if we run the process at higher temperature but lower pressure?"
        ],
        "optimization": [
            "Find the best settings to maximize yield while keeping costs low",
            "Optimize our process parameters to minimize energy consumption",
            "What's the optimal temperature and pressure for maximum efficiency?",
            "Help me find settings that maximize quality while staying within safety constraints"
        ],
        "conversational": [
            "I'm not sure what analysis I need, can you help me figure it out?",
            "What insights can you provide about my manufacturing data?",
            "I want to improve my process but don't know where to start",
            "Can you analyze my data and suggest what to focus on?"
        ]
    }
    
    return {
        "status": "success",
        "examples": examples,
        "tips": [
            "Be specific about your goals and constraints",
            "Mention what data and documentation you have available",
            "Ask follow-up questions to refine your analysis",
            "The AI can help you discover what's possible with your data"
        ]
    }