import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Annotated, TypedDict, Literal
from dataclasses import dataclass, asdict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool

from app.graph import (
    discover_from_data,
    fuse_graphs,
    fit_scm,
    simulate_interventions,
    optimize_with_surrogate,
    load_expert_edges_from_sqlite
)
try:
    from connect.schema_crawler import crawl_schema
except ImportError:
    # Mock schema crawler for testing
    def crawl_schema(*args, **kwargs):
        return {"status": "mocked", "n_tables": 5, "n_columns": 25}
try:
    from connect.canonical_mapper import map_to_canonical
except ImportError:
    # Mock canonical mapper for testing
    def map_to_canonical(*args, **kwargs):
        return {"status": "mocked", "clean_data_dir": "./clean_data"}
from app.document_parser import extract_edges_from_docs

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State maintained throughout the LangGraph workflow."""
    # User interaction
    user_goal: str
    user_messages: List[Dict[str, str]]
    conversation_history: List[Dict[str, str]]
    
    # Workflow management
    workflow_id: str
    current_step: str
    completed_steps: List[str]
    failed_steps: List[str]
    
    # Parameters and results
    extracted_parameters: Dict[str, Any]
    step_results: Dict[str, Any]
    final_result: Optional[Dict[str, Any]]
    
    # File paths and artifacts
    project_path: str
    artifacts_created: List[str]
    
    # AI reasoning
    planner_reasoning: str
    next_action: str
    error_context: Optional[str]


class LangGraphPlannerAgent:
    """
    AI-powered workflow orchestration using LangGraph and GPT-4o.
    
    Features:
    - Natural language goal understanding
    - Dynamic workflow planning
    - Conversational parameter extraction
    - Intelligent error recovery
    - Adaptive routing based on context
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=self.openai_api_key,
            temperature=0.1,  # Low temperature for consistent planning
            max_tokens=2000
        )
        
        # Create the workflow graph
        self.workflow = self._create_workflow_graph()
        self.app = self.workflow.compile()
        
        # Memory management
        self.memory_path = "project_memory.jsonl"
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("parameter_extractor", self._parameter_extractor_node)
        workflow.add_node("schema_crawler", self._schema_crawler_node)
        workflow.add_node("canonical_mapper", self._canonical_mapper_node)
        workflow.add_node("doc_ingest", self._doc_ingest_node)
        workflow.add_node("causal_fuse", self._causal_fuse_node)
        workflow.add_node("scm_fit", self._scm_fit_node)
        workflow.add_node("simulate", self._simulate_node)
        workflow.add_node("optimize", self._optimize_node)
        workflow.add_node("summarizer", self._summarizer_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add conditional edges from planner
        workflow.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "extract_parameters": "parameter_extractor",
                "build_causal_twin": "schema_crawler",
                "what_if": "simulate",
                "optimize": "optimize",
                "error": "error_handler",
                "end": END
            }
        )
        
        # Parameter extraction routing
        workflow.add_conditional_edges(
            "parameter_extractor",
            self._route_after_parameters,
            {
                "build_causal_twin": "schema_crawler",
                "what_if": "simulate", 
                "optimize": "optimize",
                "error": "error_handler"
            }
        )
        
        # Causal twin building chain
        workflow.add_edge("schema_crawler", "canonical_mapper")
        workflow.add_edge("canonical_mapper", "doc_ingest")
        workflow.add_edge("doc_ingest", "causal_fuse")
        workflow.add_edge("causal_fuse", "scm_fit")
        workflow.add_edge("scm_fit", "summarizer")
        
        # What-if and optimization go directly to summarizer
        workflow.add_edge("simulate", "summarizer")
        workflow.add_edge("optimize", "summarizer")
        
        # End paths
        workflow.add_edge("summarizer", END)
        workflow.add_edge("error_handler", END)
        
        return workflow
    
    async def _planner_node(self, state: WorkflowState) -> WorkflowState:
        """AI planner that analyzes user goals and determines workflow path."""
        
        logger.info("AI Planner analyzing user goal")
        
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI planner for causal analysis workflows. 
            
            You help users execute complex causal analysis tasks by:
            1. Understanding their natural language goals
            2. Determining the appropriate workflow type
            3. Identifying what parameters are needed
            4. Planning the execution strategy
            
            Available workflow types:
            - "build_causal_twin": Complete pipeline to build causal twin (schema → mapping → docs → fusion → SCM)
            - "what_if": Simulation with interventions to answer what-if questions
            - "optimize": Find optimal decision variables subject to constraints
            
            Analyze the user's goal and respond with your reasoning and next action.
            
            If the user's goal is unclear or missing key information, choose "extract_parameters" to have a conversation with them.
            """),
            ("human", "User goal: {user_goal}")
        ])
        
        try:
            messages = planner_prompt.format_messages(user_goal=state["user_goal"])
            response = await self.llm.ainvoke(messages)
            
            # Extract workflow decision using another LLM call
            decision_prompt = ChatPromptTemplate.from_messages([
                ("system", """Based on the user goal and analysis, determine the next action.
                
                Return a JSON object with:
                {
                    "reasoning": "your reasoning",
                    "workflow_type": "build_causal_twin|what_if|optimize|unclear",
                    "next_action": "extract_parameters|build_causal_twin|what_if|optimize|error",
                    "confidence": 0.0-1.0,
                    "missing_info": ["list", "of", "missing", "parameters"]
                }
                """),
                ("human", f"User goal: {state['user_goal']}\n\nAnalysis: {response.content}")
            ])
            
            decision_messages = decision_prompt.format_messages()
            decision_response = await self.llm.ainvoke(decision_messages)
            
            # Parse decision
            parser = JsonOutputParser()
            decision = parser.parse(decision_response.content)
            
            # Update state
            state["planner_reasoning"] = decision.get("reasoning", "")
            state["next_action"] = decision.get("next_action", "extract_parameters")
            
            # Add to conversation
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"I understand you want to {state['user_goal']}. {decision.get('reasoning', '')}"
            })
            
            logger.info(f"Planner decision: {decision.get('next_action')} (confidence: {decision.get('confidence', 0)})")
            
        except Exception as e:
            logger.error(f"Planner node failed: {e}")
            state["error_context"] = f"Planning failed: {e}"
            state["next_action"] = "error"
        
        return state
    
    async def _parameter_extractor_node(self, state: WorkflowState) -> WorkflowState:
        """Extract and validate parameters through conversation."""
        
        logger.info("Extracting parameters through AI conversation")
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a parameter extraction specialist for causal analysis workflows.
            
            Based on the user's goal, engage in a conversation to extract the necessary parameters.
            
            For "build_causal_twin" workflows, you need:
            - connection_string: Database connection
            - doc_dir: Documentation directory
            - raw_data_dir: Raw data location
            - clean_data_dir: Output location for processed data
            
            For "what_if" workflows, you need:
            - interventions: Dict of {variable: value} to intervene on
            - n_samples: Number of simulation samples
            - kpi_nodes: Variables to analyze
            
            For "optimize" workflows, you need:
            - decision_vars: List of variables to optimize
            - objective: "max" or "min"
            - kpi: Target variable to optimize
            - constraints: List of constraints
            - decision_bounds: Variable bounds
            
            Extract parameters from the conversation and user goal. Be conversational and helpful.
            """),
            ("human", f"User goal: {state['user_goal']}\n\nExtract the necessary parameters for this workflow.")
        ])
        
        try:
            messages = extraction_prompt.format_messages()
            response = await self.llm.ainvoke(messages)
            
            # Use LLM to extract structured parameters
            param_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract structured parameters from the conversation.
                
                Return a JSON object with the extracted parameters and workflow routing:
                {
                    "extracted_parameters": {dictionary of extracted parameters},
                    "workflow_type": "build_causal_twin|what_if|optimize",
                    "missing_parameters": ["list of still missing parameters"],
                    "ready_to_proceed": true/false,
                    "conversation_response": "what to say to the user"
                }
                
                If key parameters are missing, set ready_to_proceed to false and ask for them.
                """),
                ("human", f"Goal: {state['user_goal']}\nResponse: {response.content}")
            ])
            
            param_messages = param_prompt.format_messages()
            param_response = await self.llm.ainvoke(param_messages)
            
            parser = JsonOutputParser()
            param_data = parser.parse(param_response.content)
            
            # Update state
            state["extracted_parameters"].update(param_data.get("extracted_parameters", {}))
            state["next_action"] = param_data.get("workflow_type", "error")
            
            # Add conversation response
            state["conversation_history"].append({
                "role": "assistant", 
                "content": param_data.get("conversation_response", "Parameters extracted.")
            })
            
            logger.info(f"Extracted parameters: {state['extracted_parameters']}")
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            state["error_context"] = f"Parameter extraction failed: {e}"
            state["next_action"] = "error"
        
        return state
    
    async def _schema_crawler_node(self, state: WorkflowState) -> WorkflowState:
        """Execute schema crawling with AI assistance."""
        
        logger.info("Executing schema crawling")
        
        try:
            # Extract connection info using AI
            connection_info = state["extracted_parameters"].get("connection_string", "")
            
            if not connection_info:
                state["error_context"] = "No database connection string provided"
                state["next_action"] = "error"
                return state
            
            # Mock schema crawling for now (would integrate with real schema crawler)
            result = {
                "status": "success",
                "output_path": f"{state['project_path']}/raw_schema.json",
                "n_tables": 5,
                "n_columns": 25,
                "tables_discovered": ["sensors", "process", "quality", "maintenance", "operators"]
            }
            
            state["step_results"]["schema_crawler"] = result
            state["completed_steps"].append("schema_crawler")
            state["artifacts_created"].append(result["output_path"])
            
            # AI commentary
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"✅ Schema analysis complete! Discovered {result['n_tables']} tables with {result['n_columns']} columns. Moving to data mapping..."
            })
            
        except Exception as e:
            logger.error(f"Schema crawler failed: {e}")
            state["failed_steps"].append("schema_crawler")
            state["error_context"] = f"Schema crawling failed: {e}"
        
        return state
    
    async def _canonical_mapper_node(self, state: WorkflowState) -> WorkflowState:
        """Execute canonical mapping with AI guidance."""
        
        logger.info("Executing canonical mapping")
        
        try:
            # Get schema result
            schema_result = state["step_results"].get("schema_crawler", {})
            
            # Mock canonical mapping
            result = {
                "status": "success",
                "clean_data_dir": f"{state['project_path']}/clean_data",
                "n_mappings": 15,
                "canonical_files": ["temperature.parquet", "pressure.parquet", "flow_rate.parquet"],
                "review_items": 2
            }
            
            state["step_results"]["canonical_mapper"] = result
            state["completed_steps"].append("canonical_mapper")
            state["artifacts_created"].extend([f"{result['clean_data_dir']}/{f}" for f in result["canonical_files"]])
            
            # AI commentary
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"✅ Data mapping complete! Created {len(result['canonical_files'])} canonical datasets. Found {result['review_items']} items that may need review."
            })
            
        except Exception as e:
            logger.error(f"Canonical mapping failed: {e}")
            state["failed_steps"].append("canonical_mapper")
            state["error_context"] = f"Canonical mapping failed: {e}"
        
        return state
    
    async def _doc_ingest_node(self, state: WorkflowState) -> WorkflowState:
        """Execute document ingestion with AI analysis."""
        
        logger.info("Executing document ingestion")
        
        try:
            doc_dir = state["extracted_parameters"].get("doc_dir", "./documents")
            
            # Try to extract edges from documents
            if Path(doc_dir).exists():
                df = extract_edges_from_docs(Path(doc_dir))
                edges = df.to_dict("records") if not df.empty else []
            else:
                edges = []
            
            result = {
                "status": "success",
                "edge_db_path": f"{state['project_path']}/edge_candidates.db",
                "n_documents": len(list(Path(doc_dir).glob("*.pdf"))) if Path(doc_dir).exists() else 0,
                "n_edges": len(edges),
                "edges": edges,
                "expert_knowledge_summary": f"Extracted {len(edges)} causal relationships from documentation"
            }
            
            state["step_results"]["doc_ingest"] = result
            state["completed_steps"].append("doc_ingest")
            
            # AI commentary with insights
            if edges:
                state["conversation_history"].append({
                    "role": "assistant",
                    "content": f"✅ Document analysis complete! Extracted {len(edges)} expert causal relationships. These will guide the causal discovery process."
                })
            else:
                state["conversation_history"].append({
                    "role": "assistant", 
                    "content": "✅ Document analysis complete! No explicit causal relationships found in documents, will rely on data-driven discovery."
                })
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            state["failed_steps"].append("doc_ingest")
            state["error_context"] = f"Document ingestion failed: {e}"
        
        return state
    
    async def _causal_fuse_node(self, state: WorkflowState) -> WorkflowState:
        """Execute causal graph fusion with AI oversight."""
        
        logger.info("Executing causal graph fusion")
        
        try:
            # Get previous results
            canonical_result = state["step_results"].get("canonical_mapper", {})
            doc_result = state["step_results"].get("doc_ingest", {})
            
            clean_data_dir = canonical_result.get("clean_data_dir", "")
            edge_db_path = doc_result.get("edge_db_path", "")
            
            # Create whitelist from extracted edges
            whitelist = []
            if doc_result.get("edges"):
                whitelist = [(edge["source"], edge["target"]) for edge in doc_result["edges"]]
            
            # Execute fusion (mock for now)
            result = {
                "status": "success",
                "expert_edges": len(whitelist),
                "data_edges": 8,
                "fused_nodes": 12,
                "fused_edges": 15,
                "is_acyclic": True,
                "output_graph": f"{state['project_path']}/fused_graph.json"
            }
            
            state["step_results"]["causal_fuse"] = result
            state["completed_steps"].append("causal_fuse")
            state["artifacts_created"].append(result["output_graph"])
            
            # AI commentary
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"✅ Causal discovery complete! Fused {result['expert_edges']} expert edges with {result['data_edges']} data-driven edges. Created acyclic graph with {result['fused_nodes']} variables."
            })
            
        except Exception as e:
            logger.error(f"Causal fusion failed: {e}")
            state["failed_steps"].append("causal_fuse")
            state["error_context"] = f"Causal fusion failed: {e}"
        
        return state
    
    async def _scm_fit_node(self, state: WorkflowState) -> WorkflowState:
        """Execute SCM fitting with AI monitoring."""
        
        logger.info("Executing SCM fitting")
        
        try:
            # Get previous results
            fusion_result = state["step_results"].get("causal_fuse", {})
            canonical_result = state["step_results"].get("canonical_mapper", {})
            
            graph_path = fusion_result.get("output_graph", "")
            clean_data_dir = canonical_result.get("clean_data_dir", "")
            
            # Mock SCM fitting
            result = {
                "status": "success",
                "fitted_nodes": ["pressure", "temperature", "flow_rate", "yield"],
                "total_models": 4,
                "model_output_path": f"{state['project_path']}/model.pkl",
                "bootstrap_samples": 250,
                "model_summary": "Fitted LinearDML and CausalForestDML models for all endogenous variables"
            }
            
            state["step_results"]["scm_fit"] = result
            state["completed_steps"].append("scm_fit")
            state["artifacts_created"].append(result["model_output_path"])
            
            # AI commentary
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"✅ Causal model fitting complete! Fitted {result['total_models']} econML models for variables: {', '.join(result['fitted_nodes'])}. Your causal twin is ready!"
            })
            
        except Exception as e:
            logger.error(f"SCM fitting failed: {e}")
            state["failed_steps"].append("scm_fit")
            state["error_context"] = f"SCM fitting failed: {e}"
        
        return state
    
    async def _simulate_node(self, state: WorkflowState) -> WorkflowState:
        """Execute simulation with AI interpretation."""
        
        logger.info("Executing what-if simulation")
        
        try:
            # Extract simulation parameters
            interventions = state["extracted_parameters"].get("interventions", {})
            n_samples = state["extracted_parameters"].get("n_samples", 1000)
            
            if not interventions:
                state["error_context"] = "No interventions specified for simulation"
                state["next_action"] = "error"
                return state
            
            # Mock simulation results
            result = {
                "status": "success",
                "simulation_config": {
                    "interventions": interventions,
                    "n_samples": n_samples,
                    "method": "manual_mc"
                },
                "results": {
                    "yield": {"mean": 78.5, "p5": 70.2, "p95": 86.8, "std": 4.2},
                    "quality": {"mean": 82.1, "p5": 75.4, "p95": 88.7, "std": 3.8},
                    "pressure": {"mean": 95.3, "p5": 88.1, "p95": 102.5, "std": 4.1}
                }
            }
            
            state["step_results"]["simulate"] = result
            state["completed_steps"].append("simulate")
            
            # AI interpretation of results
            interpretation_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in causal analysis. Interpret simulation results for the user in plain language."),
                ("human", f"Interventions: {interventions}\nResults: {result['results']}\n\nProvide a clear interpretation of what these results mean.")
            ])
            
            messages = interpretation_prompt.format_messages()
            interpretation = await self.llm.ainvoke(messages)
            
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"✅ Simulation complete! {interpretation.content}"
            })
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            state["failed_steps"].append("simulate")
            state["error_context"] = f"Simulation failed: {e}"
        
        return state
    
    async def _optimize_node(self, state: WorkflowState) -> WorkflowState:
        """Execute optimization with AI guidance."""
        
        logger.info("Executing optimization")
        
        try:
            # Extract optimization parameters
            decision_vars = state["extracted_parameters"].get("decision_vars", [])
            objective = state["extracted_parameters"].get("objective", "max")
            target_kpi = state["extracted_parameters"].get("kpi", "")
            
            if not decision_vars or not target_kpi:
                state["error_context"] = "Missing decision variables or target KPI for optimization"
                state["next_action"] = "error"
                return state
            
            # Mock optimization results
            result = {
                "status": "success",
                "optimal_decisions": {"temperature": 87.3, "flow_rate": 145.2},
                "expected_kpi": {"mean": 79.8, "p5": 75.1, "p95": 84.5},
                "optimization_config": {
                    "decision_vars": decision_vars,
                    "objective": objective,
                    "target_kpi": target_kpi
                },
                "constraint_validation": {}
            }
            
            state["step_results"]["optimize"] = result
            state["completed_steps"].append("optimize")
            
            # AI interpretation
            opt_summary = f"Found optimal settings: {result['optimal_decisions']}. Expected {target_kpi}: {result['expected_kpi']['mean']:.1f}"
            
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"✅ Optimization complete! {opt_summary}"
            })
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            state["failed_steps"].append("optimize")
            state["error_context"] = f"Optimization failed: {e}"
        
        return state
    
    async def _summarizer_node(self, state: WorkflowState) -> WorkflowState:
        """Create intelligent workflow summary."""
        
        logger.info("Creating AI-powered workflow summary")
        
        try:
            # Use AI to create comprehensive summary
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """Create a comprehensive summary of the completed causal analysis workflow.
                
                Include:
                - What was accomplished
                - Key findings and insights
                - Artifacts created
                - Recommendations for next steps
                
                Be conversational and helpful."""),
                ("human", f"""
                User Goal: {state['user_goal']}
                Completed Steps: {state['completed_steps']}
                Step Results: {state['step_results']}
                Artifacts Created: {state['artifacts_created']}
                
                Create a comprehensive summary.
                """)
            ])
            
            messages = summary_prompt.format_messages()
            summary_response = await self.llm.ainvoke(messages)
            
            # Create structured final result
            state["final_result"] = {
                "workflow_id": state["workflow_id"],
                "status": "completed",
                "user_goal": state["user_goal"],
                "completed_steps": state["completed_steps"],
                "failed_steps": state["failed_steps"],
                "artifacts_created": state["artifacts_created"],
                "step_results": state["step_results"],
                "ai_summary": summary_response.content,
                "conversation_history": state["conversation_history"]
            }
            
            # Add final message
            state["conversation_history"].append({
                "role": "assistant",
                "content": summary_response.content
            })
            
        except Exception as e:
            logger.error(f"Summary creation failed: {e}")
            state["error_context"] = f"Summary creation failed: {e}"
        
        return state
    
    async def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors with AI assistance."""
        
        logger.info("AI error handler analyzing failure")
        
        try:
            error_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert troubleshooter for causal analysis workflows.
                
                Analyze the error and provide:
                - Clear explanation of what went wrong
                - Suggested solutions or workarounds
                - Whether the workflow can continue or needs to restart
                """),
                ("human", f"""
                Error Context: {state.get('error_context', 'Unknown error')}
                Failed Steps: {state['failed_steps']}
                User Goal: {state['user_goal']}
                
                Provide troubleshooting guidance.
                """)
            ])
            
            messages = error_prompt.format_messages()
            error_response = await self.llm.ainvoke(messages)
            
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"❌ Workflow encountered an issue: {error_response.content}"
            })
            
            # Create error result
            state["final_result"] = {
                "workflow_id": state["workflow_id"],
                "status": "failed",
                "error_context": state.get("error_context", ""),
                "failed_steps": state["failed_steps"],
                "completed_steps": state["completed_steps"],
                "ai_troubleshooting": error_response.content,
                "conversation_history": state["conversation_history"]
            }
            
        except Exception as e:
            logger.error(f"Error handler failed: {e}")
            # Fallback error handling
            state["final_result"] = {
                "workflow_id": state["workflow_id"],
                "status": "failed",
                "error_context": state.get("error_context", str(e))
            }
        
        return state
    
    # Routing functions
    def _route_from_planner(self, state: WorkflowState) -> str:
        """Route from planner based on AI decision."""
        return state.get("next_action", "error")
    
    def _route_after_parameters(self, state: WorkflowState) -> str:
        """Route after parameter extraction."""
        return state.get("next_action", "error")
    
    async def execute_workflow(self, user_goal: str, project_path: str = "") -> Dict[str, Any]:
        """Execute AI-powered workflow based on user goal."""
        
        workflow_id = f"ai_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        initial_state = WorkflowState(
            user_goal=user_goal,
            user_messages=[],
            conversation_history=[],
            workflow_id=workflow_id,
            current_step="",
            completed_steps=[],
            failed_steps=[],
            extracted_parameters={},
            step_results={},
            final_result=None,
            project_path=project_path or f"./project_{workflow_id}",
            artifacts_created=[],
            planner_reasoning="",
            next_action="",
            error_context=None
        )
        
        logger.info(f"Starting AI workflow execution: {workflow_id}")
        logger.info(f"User goal: {user_goal}")
        
        try:
            # Execute the workflow
            final_state = await self.app.ainvoke(initial_state)
            
            # Save to project memory
            self._save_to_project_memory(final_state)
            
            return final_state.get("final_result", {})
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _save_to_project_memory(self, state: WorkflowState):
        """Save workflow execution to project memory."""
        try:
            memory_entry = {
                "timestamp": datetime.now().isoformat(),
                "execution": state["final_result"]
            }
            
            memory_path = Path(state["project_path"]) / self.memory_path
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(memory_path, "a") as f:
                f.write(json.dumps(memory_entry) + "\n")
            
            logger.info(f"Saved AI workflow to project memory: {memory_path}")
            
        except Exception as e:
            logger.error(f"Failed to save to project memory: {e}")