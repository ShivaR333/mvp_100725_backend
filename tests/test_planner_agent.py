import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.planner_agent import PlannerAgent, WorkflowType, WorkflowStep, WorkflowExecution


class TestPlannerAgent:
    """Test the PlannerAgent functionality."""
    
    def test_parse_user_goal(self):
        """Test goal parsing for different workflow types."""
        agent = PlannerAgent()
        
        # Test causal twin building goals
        causal_twin_goals = [
            "build causal twin",
            "create causal twin for manufacturing",
            "construct digital twin",
            "build a causal model"
        ]
        
        for goal in causal_twin_goals:
            workflow_type = agent.parse_user_goal(goal)
            assert workflow_type == WorkflowType.BUILD_CAUSAL_TWIN
        
        # Test what-if analysis goals
        what_if_goals = [
            "what if we increase temperature",
            "simulate intervention on pressure",
            "run scenario analysis",
            "what happens if flow rate changes"
        ]
        
        for goal in what_if_goals:
            workflow_type = agent.parse_user_goal(goal)
            assert workflow_type == WorkflowType.WHAT_IF
        
        # Test optimization goals
        optimize_goals = [
            "optimize yield",
            "maximize efficiency",
            "minimize cost",
            "find best parameters",
            "optimise production"
        ]
        
        for goal in optimize_goals:
            workflow_type = agent.parse_user_goal(goal)
            assert workflow_type == WorkflowType.OPTIMIZE
    
    def test_create_causal_twin_template(self):
        """Test causal twin workflow template creation."""
        agent = PlannerAgent()
        
        parameters = {
            "connection_string": "postgresql://localhost/test",
            "doc_dir": "./docs",
            "raw_data_dir": "./raw",
            "clean_data_dir": "./clean"
        }
        
        steps = agent.create_workflow_template(WorkflowType.BUILD_CAUSAL_TWIN, parameters)
        
        # Should have 5 steps
        assert len(steps) == 5
        
        # Check step sequence
        expected_tools = ["schema_crawler", "canonical_mapper", "doc_ingest", "causal_fuse", "scm_fit"]
        actual_tools = [step.tool_name for step in steps]
        assert actual_tools == expected_tools
        
        # Check step IDs are sequential
        expected_ids = ["1_schema_crawler", "2_canonical_mapper", "3_doc_ingest", "4_causal_fuse", "5_scm_fit"]
        actual_ids = [step.step_id for step in steps]
        assert actual_ids == expected_ids
        
        # Check parameters are passed correctly
        assert steps[0].parameters["connection_string"] == "postgresql://localhost/test"
        assert steps[1].parameters["raw_data_dir"] == "./raw"
        assert steps[2].parameters["doc_dir"] == "./docs"
    
    def test_create_what_if_template(self):
        """Test what-if workflow template creation."""
        agent = PlannerAgent()
        
        parameters = {
            "interventions": {"temperature": 85.0},
            "n_samples": 1000,
            "kpi_nodes": ["yield", "quality"]
        }
        
        steps = agent.create_workflow_template(WorkflowType.WHAT_IF, parameters)
        
        # Should have 1 step
        assert len(steps) == 1
        assert steps[0].tool_name == "simulate"
        assert steps[0].parameters["interventions"] == {"temperature": 85.0}
        assert steps[0].parameters["n_samples"] == 1000
    
    def test_create_optimize_template(self):
        """Test optimization workflow template creation."""
        agent = PlannerAgent()
        
        parameters = {
            "decision_vars": ["temperature", "flow_rate"],
            "objective": "max",
            "kpi": "yield",
            "constraints": [{"lhs": "pressure", "op": "<=", "rhs": 100}]
        }
        
        steps = agent.create_workflow_template(WorkflowType.OPTIMIZE, parameters)
        
        # Should have 1 step
        assert len(steps) == 1
        assert steps[0].tool_name == "optimize"
        assert steps[0].parameters["decision_vars"] == ["temperature", "flow_rate"]
        assert steps[0].parameters["objective"] == "max"
        assert steps[0].parameters["target_kpi"] == "yield"
    
    @pytest.mark.asyncio
    async def test_execute_what_if_workflow(self):
        """Test execution of what-if workflow."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "test_memory.jsonl"
            agent = PlannerAgent(str(memory_path))
            
            # Mock the simulate tool
            async def mock_simulate(parameters):
                return {
                    "status": "success",
                    "simulation_config": {
                        "interventions": parameters["interventions"],
                        "n_samples": parameters["n_samples"],
                        "method": "manual_mc",
                        "kpi_nodes": ["yield", "quality"]
                    },
                    "results": {
                        "yield": {"mean": 78.5, "p5": 70.2, "p95": 86.8},
                        "quality": {"mean": 82.1, "p5": 75.4, "p95": 88.7}
                    }
                }
            
            # Replace the simulate tool
            agent.tools["simulate"] = mock_simulate
            
            # Execute workflow
            execution = await agent.execute_workflow(
                user_goal="what if we increase temperature to 85",
                parameters={
                    "interventions": {"temperature": 85.0},
                    "n_samples": 500,
                    "project_path": temp_dir
                }
            )
            
            # Verify execution
            assert execution.status == "completed"
            assert execution.workflow_type == WorkflowType.WHAT_IF
            assert len(execution.steps) == 1
            assert execution.steps[0].status == "completed"
            assert execution.final_result is not None
            
            # Check final result structure
            final_result = execution.final_result
            assert "what_if_summary" in final_result
            assert final_result["what_if_summary"]["interventions_applied"] == {"temperature": 85.0}
            
            # Verify project memory was saved
            assert memory_path.exists()
            
            # Load and verify memory
            executions = agent.load_project_memory(temp_dir)
            assert len(executions) == 1
            assert executions[0]["execution"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_optimization_workflow(self):
        """Test execution of optimization workflow."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "test_memory.jsonl"
            agent = PlannerAgent(str(memory_path))
            
            # Mock the optimize tool
            async def mock_optimize(parameters):
                return {
                    "status": "success",
                    "optimal_decisions": {"temperature": 87.3, "flow_rate": 145.2},
                    "expected_kpi": {"mean": 79.8, "p5": 75.1, "p95": 84.5},
                    "optimization_config": {
                        "decision_vars": parameters["decision_vars"],
                        "objective": parameters["objective"],
                        "target_kpi": parameters["target_kpi"]
                    },
                    "constraint_validation": {}
                }
            
            # Replace the optimize tool
            agent.tools["optimize"] = mock_optimize
            
            # Execute workflow
            execution = await agent.execute_workflow(
                user_goal="optimize yield",
                parameters={
                    "decision_vars": ["temperature", "flow_rate"],
                    "objective": "max",
                    "kpi": "yield",
                    "project_path": temp_dir
                }
            )
            
            # Verify execution
            assert execution.status == "completed"
            assert execution.workflow_type == WorkflowType.OPTIMIZE
            assert len(execution.steps) == 1
            assert execution.steps[0].status == "completed"
            
            # Check optimization summary
            final_result = execution.final_result
            assert "optimization_summary" in final_result
            opt_summary = final_result["optimization_summary"]
            assert opt_summary["decision_variables"] == ["temperature", "flow_rate"]
            assert opt_summary["optimal_decisions"] == {"temperature": 87.3, "flow_rate": 145.2}
    
    @pytest.mark.asyncio
    async def test_workflow_failure_handling(self):
        """Test handling of workflow step failures."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = PlannerAgent()
            
            # Mock a failing tool
            async def mock_failing_simulate(parameters):
                raise Exception("Simulation failed due to missing files")
            
            agent.tools["simulate"] = mock_failing_simulate
            
            # Execute workflow that should fail
            execution = await agent.execute_workflow(
                user_goal="what if temperature increases",
                parameters={
                    "interventions": {"temperature": 85.0},
                    "project_path": temp_dir
                }
            )
            
            # Verify failure handling
            assert execution.status == "failed"
            assert len(execution.steps) == 1
            assert execution.steps[0].status == "failed"
            assert execution.steps[0].error == "Simulation failed due to missing files"
    
    def test_parameter_flow_between_steps(self):
        """Test parameter updates between workflow steps."""
        agent = PlannerAgent()
        
        # Create causal twin workflow steps
        parameters = {"clean_data_dir": "./clean", "edge_db": "./edges.db"}
        steps = agent.create_workflow_template(WorkflowType.BUILD_CAUSAL_TWIN, parameters)
        
        # Mock results from schema crawler
        schema_result = {"output_path": "./updated_schema.json"}
        agent._update_downstream_parameters(steps, 0, schema_result)
        
        # Check that canonical mapper got updated schema path
        canonical_step = steps[1]  # canonical_mapper is step 2
        assert canonical_step.parameters["raw_schema_path"] == "./updated_schema.json"
        
        # Mock results from doc ingest
        doc_result = {
            "edge_db_path": "./edges.db",
            "edges": [
                {"source": "temperature", "target": "pressure"},
                {"source": "pressure", "target": "yield"}
            ]
        }
        agent._update_downstream_parameters(steps, 2, doc_result)
        
        # Check that causal fusion got whitelist from doc ingest
        fusion_step = steps[3]  # causal_fuse is step 4
        expected_whitelist = [("temperature", "pressure"), ("pressure", "yield")]
        assert fusion_step.parameters["whitelist"] == expected_whitelist
    
    def test_project_memory_persistence(self):
        """Test project memory loading and saving."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "test_memory.jsonl"
            agent = PlannerAgent(str(memory_path))
            
            # Create mock execution
            execution = WorkflowExecution(
                workflow_id="test_123",
                workflow_type=WorkflowType.WHAT_IF,
                user_goal="test goal",
                steps=[],
                status="completed",
                project_path=temp_dir
            )
            
            # Save to memory
            agent._save_to_project_memory(execution)
            
            # Verify file was created
            assert memory_path.exists()
            
            # Load memory and verify
            executions = agent.load_project_memory(temp_dir)
            assert len(executions) == 1
            assert executions[0]["execution"]["workflow_id"] == "test_123"
            assert executions[0]["execution"]["workflow_type"] == "what_if"
    
    def test_workflow_summary_creation(self):
        """Test creation of workflow summaries."""
        agent = PlannerAgent()
        
        # Create mock execution with completed steps
        step1 = WorkflowStep(
            step_id="1_simulate",
            tool_name="simulate",
            description="Run simulation",
            parameters={},
            status="completed",
            result={
                "simulation_config": {"interventions": {"temp": 85}, "n_samples": 1000},
                "results": {"yield": {"mean": 78.5}}
            }
        )
        
        execution = WorkflowExecution(
            workflow_id="test_456",
            workflow_type=WorkflowType.WHAT_IF,
            user_goal="test simulation",
            steps=[step1],
            status="completed"
        )
        
        # Create summary
        summary = agent._create_workflow_summary(execution)
        
        # Verify summary structure
        assert summary["workflow_type"] == "what_if"
        assert summary["total_steps"] == 1
        assert summary["completed_steps"] == 1
        assert "what_if_summary" in summary
        assert summary["what_if_summary"]["interventions_applied"] == {"temp": 85}


class TestWorkflowTemplates:
    """Test workflow template functionality."""
    
    def test_workflow_step_creation(self):
        """Test WorkflowStep creation and attributes."""
        step = WorkflowStep(
            step_id="1_test",
            tool_name="test_tool",
            description="Test step",
            parameters={"param1": "value1"}
        )
        
        assert step.step_id == "1_test"
        assert step.tool_name == "test_tool"
        assert step.status == "pending"
        assert step.result is None
        assert step.error is None
    
    def test_workflow_execution_creation(self):
        """Test WorkflowExecution creation and attributes."""
        execution = WorkflowExecution(
            workflow_id="test_789",
            workflow_type=WorkflowType.BUILD_CAUSAL_TWIN,
            user_goal="test goal",
            steps=[]
        )
        
        assert execution.workflow_id == "test_789"
        assert execution.workflow_type == WorkflowType.BUILD_CAUSAL_TWIN
        assert execution.status == "pending"
        assert execution.final_result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])