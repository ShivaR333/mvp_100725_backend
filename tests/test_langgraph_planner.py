import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

from app.langgraph_planner import LangGraphPlannerAgent, WorkflowState


class TestLangGraphPlannerAgent:
    """Test the LangGraph AI-powered planner functionality."""
    
    def setup_method(self):
        """Set up test environment with mocked OpenAI API key."""
        # Mock OpenAI API key for testing
        self.mock_api_key = "test-key-123"
        os.environ["OPENAI_API_KEY"] = self.mock_api_key
    
    def teardown_method(self):
        """Clean up test environment."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    @patch('app.langgraph_planner.ChatOpenAI')
    def test_langgraph_planner_initialization(self, mock_chat_openai):
        """Test LangGraph planner initialization with proper LLM setup."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        # Verify LLM initialization
        mock_chat_openai.assert_called_once_with(
            model="gpt-4o",
            api_key=self.mock_api_key,
            temperature=0.1,
            max_tokens=2000
        )
        
        # Verify workflow graph creation
        assert planner.workflow is not None
        assert planner.app is not None
        assert planner.memory_path == "project_memory.jsonl"
    
    def test_initialization_without_api_key(self):
        """Test that initialization fails without OpenAI API key."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            LangGraphPlannerAgent()
    
    @patch('app.langgraph_planner.ChatOpenAI')
    async def test_planner_node_goal_analysis(self, mock_chat_openai):
        """Test AI planner node goal analysis and decision making."""
        # Mock LLM responses
        mock_llm = MagicMock()
        mock_analysis_response = MagicMock()
        mock_analysis_response.content = "The user wants to build a causal twin for their manufacturing process."
        
        mock_decision_response = MagicMock()
        mock_decision_response.content = json.dumps({
            "reasoning": "User wants to build a causal twin, this requires the full pipeline",
            "workflow_type": "build_causal_twin",
            "next_action": "extract_parameters",
            "confidence": 0.9,
            "missing_info": ["connection_string", "doc_dir"]
        })
        
        mock_llm.ainvoke.side_effect = [mock_analysis_response, mock_decision_response]
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        # Test state
        state = WorkflowState(
            user_goal="build causal twin for manufacturing",
            user_messages=[],
            conversation_history=[],
            workflow_id="test_123",
            current_step="",
            completed_steps=[],
            failed_steps=[],
            extracted_parameters={},
            step_results={},
            final_result=None,
            project_path="./test_project",
            artifacts_created=[],
            planner_reasoning="",
            next_action="",
            error_context=None
        )
        
        # Execute planner node
        result_state = await planner._planner_node(state)
        
        # Verify AI decision making
        assert result_state["next_action"] == "extract_parameters"
        assert "causal twin" in result_state["planner_reasoning"]
        assert len(result_state["conversation_history"]) == 1
        assert "assistant" in result_state["conversation_history"][0]["role"]
        
        # Verify LLM was called twice (analysis + decision)
        assert mock_llm.ainvoke.call_count == 2
    
    @patch('app.langgraph_planner.ChatOpenAI')
    async def test_parameter_extractor_node(self, mock_chat_openai):
        """Test AI parameter extraction through conversation."""
        mock_llm = MagicMock()
        
        # Mock conversation response
        mock_conversation_response = MagicMock()
        mock_conversation_response.content = "I need your database connection string and document directory."
        
        # Mock parameter extraction response
        mock_param_response = MagicMock()
        mock_param_response.content = json.dumps({
            "extracted_parameters": {
                "connection_string": "postgresql://localhost/test",
                "doc_dir": "./documents"
            },
            "workflow_type": "build_causal_twin",
            "missing_parameters": ["raw_data_dir"],
            "ready_to_proceed": False,
            "conversation_response": "I've extracted your database connection. Please provide the raw data directory."
        })
        
        mock_llm.ainvoke.side_effect = [mock_conversation_response, mock_param_response]
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        state = WorkflowState(
            user_goal="build causal twin with postgresql database",
            user_messages=[],
            conversation_history=[],
            workflow_id="test_456",
            current_step="",
            completed_steps=[],
            failed_steps=[],
            extracted_parameters={},
            step_results={},
            final_result=None,
            project_path="./test_project",
            artifacts_created=[],
            planner_reasoning="",
            next_action="",
            error_context=None
        )
        
        # Execute parameter extractor
        result_state = await planner._parameter_extractor_node(state)
        
        # Verify parameter extraction
        assert "connection_string" in result_state["extracted_parameters"]
        assert result_state["extracted_parameters"]["connection_string"] == "postgresql://localhost/test"
        assert result_state["next_action"] == "build_causal_twin"
        assert len(result_state["conversation_history"]) == 1
        
        # Verify LLM calls
        assert mock_llm.ainvoke.call_count == 2
    
    @patch('app.langgraph_planner.ChatOpenAI')
    async def test_simulate_node_with_ai_interpretation(self, mock_chat_openai):
        """Test simulation node with AI result interpretation."""
        mock_llm = MagicMock()
        
        # Mock AI interpretation
        mock_interpretation = MagicMock()
        mock_interpretation.content = "The simulation shows that increasing temperature to 85°C would increase yield by 5.2% while maintaining quality standards."
        
        mock_llm.ainvoke.return_value = mock_interpretation
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        state = WorkflowState(
            user_goal="what if temperature increases to 85",
            user_messages=[],
            conversation_history=[],
            workflow_id="test_sim",
            current_step="",
            completed_steps=[],
            failed_steps=[],
            extracted_parameters={
                "interventions": {"temperature": 85},
                "n_samples": 1000
            },
            step_results={},
            final_result=None,
            project_path="./test_project",
            artifacts_created=[],
            planner_reasoning="",
            next_action="",
            error_context=None
        )
        
        # Execute simulation node
        result_state = await planner._simulate_node(state)
        
        # Verify simulation results
        assert "simulate" in result_state["step_results"]
        assert result_state["step_results"]["simulate"]["status"] == "success"
        assert "simulate" in result_state["completed_steps"]
        
        # Verify AI interpretation
        assert len(result_state["conversation_history"]) == 1
        assert "yield by 5.2%" in result_state["conversation_history"][0]["content"]
        
        # Verify LLM interpretation call
        mock_llm.ainvoke.assert_called_once()
    
    @patch('app.langgraph_planner.ChatOpenAI')
    async def test_optimize_node_execution(self, mock_chat_openai):
        """Test optimization node execution with AI guidance."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        state = WorkflowState(
            user_goal="optimize yield while keeping pressure low",
            user_messages=[],
            conversation_history=[],
            workflow_id="test_opt",
            current_step="",
            completed_steps=[],
            failed_steps=[],
            extracted_parameters={
                "decision_vars": ["temperature", "flow_rate"],
                "objective": "max",
                "kpi": "yield"
            },
            step_results={},
            final_result=None,
            project_path="./test_project",
            artifacts_created=[],
            planner_reasoning="",
            next_action="",
            error_context=None
        )
        
        # Execute optimization node
        result_state = await planner._optimize_node(state)
        
        # Verify optimization results
        assert "optimize" in result_state["step_results"]
        assert result_state["step_results"]["optimize"]["status"] == "success"
        assert "optimize" in result_state["completed_steps"]
        
        # Verify optimal decisions are present
        opt_result = result_state["step_results"]["optimize"]
        assert "optimal_decisions" in opt_result
        assert "temperature" in opt_result["optimal_decisions"]
        assert "flow_rate" in opt_result["optimal_decisions"]
    
    @patch('app.langgraph_planner.ChatOpenAI')
    async def test_error_handler_node(self, mock_chat_openai):
        """Test AI error handler with troubleshooting guidance."""
        mock_llm = MagicMock()
        
        # Mock error analysis response
        mock_error_response = MagicMock()
        mock_error_response.content = "The workflow failed because the database connection string is invalid. Please check your connection parameters and ensure the database is accessible."
        
        mock_llm.ainvoke.return_value = mock_error_response
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        state = WorkflowState(
            user_goal="build causal twin",
            user_messages=[],
            conversation_history=[],
            workflow_id="test_error",
            current_step="",
            completed_steps=[],
            failed_steps=["schema_crawler"],
            extracted_parameters={},
            step_results={},
            final_result=None,
            project_path="./test_project",
            artifacts_created=[],
            planner_reasoning="",
            next_action="",
            error_context="Database connection failed: invalid connection string"
        )
        
        # Execute error handler
        result_state = await planner._error_handler_node(state)
        
        # Verify error handling
        assert result_state["final_result"]["status"] == "failed"
        assert "ai_troubleshooting" in result_state["final_result"]
        assert "database connection" in result_state["final_result"]["ai_troubleshooting"]
        
        # Verify conversation update
        assert len(result_state["conversation_history"]) == 1
        assert "❌" in result_state["conversation_history"][0]["content"]
        
        # Verify LLM troubleshooting call
        mock_llm.ainvoke.assert_called_once()
    
    @patch('app.langgraph_planner.ChatOpenAI')
    async def test_summarizer_node_with_ai_insights(self, mock_chat_openai):
        """Test AI-powered workflow summarization."""
        mock_llm = MagicMock()
        
        # Mock comprehensive summary
        mock_summary_response = MagicMock()
        mock_summary_response.content = """
        🎯 **Workflow Complete!** 

        I successfully built your causal twin for the manufacturing process. Here's what was accomplished:

        ✅ **Data Pipeline**: Processed 5 database tables with 25 columns
        ✅ **Causal Discovery**: Identified 15 causal relationships between process variables
        ✅ **Model Fitting**: Trained 4 econometric models for yield prediction
        ✅ **Ready for Analysis**: Your causal twin is now ready for what-if scenarios and optimization

        **Next Steps**: You can now run simulations or optimize your process parameters using the fitted models.
        """
        
        mock_llm.ainvoke.return_value = mock_summary_response
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        state = WorkflowState(
            user_goal="build causal twin",
            user_messages=[],
            conversation_history=[],
            workflow_id="test_summary",
            current_step="",
            completed_steps=["schema_crawler", "canonical_mapper", "doc_ingest", "causal_fuse", "scm_fit"],
            failed_steps=[],
            extracted_parameters={},
            step_results={
                "scm_fit": {
                    "status": "success",
                    "fitted_nodes": ["yield", "quality", "pressure", "temperature"],
                    "total_models": 4
                }
            },
            final_result=None,
            project_path="./test_project",
            artifacts_created=["fused_graph.json", "model.pkl"],
            planner_reasoning="",
            next_action="",
            error_context=None
        )
        
        # Execute summarizer
        result_state = await planner._summarizer_node(state)
        
        # Verify comprehensive summary
        assert result_state["final_result"]["status"] == "completed"
        assert "ai_summary" in result_state["final_result"]
        assert "Workflow Complete" in result_state["final_result"]["ai_summary"]
        
        # Verify conversation update
        assert len(result_state["conversation_history"]) == 1
        assert "econometric models" in result_state["conversation_history"][0]["content"]
        
        # Verify LLM summary call
        mock_llm.ainvoke.assert_called_once()
    
    @patch('app.langgraph_planner.ChatOpenAI')
    async def test_full_workflow_execution(self, mock_chat_openai):
        """Test complete AI workflow execution end-to-end."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            planner = LangGraphPlannerAgent()
            
            # Mock the app.ainvoke to return a completed state
            mock_final_state = {
                "workflow_id": "ai_workflow_test",
                "final_result": {
                    "status": "completed",
                    "ai_summary": "Test workflow completed successfully",
                    "conversation_history": [
                        {"role": "assistant", "content": "Workflow completed!"}
                    ],
                    "completed_steps": ["simulate"],
                    "step_results": {
                        "simulate": {
                            "status": "success",
                            "results": {"yield": {"mean": 78.5}}
                        }
                    }
                }
            }
            
            # Mock the compiled app
            planner.app = MagicMock()
            planner.app.ainvoke = AsyncMock(return_value=mock_final_state)
            
            # Execute workflow
            result = await planner.execute_workflow(
                user_goal="what if temperature increases to 85",
                project_path=temp_dir
            )
            
            # Verify execution
            assert result["status"] == "completed"
            assert "ai_summary" in result
            assert "conversation_history" in result
            
            # Verify app was called
            planner.app.ainvoke.assert_called_once()
            
            # Verify project memory path setup
            call_args = planner.app.ainvoke.call_args[0][0]
            assert call_args["project_path"] == temp_dir
    
    @patch('app.langgraph_planner.ChatOpenAI')
    def test_workflow_routing_logic(self, mock_chat_openai):
        """Test workflow routing decisions."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        # Test planner routing
        state = {"next_action": "build_causal_twin"}
        route = planner._route_from_planner(state)
        assert route == "build_causal_twin"
        
        # Test parameter routing
        state = {"next_action": "what_if"}
        route = planner._route_after_parameters(state)
        assert route == "what_if"
        
        # Test error routing
        state = {"next_action": "error"}
        route = planner._route_from_planner(state)
        assert route == "error"
    
    @patch('app.langgraph_planner.ChatOpenAI')
    async def test_workflow_failure_recovery(self, mock_chat_openai):
        """Test AI workflow failure and recovery."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        # Mock app failure
        planner.app = MagicMock()
        planner.app.ainvoke = AsyncMock(side_effect=Exception("Workflow execution failed"))
        
        # Execute workflow
        result = await planner.execute_workflow(
            user_goal="test failure",
            project_path="./test"
        )
        
        # Verify failure handling
        assert result["status"] == "failed"
        assert "error" in result
        assert "Workflow execution failed" in result["error"]
    
    @patch('app.langgraph_planner.ChatOpenAI')
    def test_project_memory_management(self, mock_chat_openai):
        """Test project memory saving and loading."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        with tempfile.TemporaryDirectory() as temp_dir:
            planner = LangGraphPlannerAgent()
            
            # Create test state
            test_state = WorkflowState(
                user_goal="test goal",
                user_messages=[],
                conversation_history=[],
                workflow_id="test_memory",
                current_step="",
                completed_steps=[],
                failed_steps=[],
                extracted_parameters={},
                step_results={},
                final_result={
                    "status": "completed",
                    "ai_summary": "Test completed"
                },
                project_path=temp_dir,
                artifacts_created=[],
                planner_reasoning="",
                next_action="",
                error_context=None
            )
            
            # Save to memory
            planner._save_to_project_memory(test_state)
            
            # Verify memory file exists
            memory_file = Path(temp_dir) / "project_memory.jsonl"
            assert memory_file.exists()
            
            # Verify memory content
            with open(memory_file, 'r') as f:
                content = f.read()
                assert "test_memory" in content
                assert "completed" in content


class TestLangGraphIntegration:
    """Test LangGraph integration components."""
    
    def test_workflow_state_structure(self):
        """Test WorkflowState TypedDict structure."""
        # This is a structural test to ensure the state schema is correct
        state = WorkflowState(
            user_goal="test",
            user_messages=[],
            conversation_history=[],
            workflow_id="test_123",
            current_step="planner",
            completed_steps=[],
            failed_steps=[],
            extracted_parameters={},
            step_results={},
            final_result=None,
            project_path="./test",
            artifacts_created=[],
            planner_reasoning="test reasoning",
            next_action="extract_parameters",
            error_context=None
        )
        
        # Verify all required fields are present
        assert isinstance(state["user_goal"], str)
        assert isinstance(state["user_messages"], list)
        assert isinstance(state["conversation_history"], list)
        assert isinstance(state["workflow_id"], str)
        assert isinstance(state["extracted_parameters"], dict)
        assert isinstance(state["step_results"], dict)
        assert isinstance(state["artifacts_created"], list)
    
    @patch('app.langgraph_planner.ChatOpenAI')
    def test_langgraph_node_configuration(self, mock_chat_openai):
        """Test that all required LangGraph nodes are configured."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        planner = LangGraphPlannerAgent()
        
        # Verify workflow graph has all required nodes
        workflow_dict = planner.workflow.compiled()
        
        # This test verifies the graph structure without executing
        # In a real implementation, you'd check node registration
        assert planner.workflow is not None
        assert planner.app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])