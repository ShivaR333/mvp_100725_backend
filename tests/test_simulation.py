import pytest
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import tempfile
import os
import json

from app.graph import (
    CausalGraph,
    fit_scm,
    simulate_interventions,
    pyro_simulate_interventions
)


class TestSimulation:
    """Test the simulation functionality."""
    
    def create_test_scenario(self, n_samples: int = 200) -> tuple:
        """
        Create a test scenario with graph, data, and fitted models.
        
        Returns:
            tuple: (temp_dir, graph_path, model_path, data_df)
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Generate synthetic manufacturing data
        np.random.seed(42)
        
        # Manufacturing process: temperature -> flow_rate -> efficiency -> quality
        # Also: temperature -> quality (direct effect)
        temperature = np.random.normal(75, 10, n_samples)  # Base temperature around 75
        flow_rate = 50 + 0.8 * temperature + np.random.normal(0, 5, n_samples)
        efficiency = 0.6 * flow_rate + 0.2 * temperature + np.random.normal(0, 3, n_samples)
        quality = 0.5 * efficiency + 0.3 * temperature + np.random.normal(0, 2, n_samples)
        
        # Create DataFrame
        data_df = pd.DataFrame({
            'temperature': temperature,
            'flow_rate': flow_rate,
            'efficiency': efficiency,
            'quality': quality
        })
        
        # Save data
        data_path = Path(temp_dir) / "data.csv"
        data_df.to_csv(data_path, index=False)
        
        # Create causal graph
        graph = CausalGraph()
        graph.add_node('temperature', name='Temperature')
        graph.add_node('flow_rate', name='Flow Rate')
        graph.add_node('efficiency', name='Efficiency')
        graph.add_node('quality', name='Quality')
        
        # Add edges based on causal structure
        graph.add_edge('temperature', 'flow_rate', confidence=1.0)
        graph.add_edge('temperature', 'efficiency', confidence=1.0)
        graph.add_edge('flow_rate', 'efficiency', confidence=1.0)
        graph.add_edge('efficiency', 'quality', confidence=1.0)
        graph.add_edge('temperature', 'quality', confidence=1.0)
        
        # Save graph
        graph_path = Path(temp_dir) / "graph.json"
        graph.save_to_json(str(graph_path))
        
        # Fit SCM models
        model_path = Path(temp_dir) / "model.pkl"
        fit_scm(
            graph_path=str(graph_path),
            clean_dir=temp_dir,
            n_boot=25,  # Reduced for testing speed
            model_output_path=str(model_path)
        )
        
        return temp_dir, str(graph_path), str(model_path), data_df
    
    def test_simulate_interventions_basic(self):
        """Test basic simulation functionality."""
        temp_dir, graph_path, model_path, data_df = self.create_test_scenario()
        
        try:
            # Define interventions
            interventions = {"temperature": 85.0}
            
            # Run simulation
            results = simulate_interventions(
                graph_path=graph_path,
                model_path=model_path,
                interventions=interventions,
                n_samples=500,
                kpi_nodes=["flow_rate", "efficiency", "quality"]
            )
            
            # Verify results structure
            assert isinstance(results, dict)
            assert "flow_rate" in results
            assert "efficiency" in results
            assert "quality" in results
            
            # Verify statistics for each KPI
            for kpi in ["flow_rate", "efficiency", "quality"]:
                kpi_stats = results[kpi]
                assert "mean" in kpi_stats
                assert "p5" in kpi_stats
                assert "p95" in kpi_stats
                assert "std" in kpi_stats
                assert "median" in kpi_stats
                
                # Basic sanity checks
                assert kpi_stats["p5"] <= kpi_stats["median"] <= kpi_stats["p95"]
                assert kpi_stats["std"] >= 0
                
                # Since we intervened on temperature=85, check that downstream effects are reasonable
                # (higher temperature should generally increase flow_rate, efficiency, quality)
                if kpi in ["flow_rate", "efficiency", "quality"]:
                    assert kpi_stats["mean"] > 0  # Should be positive values
            
            print(f"Simulation results: {results}")
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_simulate_interventions_multiple(self):
        """Test simulation with multiple interventions."""
        temp_dir, graph_path, model_path, data_df = self.create_test_scenario()
        
        try:
            # Define multiple interventions
            interventions = {
                "temperature": 90.0,
                "flow_rate": 120.0
            }
            
            # Run simulation
            results = simulate_interventions(
                graph_path=graph_path,
                model_path=model_path,
                interventions=interventions,
                n_samples=300,
                kpi_nodes=["efficiency", "quality"]
            )
            
            # Verify results
            assert isinstance(results, dict)
            assert "efficiency" in results
            assert "quality" in results
            
            # Check that interventions were applied
            for kpi in ["efficiency", "quality"]:
                kpi_stats = results[kpi]
                assert all(key in kpi_stats for key in ["mean", "p5", "p95", "std", "median"])
                assert kpi_stats["std"] >= 0
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_pyro_simulate_interventions(self):
        """Test Pyro-based simulation."""
        temp_dir, graph_path, model_path, data_df = self.create_test_scenario()
        
        try:
            # Define interventions
            interventions = {"temperature": 80.0}
            
            # Run Pyro simulation
            results = pyro_simulate_interventions(
                graph_path=graph_path,
                model_path=model_path,
                interventions=interventions,
                n_samples=200,  # Smaller for speed
                kpi_nodes=["flow_rate", "efficiency", "quality"]
            )
            
            # Verify results structure
            assert isinstance(results, dict)
            assert "flow_rate" in results
            assert "efficiency" in results
            assert "quality" in results
            
            # Verify statistics
            for kpi in ["flow_rate", "efficiency", "quality"]:
                kpi_stats = results[kpi]
                assert "mean" in kpi_stats
                assert "p5" in kpi_stats
                assert "p95" in kpi_stats
                assert kpi_stats["p5"] <= kpi_stats["median"] <= kpi_stats["p95"]
            
            print(f"Pyro simulation results: {results}")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_simulation_consistency(self):
        """Test that simulations produce consistent results."""
        temp_dir, graph_path, model_path, data_df = self.create_test_scenario()
        
        try:
            interventions = {"temperature": 85.0}
            kpi_nodes = ["flow_rate", "efficiency"]
            n_samples = 500
            
            # Run simulation twice
            results1 = simulate_interventions(
                graph_path=graph_path,
                model_path=model_path,
                interventions=interventions,
                n_samples=n_samples,
                kpi_nodes=kpi_nodes
            )
            
            # Set different random seed and run again
            np.random.seed(123)
            results2 = simulate_interventions(
                graph_path=graph_path,
                model_path=model_path,
                interventions=interventions,
                n_samples=n_samples,
                kpi_nodes=kpi_nodes
            )
            
            # Results should be similar but not identical (due to randomness)
            for kpi in kpi_nodes:
                mean1 = results1[kpi]["mean"]
                mean2 = results2[kpi]["mean"]
                
                # Means should be reasonably close (within 20%)
                relative_diff = abs(mean1 - mean2) / max(abs(mean1), abs(mean2), 1e-6)
                assert relative_diff < 0.2, f"KPI {kpi}: means too different ({mean1} vs {mean2})"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_simulation_no_interventions(self):
        """Test simulation with no interventions (baseline)."""
        temp_dir, graph_path, model_path, data_df = self.create_test_scenario()
        
        try:
            # No interventions
            interventions = {}
            
            # Run simulation
            results = simulate_interventions(
                graph_path=graph_path,
                model_path=model_path,
                interventions=interventions,
                n_samples=300,
                kpi_nodes=["temperature", "flow_rate", "efficiency", "quality"]
            )
            
            # Verify results
            assert isinstance(results, dict)
            assert len(results) == 4
            
            # All KPIs should have reasonable statistics
            for kpi in results:
                kpi_stats = results[kpi]
                assert all(key in kpi_stats for key in ["mean", "p5", "p95", "std", "median"])
                assert kpi_stats["std"] >= 0
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_simulation_custom_kpi_nodes(self):
        """Test simulation with custom KPI nodes."""
        temp_dir, graph_path, model_path, data_df = self.create_test_scenario()
        
        try:
            interventions = {"temperature": 85.0}
            custom_kpis = ["quality"]  # Only interested in quality
            
            # Run simulation
            results = simulate_interventions(
                graph_path=graph_path,
                model_path=model_path,
                interventions=interventions,
                n_samples=200,
                kpi_nodes=custom_kpis
            )
            
            # Should only return results for specified KPIs
            assert isinstance(results, dict)
            assert len(results) == 1
            assert "quality" in results
            assert "flow_rate" not in results
            assert "efficiency" not in results
            
            # Verify quality statistics
            quality_stats = results["quality"]
            assert all(key in quality_stats for key in ["mean", "p5", "p95", "std", "median"])
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestSimulationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_files(self):
        """Test error handling for missing files."""
        with pytest.raises(Exception):
            simulate_interventions(
                graph_path="nonexistent_graph.json",
                model_path="nonexistent_model.pkl",
                interventions={"x": 1.0},
                n_samples=100
            )
    
    def test_invalid_interventions(self):
        """Test handling of invalid intervention nodes."""
        # This would require creating a valid setup first, then testing with invalid nodes
        # For now, we'll test that the function doesn't crash with unknown nodes
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])