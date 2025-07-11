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
    QuadraticSurrogate,
    optimize_with_surrogate
)


class TestQuadraticSurrogate:
    """Test the QuadraticSurrogate class."""
    
    def create_test_scenario(self, n_samples: int = 100) -> tuple:
        """Create a test scenario for optimization testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Generate synthetic process data
        np.random.seed(42)
        
        # Process: temperature -> pressure -> yield
        #         flow_rate -> pressure -> yield  
        #         temperature -> yield (direct effect)
        temperature = np.random.uniform(60, 100, n_samples)
        flow_rate = np.random.uniform(50, 200, n_samples)
        
        # Pressure depends on both temperature and flow_rate
        pressure = 0.5 * temperature + 0.3 * flow_rate + np.random.normal(0, 5, n_samples)
        
        # Yield depends on temperature, pressure (quadratic effect)
        yield_base = 20 + 0.8 * temperature + 0.4 * pressure - 0.001 * pressure**2
        yield_values = yield_base + np.random.normal(0, 2, n_samples)
        
        # CO2 intensity (environmental constraint)
        co2_intensity = 0.01 * temperature + 0.005 * flow_rate + np.random.normal(0, 0.1, n_samples)
        
        # Create DataFrame
        data_df = pd.DataFrame({
            'temperature': temperature,
            'flow_rate': flow_rate,
            'pressure': pressure,
            'yield': yield_values,
            'co2_intensity': co2_intensity
        })
        
        # Save data
        data_path = Path(temp_dir) / "data.csv"
        data_df.to_csv(data_path, index=False)
        
        # Create causal graph
        graph = CausalGraph()
        for var in ['temperature', 'flow_rate', 'pressure', 'yield', 'co2_intensity']:
            graph.add_node(var, name=var.replace('_', ' ').title())
        
        # Add causal edges
        graph.add_edge('temperature', 'pressure', confidence=1.0)
        graph.add_edge('flow_rate', 'pressure', confidence=1.0)
        graph.add_edge('pressure', 'yield', confidence=1.0)
        graph.add_edge('temperature', 'yield', confidence=1.0)
        graph.add_edge('temperature', 'co2_intensity', confidence=1.0)
        graph.add_edge('flow_rate', 'co2_intensity', confidence=1.0)
        
        # Save graph
        graph_path = Path(temp_dir) / "graph.json"
        graph.save_to_json(str(graph_path))
        
        # Fit SCM models
        model_path = Path(temp_dir) / "model.pkl"
        fit_scm(
            graph_path=str(graph_path),
            clean_dir=temp_dir,
            n_boot=10,  # Reduced for testing speed
            model_output_path=str(model_path)
        )
        
        return temp_dir, str(graph_path), str(model_path), data_df
    
    def test_surrogate_training(self):
        """Test surrogate model training."""
        temp_dir, graph_path, model_path, data_df = self.create_test_scenario()
        
        try:
            # Define decision variables and KPIs
            decision_vars = ["temperature", "flow_rate"]
            target_kpis = ["yield", "pressure", "co2_intensity"]
            
            # Create surrogate
            surrogate = QuadraticSurrogate(decision_vars, target_kpis)
            
            # Train surrogate
            decision_bounds = {
                "temperature": (60.0, 100.0),
                "flow_rate": (50.0, 200.0)
            }
            
            r2_scores = surrogate.train(
                graph_path=graph_path,
                model_path=model_path,
                n_samples=50,  # Small for testing
                decision_bounds=decision_bounds
            )
            
            # Verify training results
            assert isinstance(r2_scores, dict)
            for kpi in target_kpis:
                assert kpi in r2_scores
                assert isinstance(r2_scores[kpi], float)
                # R² should be reasonable (though may be low for small sample)
                assert -1.0 <= r2_scores[kpi] <= 1.0
            
            # Test prediction
            test_X = np.array([[75.0, 125.0], [85.0, 150.0]])
            predictions = surrogate.predict(test_X)
            
            assert isinstance(predictions, dict)
            for kpi in target_kpis:
                assert kpi in predictions
                assert len(predictions[kpi]) == 2  # Two test samples
                assert all(isinstance(p, (int, float)) for p in predictions[kpi])
            
            # Test coefficient extraction
            for kpi in target_kpis:
                linear_coeffs, Q, intercept = surrogate.get_coefficients(kpi)
                
                assert len(linear_coeffs) == 2  # Two decision variables
                assert Q.shape == (2, 2)  # 2x2 quadratic matrix
                assert isinstance(intercept, (int, float))
                
                # Q should be symmetric
                np.testing.assert_allclose(Q, Q.T, atol=1e-10)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_surrogate_prediction_consistency(self):
        """Test that surrogate predictions are consistent."""
        temp_dir, graph_path, model_path, data_df = self.create_test_scenario()
        
        try:
            decision_vars = ["temperature", "flow_rate"]
            target_kpis = ["yield"]
            
            surrogate = QuadraticSurrogate(decision_vars, target_kpis)
            surrogate.train(
                graph_path=graph_path,
                model_path=model_path,
                n_samples=30,
                decision_bounds={"temperature": (60.0, 100.0), "flow_rate": (50.0, 200.0)}
            )
            
            # Test same input gives same output
            test_X = np.array([[80.0, 120.0]])
            pred1 = surrogate.predict(test_X)
            pred2 = surrogate.predict(test_X)
            
            for kpi in target_kpis:
                np.testing.assert_array_equal(pred1[kpi], pred2[kpi])
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestOptimization:
    """Test the optimization functionality."""
    
    def create_optimization_scenario(self) -> tuple:
        """Create a scenario for testing optimization."""
        temp_dir = tempfile.mkdtemp()
        
        # Simple 2D optimization problem: maximize yield subject to constraints
        np.random.seed(42)
        n_samples = 80
        
        # Decision variables
        temperature = np.random.uniform(60, 100, n_samples)
        flow_rate = np.random.uniform(50, 200, n_samples)
        
        # Simple quadratic yield function with known optimum
        # yield = 100 + 2*temp + 1*flow - 0.01*temp^2 - 0.005*flow^2 + noise
        yield_values = (100 + 2 * temperature + 1 * flow_rate 
                       - 0.01 * temperature**2 - 0.005 * flow_rate**2 
                       + np.random.normal(0, 1, n_samples))
        
        # Constraint: pressure <= 90
        pressure = 0.5 * temperature + 0.2 * flow_rate + np.random.normal(0, 2, n_samples)
        
        # Environmental constraint: co2_intensity <= 2.0
        co2_intensity = 0.02 * temperature + 0.01 * flow_rate + np.random.normal(0, 0.1, n_samples)
        
        data_df = pd.DataFrame({
            'temperature': temperature,
            'flow_rate': flow_rate,
            'yield': yield_values,
            'pressure': pressure,
            'co2_intensity': co2_intensity
        })
        
        # Save data
        data_path = Path(temp_dir) / "data.csv"
        data_df.to_csv(data_path, index=False)
        
        # Create simple causal graph
        graph = CausalGraph()
        for var in ['temperature', 'flow_rate', 'yield', 'pressure', 'co2_intensity']:
            graph.add_node(var, name=var.replace('_', ' ').title())
        
        # Simple structure: inputs -> outputs
        graph.add_edge('temperature', 'yield', confidence=1.0)
        graph.add_edge('flow_rate', 'yield', confidence=1.0)
        graph.add_edge('temperature', 'pressure', confidence=1.0)
        graph.add_edge('flow_rate', 'pressure', confidence=1.0)
        graph.add_edge('temperature', 'co2_intensity', confidence=1.0)
        graph.add_edge('flow_rate', 'co2_intensity', confidence=1.0)
        
        graph_path = Path(temp_dir) / "graph.json"
        graph.save_to_json(str(graph_path))
        
        # Fit models
        model_path = Path(temp_dir) / "model.pkl"
        fit_scm(
            graph_path=str(graph_path),
            clean_dir=temp_dir,
            n_boot=5,
            model_output_path=str(model_path)
        )
        
        return temp_dir, str(graph_path), str(model_path), data_df
    
    def test_optimization_unconstrained(self):
        """Test unconstrained optimization."""
        temp_dir, graph_path, model_path, data_df = self.create_optimization_scenario()
        
        try:
            # Maximize yield without constraints
            result = optimize_with_surrogate(
                graph_path=graph_path,
                model_path=model_path,
                decision_vars=["temperature", "flow_rate"],
                objective="max",
                target_kpi="yield",
                constraints=None,
                decision_bounds={
                    "temperature": (60.0, 100.0),
                    "flow_rate": (50.0, 200.0)
                },
                n_surrogate_samples=40  # Small for testing
            )
            
            # Verify optimization succeeded
            assert result["status"] == "success"
            assert "optimal_decisions" in result
            assert "expected_kpi" in result
            
            optimal_decisions = result["optimal_decisions"]
            assert "temperature" in optimal_decisions
            assert "flow_rate" in optimal_decisions
            
            # Check bounds are respected
            assert 60.0 <= optimal_decisions["temperature"] <= 100.0
            assert 50.0 <= optimal_decisions["flow_rate"] <= 200.0
            
            # Check expected KPI structure
            expected_kpi = result["expected_kpi"]
            assert "mean" in expected_kpi
            assert "p5" in expected_kpi
            assert "p95" in expected_kpi
            
            print(f"Optimal decisions: {optimal_decisions}")
            print(f"Expected yield: {expected_kpi['mean']:.2f}")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_optimization_with_constraints(self):
        """Test optimization with constraints."""
        temp_dir, graph_path, model_path, data_df = self.create_optimization_scenario()
        
        try:
            # Maximize yield subject to pressure <= 90 and co2_intensity <= 2.0
            constraints = [
                {"lhs": "pressure", "op": "<=", "rhs": 90.0},
                {"lhs": "co2_intensity", "op": "<=", "rhs": 2.0}
            ]
            
            result = optimize_with_surrogate(
                graph_path=graph_path,
                model_path=model_path,
                decision_vars=["temperature", "flow_rate"],
                objective="max",
                target_kpi="yield",
                constraints=constraints,
                decision_bounds={
                    "temperature": (60.0, 100.0),
                    "flow_rate": (50.0, 200.0)
                },
                n_surrogate_samples=40
            )
            
            # Verify optimization succeeded
            assert result["status"] == "success"
            
            optimal_decisions = result["optimal_decisions"]
            expected_kpi = result["expected_kpi"]
            
            # Check constraint validation
            assert "constraint_validation" in result
            constraint_validation = result["constraint_validation"]
            
            # Constraints should be satisfied (within tolerance)
            if "pressure" in constraint_validation:
                pressure_constraint = constraint_validation["pressure"]
                assert pressure_constraint["satisfied"] or pressure_constraint["actual_value"] <= 91.0  # Small tolerance
            
            if "co2_intensity" in constraint_validation:
                co2_constraint = constraint_validation["co2_intensity"]
                assert co2_constraint["satisfied"] or co2_constraint["actual_value"] <= 2.1  # Small tolerance
            
            print(f"Constrained optimal decisions: {optimal_decisions}")
            print(f"Expected yield: {expected_kpi['mean']:.2f}")
            print(f"Constraint validation: {constraint_validation}")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_optimization_minimization(self):
        """Test minimization objective."""
        temp_dir, graph_path, model_path, data_df = self.create_optimization_scenario()
        
        try:
            # Minimize co2_intensity (environmental objective)
            result = optimize_with_surrogate(
                graph_path=graph_path,
                model_path=model_path,
                decision_vars=["temperature", "flow_rate"],
                objective="min",
                target_kpi="co2_intensity",
                constraints=None,
                decision_bounds={
                    "temperature": (60.0, 100.0),
                    "flow_rate": (50.0, 200.0)
                },
                n_surrogate_samples=30
            )
            
            assert result["status"] == "success"
            
            optimal_decisions = result["optimal_decisions"]
            expected_kpi = result["expected_kpi"]
            
            # For minimization, optimal values should be at lower bounds
            # (since co2_intensity increases with both temperature and flow_rate)
            assert optimal_decisions["temperature"] <= 70.0  # Should be near lower bound
            assert optimal_decisions["flow_rate"] <= 80.0    # Should be near lower bound
            
            print(f"Minimization optimal decisions: {optimal_decisions}")
            print(f"Expected co2_intensity: {expected_kpi['mean']:.3f}")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_optimization_infeasible(self):
        """Test handling of infeasible optimization problems."""
        temp_dir, graph_path, model_path, data_df = self.create_optimization_scenario()
        
        try:
            # Create infeasible constraints
            constraints = [
                {"lhs": "temperature", "op": ">=", "rhs": 110.0},  # Impossible given bounds
                {"lhs": "temperature", "op": "<=", "rhs": 50.0}   # Contradictory
            ]
            
            result = optimize_with_surrogate(
                graph_path=graph_path,
                model_path=model_path,
                decision_vars=["temperature", "flow_rate"],
                objective="max",
                target_kpi="yield",
                constraints=constraints,
                decision_bounds={
                    "temperature": (60.0, 100.0),
                    "flow_rate": (50.0, 200.0)
                },
                n_surrogate_samples=20
            )
            
            # Should detect infeasibility
            assert result["status"] in ["failed", "error"]
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestOptimizationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_decision_variable(self):
        """Test optimization with single decision variable."""
        # This would require creating a minimal setup
        pass
    
    def test_invalid_constraints(self):
        """Test handling of invalid constraint specifications."""
        # Test with invalid operators, missing variables, etc.
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])