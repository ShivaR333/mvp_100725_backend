import pytest
import numpy as np
import pandas as pd
import networkx as nx
import sqlite3
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch

from app.graph import (
    CausalGraph,
    discover_from_data,
    fuse_graphs,
    load_expert_edges_from_sqlite
)


class TestCausalGraph:
    """Test the CausalGraph wrapper class."""
    
    def test_init(self):
        """Test CausalGraph initialization."""
        graph = CausalGraph()
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0
        assert graph.is_acyclic() == True
    
    def test_add_node(self):
        """Test adding nodes."""
        graph = CausalGraph()
        graph.add_node("A", name="Variable A", type="continuous")
        
        assert "A" in graph.nodes()
        assert graph.graph.nodes["A"]["name"] == "Variable A"
        assert graph.graph.nodes["A"]["type"] == "continuous"
    
    def test_add_edge(self):
        """Test adding edges."""
        graph = CausalGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B", confidence=0.8)
        
        assert graph.has_edge("A", "B")
        assert graph.get_edge_confidence("A", "B") == 0.8
    
    def test_cycle_detection(self):
        """Test cycle detection."""
        graph = CausalGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        
        # Add edges to create cycle: A -> B -> C -> A
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        
        assert not graph.is_acyclic()
        cycles = graph.get_cycles()
        assert len(cycles) > 0
    
    def test_save_load_json(self):
        """Test JSON serialization."""
        graph = CausalGraph()
        graph.add_node("A", name="Variable A")
        graph.add_node("B", name="Variable B")
        graph.add_edge("A", "B", confidence=0.9)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            graph.save_to_json(json_path)
            
            # Load into new graph
            new_graph = CausalGraph()
            new_graph.load_from_json(json_path)
            
            assert "A" in new_graph.nodes()
            assert "B" in new_graph.nodes()
            assert new_graph.has_edge("A", "B")
            assert new_graph.get_edge_confidence("A", "B") == 0.9
            
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


class TestDiscoverFromData:
    """Test the discover_from_data function."""
    
    def test_discover_from_data_empty_dir(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = discover_from_data(temp_dir)
            assert isinstance(result, nx.DiGraph)
            assert len(result.nodes()) == 0
    
    def test_discover_from_data_with_csv(self):
        """Test with CSV data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            np.random.seed(42)
            n_samples = 100
            
            # Create causally related data: A -> B -> C
            A = np.random.normal(0, 1, n_samples)
            B = 2 * A + np.random.normal(0, 0.5, n_samples)
            C = 1.5 * B + np.random.normal(0, 0.5, n_samples)
            
            df = pd.DataFrame({
                'variable_A': A,
                'variable_B': B,
                'variable_C': C
            })
            
            csv_path = Path(temp_dir) / "test_data.csv"
            df.to_csv(csv_path, index=False)
            
            # Run discovery
            result = discover_from_data(temp_dir)
            
            assert isinstance(result, nx.DiGraph)
            # Should have at least the three variables
            assert len(result.nodes()) >= 3
    
    def test_discover_from_data_with_whitelist(self):
        """Test with whitelist constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            np.random.seed(42)
            n_samples = 100
            
            A = np.random.normal(0, 1, n_samples)
            B = np.random.normal(0, 1, n_samples)
            C = np.random.normal(0, 1, n_samples)
            
            df = pd.DataFrame({
                'A': A,
                'B': B,
                'C': C
            })
            
            csv_path = Path(temp_dir) / "test_data.csv"
            df.to_csv(csv_path, index=False)
            
            # Run discovery with whitelist
            whitelist = [("A", "B")]
            result = discover_from_data(temp_dir, whitelist=whitelist)
            
            assert isinstance(result, nx.DiGraph)


class TestFuseGraphs:
    """Test the fuse_graphs function."""
    
    def test_fuse_graphs_basic(self):
        """Test basic graph fusion."""
        # Create expert graph
        expert_graph = nx.DiGraph()
        expert_graph.add_node("A", name="Variable A")
        expert_graph.add_node("B", name="Variable B")
        expert_graph.add_edge("A", "B", confidence=0.9, edge_type="expert")
        
        # Create data graph
        data_graph = nx.DiGraph()
        data_graph.add_node("A", name="Variable A")
        data_graph.add_node("B", name="Variable B")
        data_graph.add_node("C", name="Variable C")
        data_graph.add_edge("A", "B", confidence=0.7, edge_type="data")
        data_graph.add_edge("B", "C", confidence=0.8, edge_type="data")
        
        # Fuse graphs
        fused = fuse_graphs(expert_graph, data_graph)
        
        # Check fusion results
        assert "A" in fused.nodes()
        assert "B" in fused.nodes()
        assert "C" in fused.nodes()
        
        # Edge A->B should be fused: 1 - (1-0.9)*(1-0.7) = 1 - 0.1*0.3 = 0.97
        assert fused.has_edge("A", "B")
        confidence_ab = fused.get_edge_confidence("A", "B")
        expected_confidence = 1 - (1 - 0.9) * (1 - 0.7)
        assert abs(confidence_ab - expected_confidence) < 0.01
        
        # Edge B->C should exist with data confidence
        assert fused.has_edge("B", "C")
        assert fused.get_edge_confidence("B", "C") == 0.8
    
    def test_fuse_graphs_confidence_filter(self):
        """Test confidence filtering (< 0.2)."""
        # Create expert graph with low confidence
        expert_graph = nx.DiGraph()
        expert_graph.add_node("A")
        expert_graph.add_node("B")
        expert_graph.add_edge("A", "B", confidence=0.1)
        
        # Create empty data graph
        data_graph = nx.DiGraph()
        
        # Fuse graphs
        fused = fuse_graphs(expert_graph, data_graph)
        
        # Low confidence edge should be filtered out
        assert not fused.has_edge("A", "B")
    
    def test_fuse_graphs_cycle_breaking(self):
        """Test cycle breaking."""
        # Create expert graph with cycle
        expert_graph = nx.DiGraph()
        expert_graph.add_node("A")
        expert_graph.add_node("B")
        expert_graph.add_node("C")
        expert_graph.add_edge("A", "B", confidence=0.9)
        expert_graph.add_edge("B", "C", confidence=0.8)
        expert_graph.add_edge("C", "A", confidence=0.3)  # Lowest confidence
        
        # Create empty data graph
        data_graph = nx.DiGraph()
        
        # Fuse graphs
        fused = fuse_graphs(expert_graph, data_graph)
        
        # Should be acyclic after cycle breaking
        assert fused.is_acyclic()
        
        # The lowest confidence edge (C->A) should be removed
        assert not fused.has_edge("C", "A")
        assert fused.has_edge("A", "B")
        assert fused.has_edge("B", "C")


class TestLoadExpertEdges:
    """Test loading expert edges from SQLite."""
    
    def test_load_expert_edges_from_sqlite(self):
        """Test loading edges from SQLite database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Create test database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table
            cursor.execute('''
                CREATE TABLE edge_candidates (
                    source TEXT,
                    target TEXT,
                    confidence REAL
                )
            ''')
            
            # Insert test data
            test_edges = [
                ("A", "B", 0.8),
                ("B", "C", 0.7),
                ("C", "D", 0.5),  # Below threshold
                ("D", "E", 0.9)
            ]
            
            cursor.executemany(
                'INSERT INTO edge_candidates VALUES (?, ?, ?)',
                test_edges
            )
            
            conn.commit()
            conn.close()
            
            # Load edges with threshold 0.6
            result = load_expert_edges_from_sqlite(db_path, 0.6)
            
            assert isinstance(result, nx.DiGraph)
            assert len(result.nodes()) == 4  # A, B, C, D, E (minus filtered edge)
            assert len(result.edges()) == 3  # All edges except C->D
            
            assert result.has_edge("A", "B")
            assert result.has_edge("B", "C")
            assert not result.has_edge("C", "D")  # Below threshold
            assert result.has_edge("D", "E")
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_load_expert_edges_empty_db(self):
        """Test with empty database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Create empty database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE edge_candidates (
                    source TEXT,
                    target TEXT,
                    confidence REAL
                )
            ''')
            conn.commit()
            conn.close()
            
            # Load edges
            result = load_expert_edges_from_sqlite(db_path, 0.6)
            
            assert isinstance(result, nx.DiGraph)
            assert len(result.nodes()) == 0
            assert len(result.edges()) == 0
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__])