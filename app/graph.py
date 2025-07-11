import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import joblib

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import PC
from pgmpy.estimators.CITests import pearsonr
from neo4j import GraphDatabase
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample  # Use resample instead of bootstrap
from econml.dml import LinearDML, CausalForestDML
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import cvxpy as cp
from pyDOE2 import lhs
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class GraphNode(BaseModel):
    """Represents a node in the causal graph."""
    id: str
    name: str
    type: str = "variable"
    metadata: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    """Represents an edge in the causal graph."""
    source: str
    target: str
    confidence: float
    edge_type: str = "causal"
    metadata: Dict[str, Any] = {}


class CausalGraph:
    """Wrapper around NetworkX DiGraph for causal analysis."""
    
    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph = graph or nx.DiGraph()
        self._neo4j_driver = None
        self._neo4j_uri = None
        self._neo4j_user = None
        self._neo4j_password = None
    
    def add_node(self, node_id: str, **attributes) -> None:
        """Add a node to the graph."""
        self.graph.add_node(node_id, **attributes)
    
    def add_edge(self, source: str, target: str, confidence: float = 1.0, **attributes) -> None:
        """Add an edge to the graph."""
        self.graph.add_edge(source, target, confidence=confidence, **attributes)
    
    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the graph."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
    
    def get_edge_confidence(self, source: str, target: str) -> float:
        """Get the confidence of an edge."""
        if self.graph.has_edge(source, target):
            return self.graph[source][target].get('confidence', 0.0)
        return 0.0
    
    def set_edge_confidence(self, source: str, target: str, confidence: float) -> None:
        """Set the confidence of an edge."""
        if self.graph.has_edge(source, target):
            self.graph[source][target]['confidence'] = confidence
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists."""
        return self.graph.has_edge(source, target)
    
    def nodes(self) -> List[str]:
        """Get all nodes."""
        return list(self.graph.nodes())
    
    def edges(self) -> List[Tuple[str, str]]:
        """Get all edges."""
        return list(self.graph.edges())
    
    def get_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph."""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except nx.NetworkXNoCycle:
            return []
    
    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic."""
        return nx.is_directed_acyclic_graph(self.graph)
    
    def to_networkx(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self.graph
    
    def copy(self) -> 'CausalGraph':
        """Create a copy of the graph."""
        return CausalGraph(self.graph.copy())
    
    def configure_neo4j(self, uri: str, user: str, password: str) -> None:
        """Configure Neo4j connection."""
        self._neo4j_uri = uri
        self._neo4j_user = user
        self._neo4j_password = password
        self._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def save_to_neo4j(self) -> None:
        """Save graph to Neo4j database."""
        if not self._neo4j_driver:
            raise ValueError("Neo4j connection not configured")
        
        with self._neo4j_driver.session() as session:
            # Clear existing graph
            session.run("MATCH (n) DETACH DELETE n")
            
            # Add nodes
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                session.run(
                    "CREATE (n:Variable {id: $id, name: $name, type: $type})",
                    id=node_id,
                    name=node_data.get('name', node_id),
                    type=node_data.get('type', 'variable')
                )
            
            # Add edges
            for source, target in self.graph.edges():
                edge_data = self.graph[source][target]
                session.run(
                    """
                    MATCH (s:Variable {id: $source}), (t:Variable {id: $target})
                    CREATE (s)-[r:CAUSES {confidence: $confidence, edge_type: $edge_type}]->(t)
                    """,
                    source=source,
                    target=target,
                    confidence=edge_data.get('confidence', 1.0),
                    edge_type=edge_data.get('edge_type', 'causal')
                )
        
        logger.info(f"Saved graph to Neo4j: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
    
    def save_to_json(self, filepath: str) -> None:
        """Save graph to JSON file."""
        graph_data = {
            'nodes': [
                {
                    'id': node_id,
                    **self.graph.nodes[node_id]
                }
                for node_id in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    **self.graph[source][target]
                }
                for source, target in self.graph.edges()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Saved graph to JSON: {filepath}")
    
    def load_from_json(self, filepath: str) -> None:
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes
        for node_data in graph_data['nodes']:
            node_id = node_data.pop('id')
            self.graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in graph_data['edges']:
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            self.graph.add_edge(source, target, **edge_data)
        
        logger.info(f"Loaded graph from JSON: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")


def discover_from_data(clean_dir: str, whitelist: Optional[List[Tuple[str, str]]] = None, 
                      blacklist: Optional[List[Tuple[str, str]]] = None) -> nx.DiGraph:
    """
    Discover causal structure from data using PC-Stable algorithm.
    
    Args:
        clean_dir: Directory containing clean data files (CSV/Parquet)
        whitelist: List of (source, target) edges that must remain
        blacklist: List of (source, target) edges that must be forbidden
    
    Returns:
        NetworkX DiGraph representing discovered causal structure
    """
    logger.info(f"Discovering causal structure from data in {clean_dir}")
    
    # Load all data files and combine
    data_dir = Path(clean_dir)
    all_data = []
    
    for file_path in data_dir.glob("*.csv"):
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    for file_path in data_dir.glob("*.parquet"):
        df = pd.read_parquet(file_path)
        all_data.append(df)
    
    if not all_data:
        logger.warning("No data files found for causal discovery")
        return nx.DiGraph()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Select only numeric columns
    numeric_cols = combined_data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for causal discovery")
        return nx.DiGraph()
    
    data_matrix = combined_data[numeric_cols].dropna()
    
    if data_matrix.empty:
        logger.warning("No valid data after removing NaN values")
        return nx.DiGraph()
    
    logger.info(f"Using {len(numeric_cols)} variables with {len(data_matrix)} samples")
    
    # Create PC estimator
    pc_estimator = PC(data_matrix)
    
    # Convert whitelist/blacklist to proper format
    edge_whitelist = []
    edge_blacklist = []
    
    if whitelist:
        for source, target in whitelist:
            if source in numeric_cols and target in numeric_cols:
                edge_whitelist.append((source, target))
    
    if blacklist:
        for source, target in blacklist:
            if source in numeric_cols and target in numeric_cols:
                edge_blacklist.append((source, target))
    
    # Run PC-Stable algorithm
    try:
        if edge_whitelist or edge_blacklist:
            estimated_dag = pc_estimator.estimate(
                ci_test=pearsonr,
                white_list=edge_whitelist if edge_whitelist else None,
                black_list=edge_blacklist if edge_blacklist else None
            )
        else:
            estimated_dag = pc_estimator.estimate(ci_test=pearsonr)
        
        # Convert to NetworkX DiGraph
        nx_graph = nx.DiGraph()
        
        # Add nodes
        for node in estimated_dag.nodes():
            nx_graph.add_node(node, name=node, type='variable')
        
        # Add edges with default confidence
        for source, target in estimated_dag.edges():
            nx_graph.add_edge(source, target, confidence=0.8, edge_type='data_driven')
        
        logger.info(f"Discovered {len(nx_graph.nodes())} nodes and {len(nx_graph.edges())} edges")
        
        return nx_graph
        
    except Exception as e:
        logger.error(f"Error in causal discovery: {e}")
        return nx.DiGraph()


def fuse_graphs(G_expert: nx.DiGraph, G_data: nx.DiGraph) -> CausalGraph:
    """
    Fuse expert knowledge graph with data-driven graph.
    
    Args:
        G_expert: Expert knowledge graph
        G_data: Data-driven graph
    
    Returns:
        CausalGraph: Fused graph with confidence scores
    """
    logger.info("Fusing expert and data-driven graphs")
    
    fused_graph = CausalGraph()
    
    # Collect all nodes from both graphs
    all_nodes = set(G_expert.nodes()) | set(G_data.nodes())
    
    # Add all nodes to fused graph
    for node in all_nodes:
        # Prefer expert node attributes if available
        if node in G_expert.nodes():
            node_attrs = G_expert.nodes[node]
        else:
            node_attrs = G_data.nodes[node] if node in G_data.nodes() else {}
        
        fused_graph.add_node(node, **node_attrs)
    
    # Collect all edges from both graphs
    all_edges = set(G_expert.edges()) | set(G_data.edges())
    
    # Process each edge
    for source, target in all_edges:
        conf_expert = 0.0
        conf_data = 0.0
        
        # Get expert confidence
        if G_expert.has_edge(source, target):
            conf_expert = G_expert[source][target].get('confidence', 0.8)
        
        # Get data confidence
        if G_data.has_edge(source, target):
            conf_data = G_data[source][target].get('confidence', 0.8)
        
        # Fuse confidences: 1 - (1-conf_exp)*(1-conf_data)
        if conf_expert > 0 and conf_data > 0:
            # Both sources have the edge
            fused_confidence = 1 - (1 - conf_expert) * (1 - conf_data)
            edge_type = 'fused'
        elif conf_expert > 0:
            # Only expert has the edge
            fused_confidence = conf_expert
            edge_type = 'expert'
        else:
            # Only data has the edge
            fused_confidence = conf_data
            edge_type = 'data_driven'
        
        # Add edge if confidence >= 0.2
        if fused_confidence >= 0.2:
            # Combine edge attributes
            edge_attrs = {}
            if G_expert.has_edge(source, target):
                edge_attrs.update(G_expert[source][target])
            if G_data.has_edge(source, target):
                edge_attrs.update(G_data[source][target])
            
            edge_attrs['confidence'] = fused_confidence
            edge_attrs['edge_type'] = edge_type
            
            fused_graph.add_edge(source, target, **edge_attrs)
    
    logger.info(f"Initial fused graph: {len(fused_graph.nodes())} nodes, {len(fused_graph.edges())} edges")
    
    # Break cycles by removing lowest confidence edges
    cycles = fused_graph.get_cycles()
    
    while cycles:
        logger.info(f"Found {len(cycles)} cycles, breaking...")
        
        # Find the edge with lowest confidence across all cycles
        min_confidence = float('inf')
        edge_to_remove = None
        
        for cycle in cycles:
            for i in range(len(cycle)):
                source = cycle[i]
                target = cycle[(i + 1) % len(cycle)]
                
                if fused_graph.has_edge(source, target):
                    confidence = fused_graph.get_edge_confidence(source, target)
                    if confidence < min_confidence:
                        min_confidence = confidence
                        edge_to_remove = (source, target)
        
        # Remove the edge with lowest confidence
        if edge_to_remove:
            logger.info(f"Removing edge {edge_to_remove} with confidence {min_confidence}")
            fused_graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
        
        # Recompute cycles
        cycles = fused_graph.get_cycles()
    
    logger.info(f"Final fused graph: {len(fused_graph.nodes())} nodes, {len(fused_graph.edges())} edges")
    
    return fused_graph


def load_expert_edges_from_sqlite(db_path: str, confidence_threshold: float = 0.6) -> nx.DiGraph:
    """
    Load expert edges from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        NetworkX DiGraph with expert edges
    """
    logger.info(f"Loading expert edges from {db_path}")
    
    expert_graph = nx.DiGraph()
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Load edges with confidence >= threshold
        query = """
        SELECT source, target, confidence 
        FROM edge_candidates 
        WHERE confidence >= ?
        """
        
        df = pd.read_sql_query(query, conn, params=(confidence_threshold,))
        conn.close()
        
        if df.empty:
            logger.warning("No expert edges found meeting confidence threshold")
            return expert_graph
        
        # Add nodes and edges
        for _, row in df.iterrows():
            source = str(row['source'])
            target = str(row['target'])
            confidence = float(row['confidence'])
            
            # Add nodes if not present
            if source not in expert_graph.nodes():
                expert_graph.add_node(source, name=source, type='variable')
            if target not in expert_graph.nodes():
                expert_graph.add_node(target, name=target, type='variable')
            
            # Add edge
            expert_graph.add_edge(source, target, confidence=confidence, edge_type='expert')
        
        logger.info(f"Loaded {len(expert_graph.nodes())} nodes and {len(expert_graph.edges())} expert edges")
        
    except Exception as e:
        logger.error(f"Error loading expert edges: {e}")
    
    return expert_graph


def load_parents_and_target(parents: List[str], target: str, clean_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load parent variables and target variable from clean data directory.
    
    Args:
        parents: List of parent variable names
        target: Target variable name
        clean_dir: Directory containing clean data files
        
    Returns:
        Tuple of (X: parent variables DataFrame, y: target variable Series)
    """
    clean_data_path = Path(clean_dir)
    
    # Load all data files and combine
    all_data = []
    
    for file_path in clean_data_path.glob("*.csv"):
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    for file_path in clean_data_path.glob("*.parquet"):
        df = pd.read_parquet(file_path)
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No data files found in clean directory")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Check if all required variables are present
    required_vars = parents + [target]
    missing_vars = [var for var in required_vars if var not in combined_data.columns]
    
    if missing_vars:
        raise ValueError(f"Missing variables in data: {missing_vars}")
    
    # Extract parent variables (X) and target variable (y)
    X = combined_data[parents].copy()
    y = combined_data[target].copy()
    
    # Remove rows with missing values
    combined_xy = pd.concat([X, y], axis=1).dropna()
    
    if combined_xy.empty:
        raise ValueError("No valid data after removing missing values")
    
    X = combined_xy[parents]
    y = combined_xy[target]
    
    logger.info(f"Loaded data for {target}: {len(X)} samples, {len(parents)} parents")
    
    return X, y


def bootstrap_se(estimator, X: pd.DataFrame, y: pd.Series, B: int = 250) -> Dict[str, float]:
    """
    Calculate bootstrap standard errors for causal effect estimates.
    
    Args:
        estimator: Fitted econML estimator
        X: Parent variables
        y: Target variable
        B: Number of bootstrap samples
        
    Returns:
        Dictionary of bootstrap standard errors
    """
    logger.info(f"Calculating bootstrap standard errors with {B} samples")
    
    n_samples = len(X)
    effects = []
    
    # Generate bootstrap samples
    for b in range(B):
        # Sample with replacement
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[bootstrap_idx].reset_index(drop=True)
        y_boot = y.iloc[bootstrap_idx].reset_index(drop=True)
        
        try:
            # Fit model on bootstrap sample
            estimator_boot = type(estimator)(**estimator.get_params())
            
            # For DML estimators, we need to handle the fitting differently
            if hasattr(estimator, 'fit'):
                if len(X_boot.columns) == 1:
                    # Single treatment case
                    T_boot = X_boot.iloc[:, 0]
                    X_controls = pd.DataFrame(index=X_boot.index)  # Empty controls
                    estimator_boot.fit(y_boot, T_boot, X=X_controls, inference='blb')
                else:
                    # Multiple treatments case
                    T_boot = X_boot.iloc[:, 0]  # First parent as treatment
                    X_controls = X_boot.iloc[:, 1:]  # Rest as controls
                    estimator_boot.fit(y_boot, T_boot, X=X_controls, inference='blb')
                
                # Get effect estimates
                if hasattr(estimator_boot, 'effect'):
                    effect = estimator_boot.effect(X_boot.iloc[:1])  # Single prediction
                    effects.append(float(effect[0]) if hasattr(effect, '__len__') else float(effect))
                
        except Exception as e:
            logger.warning(f"Bootstrap iteration {b} failed: {e}")
            continue
    
    if not effects:
        logger.warning("No successful bootstrap iterations")
        return {"bootstrap_se": 0.0}
    
    # Calculate standard error
    bootstrap_se = np.std(effects)
    
    return {
        "bootstrap_se": bootstrap_se,
        "bootstrap_mean": np.mean(effects),
        "bootstrap_samples": len(effects)
    }


def fit_scm(graph_path: str, clean_dir: str, outcome_nodes: Optional[List[str]] = None, 
           n_boot: int = 250, model_output_path: str = "model.pkl") -> Dict[str, Any]:
    """
    Fit a Structural Causal Model (SCM) using econML estimators.
    
    Args:
        graph_path: Path to the causal graph JSON file
        clean_dir: Directory containing clean data files
        outcome_nodes: List of outcome nodes to focus on (None = all endogenous nodes)
        n_boot: Number of bootstrap samples for standard errors
        model_output_path: Path to save the fitted models
        
    Returns:
        Dictionary containing fitted models and metadata
    """
    logger.info("Starting SCM fitting process")
    
    # Load the causal graph
    graph = CausalGraph()
    graph.load_from_json(graph_path)
    
    # Get topological ordering
    try:
        topo_order = list(nx.topological_sort(graph.to_networkx()))
    except nx.NetworkXError:
        raise ValueError("Graph is not acyclic - cannot perform topological sort")
    
    logger.info(f"Processing {len(topo_order)} nodes in topological order")
    
    models = {}
    
    for node in topo_order:
        # Get parents of the current node
        parents = list(graph.to_networkx().predecessors(node))
        
        # Skip exogenous nodes (no parents)
        if not parents:
            logger.info(f"Skipping exogenous node: {node}")
            continue
        
        # Skip if outcome_nodes is specified and node is not in it
        if outcome_nodes is not None and node not in outcome_nodes:
            logger.info(f"Skipping node {node} (not in outcome_nodes)")
            continue
        
        logger.info(f"Fitting model for node {node} with parents {parents}")
        
        try:
            # Load data for this node
            X, y = load_parents_and_target(parents, node, clean_dir)
            
            # Choose estimator based on complexity and data type
            if len(parents) <= 3 and y.dtype != 'object':
                # Use LinearDML for simple cases
                logger.info(f"Using LinearDML for node {node}")
                est = LinearDML(
                    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                    model_t=RandomForestRegressor(n_estimators=100, random_state=42),
                    featurizer=None,
                    random_state=42
                )
            else:
                # Use CausalForestDML for complex cases
                logger.info(f"Using CausalForestDML for node {node}")
                est = CausalForestDML(
                    n_estimators=400,
                    min_samples_leaf=10,
                    max_depth=None,
                    random_state=42
                )
            
            # Fit the model
            if len(parents) == 1:
                # Single treatment case
                T = X.iloc[:, 0]
                X_controls = pd.DataFrame(index=X.index)  # Empty controls
                est.fit(y, T, X=X_controls, inference='blb')
            else:
                # Multiple treatments - use first parent as treatment, rest as controls
                T = X.iloc[:, 0]
                X_controls = X.iloc[:, 1:]
                est.fit(y, T, X=X_controls, inference='blb')
            
            # Calculate bootstrap standard errors
            logger.info(f"Calculating bootstrap SEs for node {node}")
            boot_se = bootstrap_se(est, X, y, B=n_boot)
            
            # Store model information
            models[node] = {
                "model": est,
                "parents": parents,
                "boot_se": boot_se,
                "n_samples": len(X),
                "model_type": type(est).__name__
            }
            
            logger.info(f"Successfully fitted model for node {node}")
            
        except Exception as e:
            logger.error(f"Error fitting model for node {node}: {e}")
            continue
    
    # Add metadata
    model_dict = {
        "models": models,
        "metadata": {
            "graph_path": graph_path,
            "clean_dir": clean_dir,
            "outcome_nodes": outcome_nodes,
            "n_boot": n_boot,
            "topological_order": topo_order,
            "fitted_nodes": list(models.keys())
        }
    }
    
    # Serialize models to file
    logger.info(f"Saving models to {model_output_path}")
    joblib.dump(model_dict, model_output_path)
    
    logger.info(f"SCM fitting completed. Fitted {len(models)} models.")
    
    return model_dict


def load_scm_models(model_path: str) -> Dict[str, Any]:
    """
    Load fitted SCM models from file.
    
    Args:
        model_path: Path to the saved models file
        
    Returns:
        Dictionary containing fitted models and metadata
    """
    return joblib.load(model_path)


def predict_scm(models: Dict[str, Any], interventions: Dict[str, float] = None) -> Dict[str, float]:
    """
    Make predictions using fitted SCM models.
    
    Args:
        models: Dictionary of fitted models (from fit_scm or load_scm_models)
        interventions: Dictionary of interventions {node: value}
        
    Returns:
        Dictionary of predicted values for each node
    """
    if interventions is None:
        interventions = {}
    
    model_dict = models["models"] if "models" in models else models
    topo_order = models.get("metadata", {}).get("topological_order", list(model_dict.keys()))
    
    predictions = {}
    
    # Process nodes in topological order
    for node in topo_order:
        if node in interventions:
            # Use intervention value
            predictions[node] = interventions[node]
        elif node in model_dict:
            # Predict using fitted model
            model_info = model_dict[node]
            estimator = model_info["model"]
            parents = model_info["parents"]
            
            # Get parent values
            parent_values = []
            for parent in parents:
                if parent in predictions:
                    parent_values.append(predictions[parent])
                else:
                    # Use default value (0) if parent not predicted yet
                    parent_values.append(0.0)
            
            # Make prediction
            X_pred = pd.DataFrame([parent_values], columns=parents)
            
            try:
                if hasattr(estimator, 'effect'):
                    if len(parents) == 1:
                        pred = estimator.effect(X_pred.iloc[:, 0])
                    else:
                        pred = estimator.effect(X_pred.iloc[:, 0], X=X_pred.iloc[:, 1:])
                    predictions[node] = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                else:
                    predictions[node] = 0.0
            except Exception as e:
                logger.warning(f"Prediction failed for node {node}: {e}")
                predictions[node] = 0.0
        else:
            # Exogenous node - use default value
            predictions[node] = 0.0
    
    return predictions


def simulate_interventions(
    graph_path: str, 
    model_path: str, 
    interventions: Dict[str, float], 
    n_samples: int = 1000,
    kpi_nodes: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Perform Monte Carlo simulation with interventions using do-calculus.
    
    Args:
        graph_path: Path to the fused causal graph JSON
        model_path: Path to the fitted SCM models
        interventions: Dictionary of interventions {node: value}
        n_samples: Number of Monte Carlo samples
        kpi_nodes: List of KPI nodes to focus on (None = all nodes)
        
    Returns:
        Dictionary with statistics for each node: {node: {mean, p5, p95}}
    """
    logger.info(f"Starting Monte Carlo simulation with {n_samples} samples")
    
    # Load graph and models
    graph = CausalGraph()
    graph.load_from_json(graph_path)
    models = load_scm_models(model_path)
    
    # Get topological order
    topo_order = list(nx.topological_sort(graph.to_networkx()))
    model_dict = models["models"]
    
    # Identify KPI nodes
    if kpi_nodes is None:
        # Default: all endogenous nodes (nodes with parents)
        kpi_nodes = [node for node in topo_order if list(graph.to_networkx().predecessors(node))]
    
    logger.info(f"KPI nodes: {kpi_nodes}")
    logger.info(f"Interventions: {interventions}")
    
    # Monte Carlo sampling
    samples = {node: [] for node in topo_order}
    
    for i in range(n_samples):
        # Sample values for each node in topological order
        node_values = {}
        
        for node in topo_order:
            if node in interventions:
                # Apply intervention: replace structural equation with fixed value
                node_values[node] = interventions[node]
                
            else:
                # Sample from structural equation
                parents = list(graph.to_networkx().predecessors(node))
                
                if not parents:
                    # Exogenous node: sample from prior
                    if node in model_dict:
                        # Use noise from fitted model if available
                        node_values[node] = np.random.normal(0, 1)
                    else:
                        # Default prior
                        node_values[node] = np.random.normal(0, 1)
                        
                elif node in model_dict:
                    # Endogenous node with fitted model
                    model_info = model_dict[node]
                    estimator = model_info["model"]
                    
                    # Get parent values
                    parent_values = [node_values.get(parent, 0.0) for parent in parents]
                    X_sample = pd.DataFrame([parent_values], columns=parents)
                    
                    try:
                        # Get predicted effect
                        if len(parents) == 1:
                            effect = estimator.effect(X_sample.iloc[0, 0])
                        else:
                            effect = estimator.effect(X_sample.iloc[0, 0], X=X_sample.iloc[:, 1:])
                        
                        predicted_mean = float(effect[0]) if hasattr(effect, '__len__') else float(effect)
                        
                        # Add noise (assume unit variance for simplicity)
                        noise_std = model_info["boot_se"].get("bootstrap_se", 0.5)
                        noise_std = max(noise_std, 0.1)  # Minimum noise level
                        
                        node_values[node] = predicted_mean + np.random.normal(0, noise_std)
                        
                    except Exception as e:
                        logger.warning(f"Error predicting {node}: {e}")
                        # Fallback: linear combination of parents + noise
                        node_values[node] = np.sum(parent_values) + np.random.normal(0, 0.5)
                        
                else:
                    # Endogenous node without fitted model: use simple linear combination
                    parent_values = [node_values.get(parent, 0.0) for parent in parents]
                    node_values[node] = np.sum(parent_values) + np.random.normal(0, 0.5)
            
            # Store sample
            samples[node].append(node_values[node])
    
    # Calculate statistics for KPI nodes
    results = {}
    
    for node in kpi_nodes:
        if node in samples and samples[node]:
            node_samples = np.array(samples[node])
            
            results[node] = {
                "mean": float(np.mean(node_samples)),
                "p5": float(np.percentile(node_samples, 5)),
                "p95": float(np.percentile(node_samples, 95)),
                "std": float(np.std(node_samples)),
                "median": float(np.median(node_samples))
            }
        else:
            logger.warning(f"No samples for KPI node {node}")
            results[node] = {
                "mean": 0.0,
                "p5": 0.0,
                "p95": 0.0,
                "std": 0.0,
                "median": 0.0
            }
    
    logger.info(f"Simulation completed for {len(results)} KPI nodes")
    return results


def pyro_simulate_interventions(
    graph_path: str,
    model_path: str,
    interventions: Dict[str, float],
    n_samples: int = 1000,
    kpi_nodes: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Perform Monte Carlo simulation using Pyro probabilistic programming.
    
    Args:
        graph_path: Path to the fused causal graph JSON
        model_path: Path to the fitted SCM models
        interventions: Dictionary of interventions {node: value}
        n_samples: Number of Monte Carlo samples
        kpi_nodes: List of KPI nodes to focus on
        
    Returns:
        Dictionary with statistics for each node
    """
    logger.info(f"Starting Pyro simulation with {n_samples} samples")
    
    # Load graph and models
    graph = CausalGraph()
    graph.load_from_json(graph_path)
    models = load_scm_models(model_path)
    
    topo_order = list(nx.topological_sort(graph.to_networkx()))
    model_dict = models["models"]
    
    # Identify KPI nodes
    if kpi_nodes is None:
        kpi_nodes = [node for node in topo_order if list(graph.to_networkx().predecessors(node))]
    
    def causal_model():
        """Define the causal model in Pyro."""
        node_values = {}
        
        for node in topo_order:
            if node in interventions:
                # Intervention: fix the value
                node_values[node] = pyro.sample(
                    node, 
                    dist.Delta(torch.tensor(interventions[node]))
                )
                
            else:
                parents = list(graph.to_networkx().predecessors(node))
                
                if not parents:
                    # Exogenous node
                    node_values[node] = pyro.sample(
                        node,
                        dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
                    )
                    
                elif node in model_dict:
                    # Endogenous node with fitted model
                    model_info = model_dict[node]
                    
                    # Get parent values
                    parent_values = [node_values[parent] for parent in parents]
                    parent_tensor = torch.stack(parent_values) if len(parent_values) > 1 else parent_values[0]
                    
                    # Simple linear model assumption for Pyro
                    # In practice, you'd use the actual fitted model coefficients
                    if len(parents) == 1:
                        mean = parent_tensor * 1.0  # Coefficient placeholder
                    else:
                        mean = torch.sum(parent_tensor) * 0.5  # Simplified
                    
                    noise_std = model_info["boot_se"].get("bootstrap_se", 0.5)
                    noise_std = max(noise_std, 0.1)
                    
                    node_values[node] = pyro.sample(
                        node,
                        dist.Normal(mean, torch.tensor(noise_std))
                    )
                    
                else:
                    # Endogenous node without fitted model
                    parent_values = [node_values[parent] for parent in parents]
                    parent_tensor = torch.stack(parent_values) if len(parent_values) > 1 else parent_values[0]
                    
                    mean = torch.sum(parent_tensor) if len(parent_values) > 1 else parent_tensor
                    
                    node_values[node] = pyro.sample(
                        node,
                        dist.Normal(mean, torch.tensor(0.5))
                    )
        
        return node_values
    
    # Run simulation
    predictive = Predictive(causal_model, num_samples=n_samples)
    samples = predictive()
    
    # Calculate statistics
    results = {}
    
    for node in kpi_nodes:
        if node in samples:
            node_samples = samples[node].detach().numpy()
            
            results[node] = {
                "mean": float(np.mean(node_samples)),
                "p5": float(np.percentile(node_samples, 5)),
                "p95": float(np.percentile(node_samples, 95)),
                "std": float(np.std(node_samples)),
                "median": float(np.median(node_samples))
            }
        else:
            logger.warning(f"No samples for KPI node {node}")
            results[node] = {
                "mean": 0.0,
                "p5": 0.0,
                "p95": 0.0,
                "std": 0.0,
                "median": 0.0
            }
    
    logger.info(f"Pyro simulation completed for {len(results)} KPI nodes")
    return results


class QuadraticSurrogate:
    """Quadratic surrogate model for the causal simulator."""
    
    def __init__(self, decision_vars: List[str], target_kpis: List[str]):
        self.decision_vars = decision_vars
        self.target_kpis = target_kpis
        self.models = {}  # One model per KPI
        self.poly_features = {}
        self.X_train = None
        self.y_train = {}
        self.decision_bounds = {}
        
    def train(self, 
              graph_path: str, 
              model_path: str, 
              n_samples: int = 200,
              decision_bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Train quadratic surrogates using Latin Hypercube Sampling.
        
        Args:
            graph_path: Path to causal graph
            model_path: Path to fitted SCM models
            n_samples: Number of LHS samples for training
            decision_bounds: Bounds for decision variables {var: (min, max)}
            
        Returns:
            R² scores for each KPI surrogate
        """
        logger.info(f"Training quadratic surrogates with {n_samples} LHS samples")
        
        # Set default bounds if not provided
        if decision_bounds is None:
            decision_bounds = {var: (0.0, 100.0) for var in self.decision_vars}
        
        self.decision_bounds = decision_bounds
        
        # Generate Latin Hypercube samples
        n_vars = len(self.decision_vars)
        lhs_samples = lhs(n_vars, samples=n_samples, criterion='maximin')
        
        # Scale to decision variable bounds
        X_samples = np.zeros((n_samples, n_vars))
        for i, var in enumerate(self.decision_vars):
            min_val, max_val = decision_bounds[var]
            X_samples[:, i] = min_val + lhs_samples[:, i] * (max_val - min_val)
        
        # Evaluate simulator at each sample
        kpi_samples = {kpi: [] for kpi in self.target_kpis}
        
        for i in range(n_samples):
            # Create intervention dictionary
            interventions = {var: X_samples[i, j] for j, var in enumerate(self.decision_vars)}
            
            try:
                # Run simulation
                sim_results = simulate_interventions(
                    graph_path=graph_path,
                    model_path=model_path,
                    interventions=interventions,
                    n_samples=100,  # Reduced for speed during surrogate training
                    kpi_nodes=self.target_kpis
                )
                
                # Extract mean values
                for kpi in self.target_kpis:
                    if kpi in sim_results:
                        kpi_samples[kpi].append(sim_results[kpi]["mean"])
                    else:
                        kpi_samples[kpi].append(0.0)
                        
            except Exception as e:
                logger.warning(f"Simulation failed for sample {i}: {e}")
                # Use zero as fallback
                for kpi in self.target_kpis:
                    kpi_samples[kpi].append(0.0)
        
        # Train quadratic models for each KPI
        self.X_train = X_samples
        r2_scores = {}
        
        for kpi in self.target_kpis:
            y = np.array(kpi_samples[kpi])
            self.y_train[kpi] = y
            
            # Create polynomial features (degree 2 for quadratic)
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X_samples)
            
            # Fit linear regression on polynomial features
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Store model and polynomial transformer
            self.models[kpi] = model
            self.poly_features[kpi] = poly
            
            # Calculate R²
            y_pred = model.predict(X_poly)
            r2 = r2_score(y, y_pred)
            r2_scores[kpi] = r2
            
            logger.info(f"KPI {kpi}: R² = {r2:.3f}")
        
        logger.info("Surrogate training completed")
        return r2_scores
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict KPI values using quadratic surrogates.
        
        Args:
            X: Decision variable values (n_samples, n_vars)
            
        Returns:
            Predictions for each KPI
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = {}
        
        for kpi in self.target_kpis:
            if kpi in self.models:
                poly = self.poly_features[kpi]
                model = self.models[kpi]
                
                X_poly = poly.transform(X)
                pred = model.predict(X_poly)
                predictions[kpi] = pred
            else:
                predictions[kpi] = np.zeros(X.shape[0])
        
        return predictions
    
    def get_coefficients(self, kpi: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get quadratic model coefficients for CVXPY optimization.
        
        Returns:
            Tuple of (linear_coeffs, quadratic_matrix) for the KPI
        """
        if kpi not in self.models:
            raise ValueError(f"KPI {kpi} not trained")
        
        model = self.models[kpi]
        poly = self.poly_features[kpi]
        
        # Get feature names to understand coefficient structure
        feature_names = poly.get_feature_names_out([f"x{i}" for i in range(len(self.decision_vars))])
        coeffs = model.coef_
        
        n_vars = len(self.decision_vars)
        
        # Parse coefficients
        # Constant term
        intercept = model.intercept_
        
        # Linear terms (first n_vars coefficients after intercept)
        linear_coeffs = coeffs[1:n_vars+1]
        
        # Quadratic terms
        # For PolynomialFeatures with degree=2, the order is:
        # [1, x0, x1, ..., x0^2, x0*x1, x1^2, ...]
        Q = np.zeros((n_vars, n_vars))
        
        coeff_idx = n_vars + 1  # Start after linear terms
        
        # Diagonal terms (x_i^2)
        for i in range(n_vars):
            Q[i, i] = coeffs[coeff_idx]
            coeff_idx += 1
        
        # Cross terms (x_i * x_j for i < j)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                Q[i, j] = coeffs[coeff_idx] / 2  # Divide by 2 for symmetric matrix
                Q[j, i] = Q[i, j]
                coeff_idx += 1
        
        return linear_coeffs, Q, intercept


def optimize_with_surrogate(
    graph_path: str,
    model_path: str,
    decision_vars: List[str],
    objective: str,  # "max" or "min"
    target_kpi: str,
    constraints: List[Dict[str, Any]] = None,
    decision_bounds: Dict[str, Tuple[float, float]] = None,
    n_surrogate_samples: int = 200
) -> Dict[str, Any]:
    """
    Optimize decision variables using quadratic surrogate and CVXPY.
    
    Args:
        graph_path: Path to causal graph
        model_path: Path to fitted SCM models
        decision_vars: List of decision variable names
        objective: "max" or "min"
        target_kpi: KPI to optimize
        constraints: List of constraint dictionaries
        decision_bounds: Bounds for decision variables
        n_surrogate_samples: Number of samples for surrogate training
        
    Returns:
        Optimization results including optimal decision vector and expected KPI
    """
    logger.info(f"Starting optimization for KPI: {target_kpi}")
    
    # Set default bounds
    if decision_bounds is None:
        decision_bounds = {var: (0.0, 100.0) for var in decision_vars}
    
    # Get all KPIs mentioned in constraints
    constraint_kpis = set()
    if constraints:
        for constraint in constraints:
            if constraint["lhs"] not in decision_vars:
                constraint_kpis.add(constraint["lhs"])
    
    # All KPIs to model
    all_kpis = [target_kpi] + list(constraint_kpis)
    
    # Train surrogate models
    surrogate = QuadraticSurrogate(decision_vars, all_kpis)
    r2_scores = surrogate.train(
        graph_path=graph_path,
        model_path=model_path,
        n_samples=n_surrogate_samples,
        decision_bounds=decision_bounds
    )
    
    # Set up CVXPY optimization
    n_vars = len(decision_vars)
    x = cp.Variable(n_vars)
    
    # Get quadratic model coefficients for target KPI
    linear_coeffs, Q, intercept = surrogate.get_coefficients(target_kpi)
    
    # Objective function: intercept + linear_coeffs.T @ x + 0.5 * x.T @ Q @ x
    if objective == "max":
        objective_expr = cp.Maximize(intercept + linear_coeffs.T @ x + 0.5 * cp.quad_form(x, Q))
    else:  # "min"
        objective_expr = cp.Minimize(intercept + linear_coeffs.T @ x + 0.5 * cp.quad_form(x, Q))
    
    # Set up constraints
    constraints_list = []
    
    # Decision variable bounds
    for i, var in enumerate(decision_vars):
        min_val, max_val = decision_bounds[var]
        constraints_list.append(x[i] >= min_val)
        constraints_list.append(x[i] <= max_val)
    
    # Additional constraints
    if constraints:
        for constraint in constraints:
            lhs_var = constraint["lhs"]
            op = constraint["op"]
            rhs_val = constraint["rhs"]
            
            if lhs_var in decision_vars:
                # Direct constraint on decision variable
                var_idx = decision_vars.index(lhs_var)
                if op == "<=":
                    constraints_list.append(x[var_idx] <= rhs_val)
                elif op == ">=":
                    constraints_list.append(x[var_idx] >= rhs_val)
                elif op == "==":
                    constraints_list.append(x[var_idx] == rhs_val)
                    
            elif lhs_var in all_kpis:
                # Constraint on KPI (using surrogate model)
                kpi_linear, kpi_Q, kpi_intercept = surrogate.get_coefficients(lhs_var)
                kpi_expr = kpi_intercept + kpi_linear.T @ x + 0.5 * cp.quad_form(x, kpi_Q)
                
                if op == "<=":
                    constraints_list.append(kpi_expr <= rhs_val)
                elif op == ">=":
                    constraints_list.append(kpi_expr >= rhs_val)
                elif op == "==":
                    constraints_list.append(kpi_expr == rhs_val)
    
    # Solve optimization problem
    problem = cp.Problem(objective_expr, constraints_list)
    
    try:
        problem.solve(solver=cp.ECOS)  # Good for quadratic problems
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_x = x.value
            optimal_value = problem.value
            
            # Create optimal decision dictionary
            optimal_decisions = {var: optimal_x[i] for i, var in enumerate(decision_vars)}
            
            # Validate with actual simulator
            logger.info("Validating optimization result with simulator")
            validation_results = simulate_interventions(
                graph_path=graph_path,
                model_path=model_path,
                interventions=optimal_decisions,
                n_samples=500,  # More samples for validation
                kpi_nodes=all_kpis
            )
            
            # Prepare results
            result = {
                "status": "success",
                "optimal_decisions": optimal_decisions,
                "expected_kpi": validation_results.get(target_kpi, {}),
                "optimization_info": {
                    "objective": objective,
                    "target_kpi": target_kpi,
                    "cvxpy_status": problem.status,
                    "cvxpy_value": float(optimal_value) if optimal_value is not None else None,
                    "surrogate_r2_scores": r2_scores
                },
                "constraint_validation": {}
            }
            
            # Check constraint satisfaction
            if constraints:
                for constraint in constraints:
                    lhs_var = constraint["lhs"]
                    op = constraint["op"]
                    rhs_val = constraint["rhs"]
                    
                    if lhs_var in validation_results:
                        actual_value = validation_results[lhs_var]["mean"]
                        satisfied = False
                        
                        if op == "<=":
                            satisfied = actual_value <= rhs_val
                        elif op == ">=":
                            satisfied = actual_value >= rhs_val
                        elif op == "==":
                            satisfied = abs(actual_value - rhs_val) < 1e-3
                        
                        result["constraint_validation"][lhs_var] = {
                            "constraint": f"{lhs_var} {op} {rhs_val}",
                            "actual_value": actual_value,
                            "satisfied": satisfied
                        }
            
            logger.info(f"Optimization completed successfully. Target KPI: {validation_results.get(target_kpi, {}).get('mean', 'N/A')}")
            return result
            
        else:
            return {
                "status": "failed",
                "error": f"Optimization problem is {problem.status}",
                "cvxpy_status": problem.status
            }
            
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }