import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pint
from rapidfuzz import fuzz, process


class UnitConverter:
    """Handles unit conversions using pint."""
    
    def __init__(self):
        self.ureg = pint.UnitRegistry()
        
        # Define common unit mappings
        self.unit_mappings = {
            "celsius": "degC",
            "fahrenheit": "degF",
            "kelvin": "kelvin",
            "meters": "meter",
            "kilograms": "kilogram",
            "seconds": "second",
            "percentage": "percent",
            "currency": "USD",  # Default currency
            "count": "dimensionless",
        }
        
        # Define preferred canonical units
        self.canonical_units = {
            "temperature": "degC",
            "length": "meter",
            "mass": "kilogram",
            "time": "second",
            "currency": "USD",
        }
    
    def convert_value(self, value: float, unit_in: str, unit_out: str) -> Optional[float]:
        """Convert a value from one unit to another."""
        try:
            # Map custom unit names to pint units
            pint_unit_in = self.unit_mappings.get(unit_in, unit_in)
            pint_unit_out = self.unit_mappings.get(unit_out, unit_out)
            
            # Handle special cases
            if unit_in == unit_out:
                return value
            
            if pint_unit_in == "dimensionless" or pint_unit_out == "dimensionless":
                return value
            
            # Perform conversion
            quantity = value * self.ureg(pint_unit_in)
            converted = quantity.to(pint_unit_out)
            return float(converted.magnitude)
            
        except Exception as e:
            print(f"Unit conversion error: {value} {unit_in} -> {unit_out}: {e}")
            return None
    
    def convert_series(self, series: pd.Series, unit_in: str, unit_out: str) -> pd.Series:
        """Convert a pandas series from one unit to another."""
        if unit_in == unit_out:
            return series
        
        def convert_func(x):
            if pd.isna(x):
                return x
            return self.convert_value(x, unit_in, unit_out)
        
        return series.apply(convert_func)
    
    def get_canonical_unit(self, unit: str, measurement_type: str = None) -> str:
        """Get the canonical unit for a given unit or measurement type."""
        if measurement_type and measurement_type in self.canonical_units:
            return self.canonical_units[measurement_type]
        
        # Try to infer measurement type from unit
        if unit in ["celsius", "fahrenheit", "kelvin"]:
            return "celsius"
        elif unit in ["meters", "kilometers", "feet", "inches"]:
            return "meters"
        elif unit in ["kilograms", "grams", "pounds"]:
            return "kilograms"
        elif unit in ["seconds", "minutes", "hours"]:
            return "seconds"
        else:
            return unit


class CanonicalMatcher:
    """Handles fuzzy matching of raw column names to canonical names."""
    
    def __init__(self, canonical_dict: Dict[str, Any]):
        self.canonical_dict = canonical_dict
        self.canonical_names = list(canonical_dict.keys())
        
        # Create expanded search space with synonyms
        self.expanded_names = {}
        for canonical_name, config in canonical_dict.items():
            self.expanded_names[canonical_name] = canonical_name
            
            # Add synonyms if available
            if "synonyms" in config:
                for synonym in config["synonyms"]:
                    self.expanded_names[synonym] = canonical_name
    
    def find_best_match(self, raw_name: str, threshold: int = 80) -> Tuple[Optional[str], float]:
        """Find the best matching canonical name for a raw column name."""
        # First try exact match
        if raw_name in self.expanded_names:
            return self.expanded_names[raw_name], 100.0
        
        # Try fuzzy matching
        search_names = list(self.expanded_names.keys())
        
        # Use different matching strategies
        matches = []
        
        # Token sort ratio (good for different word orders)
        token_matches = process.extract(
            raw_name, search_names, scorer=fuzz.token_sort_ratio, limit=3
        )
        matches.extend(token_matches)
        
        # Partial ratio (good for partial matches)
        partial_matches = process.extract(
            raw_name, search_names, scorer=fuzz.partial_ratio, limit=3
        )
        matches.extend(partial_matches)
        
        # Regular ratio
        regular_matches = process.extract(
            raw_name, search_names, scorer=fuzz.ratio, limit=3
        )
        matches.extend(regular_matches)
        
        if not matches:
            return None, 0.0
        
        # Find best match
        best_match = max(matches, key=lambda x: x[1])
        matched_name, score = best_match
        
        if score >= threshold:
            canonical_name = self.expanded_names[matched_name]
            return canonical_name, score
        
        return None, score
    
    def get_canonical_config(self, canonical_name: str) -> Dict[str, Any]:
        """Get configuration for a canonical name."""
        return self.canonical_dict.get(canonical_name, {})


class CanonicalMapper:
    """Main class for mapping raw schema to canonical format."""
    
    def __init__(self, canonical_dict_path: str):
        self.canonical_dict_path = Path(canonical_dict_path)
        self.canonical_dict = self._load_canonical_dict()
        self.matcher = CanonicalMatcher(self.canonical_dict)
        self.unit_converter = UnitConverter()
        
        # Store mapping results
        self.mapping_results = []
        self.human_review_items = []
    
    def _load_canonical_dict(self) -> Dict[str, Any]:
        """Load canonical dictionary from JSON file."""
        try:
            with open(self.canonical_dict_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading canonical dictionary {self.canonical_dict_path}: {e}")
    
    def map_schema(self, raw_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw schema to canonical format."""
        mapping_results = []
        
        for source in raw_schema.get("sources", []):
            source_name = source["source"]
            
            for table in source.get("tables", []):
                table_name = table["name"]
                
                for column in table.get("columns", []):
                    raw_column = column["col"]
                    
                    # Find best canonical match
                    canonical_name, confidence = self.matcher.find_best_match(raw_column)
                    
                    if canonical_name:
                        # Get canonical configuration
                        canonical_config = self.matcher.get_canonical_config(canonical_name)
                        
                        # Handle unit conversion
                        unit_in = column.get("unit", "")
                        unit_out = canonical_config.get("unit", unit_in)
                        
                        # If units are different, try to convert
                        if unit_in and unit_out and unit_in != unit_out:
                            canonical_unit = self.unit_converter.get_canonical_unit(unit_out)
                            unit_out = canonical_unit
                        
                        status = "mapped"
                    else:
                        canonical_name = ""
                        unit_in = column.get("unit", "")
                        unit_out = ""
                        status = "unmapped"
                    
                    # Create mapping result
                    mapping_result = {
                        "source_name": source_name,
                        "raw_table": table_name,
                        "raw_column": raw_column,
                        "canonical_name": canonical_name,
                        "unit_in": unit_in,
                        "unit_out": unit_out,
                        "confidence": confidence,
                        "status": status,
                        "dtype": column.get("dtype", ""),
                    }
                    
                    mapping_results.append(mapping_result)
                    
                    # Flag for human review if confidence is low
                    if confidence > 0 and confidence < 70:
                        self.human_review_items.append(mapping_result)
        
        self.mapping_results = mapping_results
        return {"mappings": mapping_results}
    
    def save_mapping_table(self, output_path: str = "mapping_table.csv") -> None:
        """Save mapping results to CSV file."""
        if not self.mapping_results:
            print("No mapping results to save")
            return
        
        df = pd.DataFrame(self.mapping_results)
        df = df[["raw_table", "raw_column", "canonical_name", "unit_in", "unit_out", "confidence"]]
        df.to_csv(output_path, index=False)
        print(f"Mapping table saved to {output_path}")
    
    def process_raw_data(self, raw_data_dir: str, out_dir: str) -> Dict[str, str]:
        """Process raw data files and create canonical Parquet files."""
        raw_data_path = Path(raw_data_dir)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        canonical_files = {}
        
        # Group mappings by canonical name
        canonical_groups = {}
        for mapping in self.mapping_results:
            if mapping["status"] == "mapped":
                canonical_name = mapping["canonical_name"]
                if canonical_name not in canonical_groups:
                    canonical_groups[canonical_name] = []
                canonical_groups[canonical_name].append(mapping)
        
        # Process each canonical group
        for canonical_name, mappings in canonical_groups.items():
            print(f"Processing canonical name: {canonical_name}")
            
            canonical_data = []
            
            for mapping in mappings:
                # Load raw data
                raw_file_path = self._find_raw_file(raw_data_path, mapping)
                if not raw_file_path:
                    print(f"  Warning: Raw file not found for {mapping['raw_table']}")
                    continue
                
                try:
                    # Load data
                    if raw_file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(raw_file_path)
                    elif raw_file_path.suffix.lower() == '.parquet':
                        df = pd.read_parquet(raw_file_path)
                    else:
                        print(f"  Warning: Unsupported file format {raw_file_path}")
                        continue
                    
                    # Extract and transform column
                    raw_column = mapping["raw_column"]
                    if raw_column not in df.columns:
                        print(f"  Warning: Column {raw_column} not found in {raw_file_path}")
                        continue
                    
                    column_data = df[raw_column].copy()
                    
                    # Apply unit conversion if needed
                    unit_in = mapping["unit_in"]
                    unit_out = mapping["unit_out"]
                    
                    if unit_in and unit_out and unit_in != unit_out:
                        print(f"  Converting {raw_column}: {unit_in} -> {unit_out}")
                        column_data = self.unit_converter.convert_series(column_data, unit_in, unit_out)
                    
                    # Add metadata
                    canonical_row = {
                        "value": column_data,
                        "source_table": mapping["raw_table"],
                        "source_column": raw_column,
                        "confidence": mapping["confidence"],
                    }
                    
                    canonical_data.append(canonical_row)
                    
                except Exception as e:
                    print(f"  Error processing {raw_file_path}: {e}")
                    continue
            
            # Combine all data for this canonical name
            if canonical_data:
                combined_df = self._combine_canonical_data(canonical_data, canonical_name)
                
                # Save to Parquet
                output_file = out_path / f"{canonical_name}.parquet"
                combined_df.to_parquet(output_file, index=False)
                canonical_files[canonical_name] = str(output_file)
                print(f"  Saved {len(combined_df)} rows to {output_file}")
        
        return canonical_files
    
    def _find_raw_file(self, raw_data_path: Path, mapping: Dict[str, Any]) -> Optional[Path]:
        """Find the raw data file for a mapping."""
        table_name = mapping["raw_table"]
        
        # Try common file patterns
        candidates = [
            raw_data_path / f"{table_name}.csv",
            raw_data_path / f"{table_name}.parquet",
            raw_data_path / table_name,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return None
    
    def _combine_canonical_data(self, canonical_data: List[Dict[str, Any]], canonical_name: str) -> pd.DataFrame:
        """Combine multiple sources of canonical data."""
        combined_rows = []
        
        for data_source in canonical_data:
            values = data_source["value"]
            source_table = data_source["source_table"]
            source_column = data_source["source_column"]
            confidence = data_source["confidence"]
            
            for i, value in enumerate(values):
                combined_rows.append({
                    "canonical_name": canonical_name,
                    "value": value,
                    "source_table": source_table,
                    "source_column": source_column,
                    "confidence": confidence,
                    "row_index": i,
                })
        
        return pd.DataFrame(combined_rows)
    
    def get_human_review_items(self) -> List[Dict[str, Any]]:
        """Get items that need human review."""
        return self.human_review_items


def map_to_canonical(
    raw_schema_path: str,
    canonical_dict_path: str,
    raw_data_dir: str,
    out_dir: str
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Main function to map raw schema to canonical format.
    
    Args:
        raw_schema_path: Path to raw_schema.json
        canonical_dict_path: Path to canonical dictionary JSON
        raw_data_dir: Directory containing raw data files
        out_dir: Output directory for canonical Parquet files
    
    Returns:
        Tuple of (canonical_files_dict, human_review_items)
    """
    # Load raw schema
    try:
        with open(raw_schema_path, 'r') as f:
            raw_schema = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading raw schema {raw_schema_path}: {e}")
    
    # Initialize mapper
    mapper = CanonicalMapper(canonical_dict_path)
    
    # Map schema
    print("Mapping raw schema to canonical format...")
    mapping_results = mapper.map_schema(raw_schema)
    
    # Save mapping table
    mapper.save_mapping_table()
    
    # Process raw data
    print("Processing raw data files...")
    canonical_files = mapper.process_raw_data(raw_data_dir, out_dir)
    
    # Get human review items
    human_review_items = mapper.get_human_review_items()
    
    print(f"Mapping complete!")
    print(f"  Canonical files created: {len(canonical_files)}")
    print(f"  Items requiring human review: {len(human_review_items)}")
    
    return canonical_files, human_review_items