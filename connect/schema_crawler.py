import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pandas.api.types as pd_types
import psycopg2
import yaml
from sqlalchemy import create_engine, text


class UnitInferencer:
    """Infers physical units from column names using regex patterns."""
    
    UNIT_PATTERNS = {
        "celsius": [
            r"(temp|temperature).*c$",
            r".*temp.*celsius",
            r".*_c$",
            r"degrees?_c",
        ],
        "fahrenheit": [
            r"(temp|temperature).*f$",
            r".*temp.*fahrenheit",
            r".*_f$",
            r"degrees?_f",
        ],
        "kelvin": [
            r"(temp|temperature).*k$",
            r".*temp.*kelvin",
            r".*_k$",
            r"degrees?_k",
        ],
        "percentage": [
            r".*pct$",
            r".*percent.*",
            r".*percentage.*",
            r".*_pct$",
            r".*rate$",
            r".*ratio$",
        ],
        "currency": [
            r".*price.*",
            r".*cost.*",
            r".*amount.*",
            r".*revenue.*",
            r".*sales.*",
            r".*usd$",
            r".*eur$",
            r".*gbp$",
            r".*dollar.*",
        ],
        "meters": [
            r".*length.*",
            r".*height.*",
            r".*width.*",
            r".*distance.*",
            r".*_m$",
            r".*meter.*",
        ],
        "kilograms": [
            r".*weight.*",
            r".*mass.*",
            r".*_kg$",
            r".*kilogram.*",
        ],
        "seconds": [
            r".*duration.*",
            r".*time.*",
            r".*_s$",
            r".*second.*",
        ],
        "count": [
            r".*count.*",
            r".*num.*",
            r".*quantity.*",
            r".*qty.*",
            r".*total.*",
        ],
    }
    
    def infer_unit(self, column_name: str) -> Optional[str]:
        """Infer physical unit from column name."""
        column_lower = column_name.lower()
        
        for unit, patterns in self.UNIT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, column_lower):
                    return unit
        
        return None


class DataSourceConnector:
    """Handles connections to different data sources."""
    
    def __init__(self, source_config: Dict[str, Any]):
        self.source_config = source_config
        self.source_type = source_config["type"]
        self.source_name = source_config["name"]
    
    def get_tables(self) -> List[str]:
        """Get list of tables/files from the data source."""
        if self.source_type == "csv":
            return self.source_config.get("files", [])
        elif self.source_type == "postgres":
            return self._get_postgres_tables()
        elif self.source_type == "sqlite":
            return self._get_sqlite_tables()
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
    
    def get_sample_data(self, table_name: str, limit: int = 1000) -> pd.DataFrame:
        """Get sample data from a table/file."""
        if self.source_type == "csv":
            return self._get_csv_sample(table_name, limit)
        elif self.source_type == "postgres":
            return self._get_postgres_sample(table_name, limit)
        elif self.source_type == "sqlite":
            return self._get_sqlite_sample(table_name, limit)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
    
    def _get_csv_sample(self, file_path: str, limit: int) -> pd.DataFrame:
        """Get sample from CSV file."""
        try:
            return pd.read_csv(file_path, nrows=limit)
        except Exception as e:
            print(f"Error reading CSV {file_path}: {e}")
            return pd.DataFrame()
    
    def _get_postgres_tables(self) -> List[str]:
        """Get list of tables from PostgreSQL database."""
        conn_str = self._build_postgres_connection_string()
        engine = create_engine(conn_str)
        
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result]
    
    def _get_postgres_sample(self, table_name: str, limit: int) -> pd.DataFrame:
        """Get sample from PostgreSQL table."""
        conn_str = self._build_postgres_connection_string()
        engine = create_engine(conn_str)
        
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        try:
            return pd.read_sql(query, engine)
        except Exception as e:
            print(f"Error reading PostgreSQL table {table_name}: {e}")
            return pd.DataFrame()
    
    def _build_postgres_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        config = self.source_config
        return (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}"
        )
    
    def _get_sqlite_tables(self) -> List[str]:
        """Get list of tables from SQLite database."""
        db_path = self.source_config["path"]
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tables
        except Exception as e:
            print(f"Error connecting to SQLite {db_path}: {e}")
            return []
    
    def _get_sqlite_sample(self, table_name: str, limit: int) -> pd.DataFrame:
        """Get sample from SQLite table."""
        db_path = self.source_config["path"]
        
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error reading SQLite table {table_name}: {e}")
            return pd.DataFrame()


class ColumnProfiler:
    """Profiles columns and infers data types and statistics."""
    
    def __init__(self):
        self.unit_inferencer = UnitInferencer()
    
    def profile_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Profile a single column."""
        profile = {
            "col": column_name,
            "dtype": self._infer_dtype(series),
            "null_pct": round(series.isnull().sum() / len(series) * 100, 2),
        }
        
        # Add statistics based on data type
        if pd_types.is_numeric_dtype(series):
            profile.update(self._profile_numeric(series))
        elif pd_types.is_datetime64_any_dtype(series):
            profile.update(self._profile_datetime(series))
        elif pd_types.is_string_dtype(series) or pd_types.is_object_dtype(series):
            profile.update(self._profile_categorical(series))
        
        # Infer physical unit
        unit = self.unit_inferencer.infer_unit(column_name)
        if unit:
            profile["unit"] = unit
        
        return profile
    
    def _infer_dtype(self, series: pd.Series) -> str:
        """Infer data type from pandas series."""
        if pd_types.is_integer_dtype(series):
            return "integer"
        elif pd_types.is_float_dtype(series):
            return "float"
        elif pd_types.is_bool_dtype(series):
            return "boolean"
        elif pd_types.is_datetime64_any_dtype(series):
            return "timestamp"
        elif pd_types.is_string_dtype(series):
            return "string"
        elif pd_types.is_object_dtype(series):
            # Try to infer if it's actually numeric or datetime
            if series.dropna().empty:
                return "string"
            
            # Try to convert to numeric
            try:
                pd.to_numeric(series.dropna().iloc[:100])
                return "numeric"
            except (ValueError, TypeError):
                pass
            
            # Try to convert to datetime
            try:
                pd.to_datetime(series.dropna().iloc[:100])
                return "timestamp"
            except (ValueError, TypeError):
                pass
            
            return "string"
        else:
            return "unknown"
    
    def _profile_numeric(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column."""
        valid_data = series.dropna()
        if valid_data.empty:
            return {}
        
        return {
            "min": float(valid_data.min()),
            "max": float(valid_data.max()),
            "mean": float(valid_data.mean()),
            "std": float(valid_data.std()) if len(valid_data) > 1 else 0.0,
        }
    
    def _profile_datetime(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column."""
        valid_data = series.dropna()
        if valid_data.empty:
            return {}
        
        return {
            "min": valid_data.min().isoformat(),
            "max": valid_data.max().isoformat(),
        }
    
    def _profile_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Profile categorical/string column."""
        valid_data = series.dropna()
        if valid_data.empty:
            return {}
        
        unique_count = valid_data.nunique()
        total_count = len(valid_data)
        
        profile = {
            "unique_count": unique_count,
            "cardinality": round(unique_count / total_count, 4),
        }
        
        # Add top values if reasonable cardinality
        if unique_count <= 50:
            top_values = valid_data.value_counts().head(10)
            profile["top_values"] = top_values.to_dict()
        
        return profile


class SchemaCrawler:
    """Main schema crawler that orchestrates the profiling process."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.profiler = ColumnProfiler()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading config file {self.config_path}: {e}")
    
    def crawl(self) -> Dict[str, Any]:
        """Crawl all sources and generate schema profile."""
        schema_profile = {
            "sources": []
        }
        
        for source_config in self.config.get("sources", []):
            print(f"Processing source: {source_config['name']}")
            source_profile = self._crawl_source(source_config)
            schema_profile["sources"].append(source_profile)
        
        return schema_profile
    
    def _crawl_source(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Crawl a single data source."""
        connector = DataSourceConnector(source_config)
        tables = connector.get_tables()
        
        source_profile = {
            "source": source_config["name"],
            "type": source_config["type"],
            "tables": []
        }
        
        for table_name in tables:
            print(f"  Processing table/file: {table_name}")
            table_profile = self._crawl_table(connector, table_name)
            if table_profile:
                source_profile["tables"].append(table_profile)
        
        return source_profile
    
    def _crawl_table(self, connector: DataSourceConnector, table_name: str) -> Optional[Dict[str, Any]]:
        """Crawl a single table/file."""
        try:
            # Get sample data
            sample_data = connector.get_sample_data(table_name, limit=1000)
            
            if sample_data.empty:
                print(f"    Warning: No data found in {table_name}")
                return None
            
            # Profile each column
            columns = []
            for column_name in sample_data.columns:
                column_profile = self.profiler.profile_column(
                    sample_data[column_name], column_name
                )
                columns.append(column_profile)
            
            return {
                "name": table_name,
                "row_count": len(sample_data),
                "columns": columns
            }
        
        except Exception as e:
            print(f"    Error processing {table_name}: {e}")
            return None
    
    def save_profile(self, profile: Dict[str, Any], output_path: str = "raw_schema.json") -> None:
        """Save schema profile to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            print(f"Schema profile saved to {output_path}")
        except Exception as e:
            print(f"Error saving profile: {e}")


def crawl_schema(config_path: str, output_path: str = "raw_schema.json") -> Dict[str, Any]:
    """Crawl schema from configuration file and return profile."""
    crawler = SchemaCrawler(config_path)
    profile = crawler.crawl()
    crawler.save_profile(profile, output_path)
    return profile


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) != 2:
        print("Usage: python -m connect.schema_crawler <config.yml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        crawler = SchemaCrawler(config_path)
        profile = crawler.crawl()
        crawler.save_profile(profile)
        print("Schema crawling completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()