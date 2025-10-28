

"""
Node Transceive Processor - Adds transceive_count column to node metrics data.
modules/transceive/transceive_processor.py
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional
import pandas as pd


class NodeTransceivePreProcessor:
    """
    Processes node metrics CSV file and adds transceive_count column.
    """
    
    def __init__(self ,input_file: str, n=0):
        """
        Initialize the processor with input CSV file.
        
        Args:
            input_file: Path to the input CSV file
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file is empty or has invalid format
        """
        self.input_file = input_file
        self.data: List[Dict] = []
        self.columns: List[str] = []
        
        # Run validation tests
        self._validate_input_file()
        self._load_data()
    
    def _validate_input_file(self) -> None:
        """Validate that the input file exists and is readable."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        if not os.path.isfile(self.input_file):
            raise ValueError(f"Path is not a file: {self.input_file}")
        
        if os.path.getsize(self.input_file) == 0:
            raise ValueError(f"Input file is empty: {self.input_file}")
    
    def _load_data(self) -> None:
        """Load and validate CSV data."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.columns = reader.fieldnames
                
                if not self.columns:
                    raise ValueError("CSV file has no headers")
                
                # Required columns check
                required_cols = ['timestamp', 'node_id', 'plr', 'cpu', 'rtt', 
                               'weighted_health_score', 'health_threshold']
                missing_cols = [col for col in required_cols if col not in self.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                self.data = list(reader)
                
                if not self.data:
                    raise ValueError("CSV file has no data rows")
                    
        except csv.Error as e:
            raise ValueError(f"Error reading CSV file: {e}")
    
    def transceive_initialisation(self, default_value: int ) -> 'NodeTransceivePreProcessor':
        """
        Add transceive_count column to the data.
        
        Args:
            default_value: Default value for transceive_count (default: 0)
        
        Returns:
            Self for method chaining
        """
        if 'transceive_count' not in self.columns:
            for item in ['transceive_count', "plr_mean", "plr_std", "rtt_mean", "rtt_std", "cpu_mean", "cpu_std", "plr_x-x^", "rtt_x-x^", "cpu_x-x^"]:
                self.columns.append(item)
        
        for row in self.data:
            row['transceive_count'] = str(default_value)
            row['plr_mean'] = row['plr']
            row['rtt_mean'] = row['rtt']
            row['cpu_mean'] = row['cpu']
            row['plr_std'] = '0.0'
            row['rtt_std'] = '0.0'
            row['cpu_std'] = '0.0' 
            row['plr_x-x^'] = '0.0'
            row['rtt_x-x^'] = '0.0'
            row['cpu_x-x^'] = '0.0' 

        return self
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert data to JSON string.
        
        Args:
            indent: JSON indentation level
        
        Returns:
            JSON string representation of the data
        """

        #print("self.data:, ", self.data); time.sleep(200)
        return json.dumps(self.data, indent=indent)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame.
        
        Returns:
            DataFrame containing the processed data
        """
        return pd.DataFrame(self.data)
    
    def to_csv(self, output_file: str) -> str:
        """
        Save data to CSV file.
        
        Args:
            output_file: Path to output CSV file
        
        Returns:
            Path to the saved file
        """
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.data)
        
        return output_file
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics about the processed data.
        
        Returns:
            Dictionary containing summary information
        """
        unique_nodes = len(set(row['node_id'] for row in self.data))
        
        return {
            'total_rows': len(self.data),
            'total_relevant_info': len(self.columns),
            'unique_nodes': unique_nodes,
            'columns': self.columns,
            'sample_row': self.data[0] if self.data else None
        }

