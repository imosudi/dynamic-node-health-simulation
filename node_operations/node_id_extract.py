# node_operations/node_id_extract.py
import csv
from typing import List, Optional, Union
from pathlib import Path

import pandas as pd

def extract_node_ids(file_path: Union[str, Path], 
                    delimiter: str = 'auto', 
                    node_id_column: str = 'node_id',
                    return_count: bool = False) -> Union[List[str], tuple]:
   
   
    
    try:
        # Convert to Path object for better path handling
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect delimiter if needed
        if delimiter == 'auto':
            # Read first line to detect delimiter
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' in first_line:
                    delimiter = '\t'
                elif ',' in first_line:
                    delimiter = ','
                elif ';' in first_line:
                    delimiter = ';'
                else:
                    delimiter = ','  # Default fallback
        
        # Method 1: Using pandas (recommended for larger files)
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            # Check if node_id column exists
            if node_id_column not in df.columns:
                raise ValueError(f"Column '{node_id_column}' not found in CSV. Available columns: {list(df.columns)}")
            
            # Extract node IDs and remove any NaN values
            node_ids = df[node_id_column].dropna().astype(str).tolist()
            
        except ImportError:
            # Fallback method using built-in csv module if pandas not available
            node_ids = []
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                # Check if node_id column exists
                if node_id_column not in reader.fieldnames:
                    raise ValueError(f"Column '{node_id_column}' not found in CSV. Available columns: {reader.fieldnames}")
                
                for row in reader:
                    node_id = row.get(node_id_column, '').strip()
                    if node_id:  # Skip empty values
                        node_ids.append(node_id)
        
        # Return based on return_count parameter
        if return_count:
            return node_ids, len(node_ids)
        else:
            return node_ids
            
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        raise

