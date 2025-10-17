import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

class NodeMetricsProcessor:
    """
    A class to process node metrics CSV data and retrieve metrics by node_id.
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the processor with a CSV file path.
        
        Args:
            csv_file_path (str): Path to the node_metrics.csv file
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.node_ids = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load and preprocess the CSV data."""
        try:
            # Read CSV file
            self.df = pd.read_csv(self.csv_file_path)
            
            # Convert timestamp to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Sort by timestamp to ensure chronological order
            self.df = self.df.sort_values('timestamp')
            
            # Get unique node IDs
            self.node_ids = self.df['node_id'].unique().tolist()
            
            print(f"Successfully loaded data for {len(self.node_ids)} nodes")
            print(f"Time range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            print(f"Available nodes: {self.node_ids}")
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
    
    def get_node_metrics(self, node_id: str) -> pd.DataFrame:
        """
        Get all metrics for a specific node.
        
        Args:
            node_id (str): The node identifier
            
        Returns:
            pd.DataFrame: DataFrame containing all metrics for the node
        """
        if node_id not in self.node_ids:
            raise ValueError(f"Node ID '{node_id}' not found. Available nodes: {self.node_ids}")
        
        return self.df[self.df['node_id'] == node_id].copy()
    
    def get_metric_timeseries(self, node_id: str, metric: str) -> Tuple[List[datetime], List[float]]:
        """
        Get a specific metric timeseries for a node.
        
        Args:
            node_id (str): The node identifier
            metric (str): Metric name ('plr', 'cpu', 'rtt')
            
        Returns:
            Tuple[List[datetime], List[float]]: Timestamps and metric values
        """
        valid_metrics = ['plr', 'cpu', 'rtt']
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of {valid_metrics}")
        
        node_data = self.get_node_metrics(node_id)
        timestamps = node_data['timestamp'].tolist()
        values = node_data[metric].tolist()
        
        return timestamps, values
    
    def get_latest_metrics(self, node_id: str) -> Dict[str, float]:
        """
        Get the latest metrics for a specific node.
        
        Args:
            node_id (str): The node identifier
            
        Returns:
            Dict[str, float]: Latest metric values
        """
        node_data = self.get_node_metrics(node_id)
        latest = node_data.iloc[-1]  # Get last row (latest timestamp)
        
        return {
            'plr': latest['plr'],
            'cpu': latest['cpu'],
            'rtt': latest['rtt'],
            'timestamp': latest['timestamp']
        }
    
    def get_metric_statistics(self, node_id: str, metric: str) -> Dict[str, float]:
        """
        Calculate statistics for a specific metric of a node.
        
        Args:
            node_id (str): The node identifier
            metric (str): Metric name ('plr', 'cpu', 'rtt')
            
        Returns:
            Dict[str, float]: Statistical summary
        """
        node_data = self.get_node_metrics(node_id)
        values = node_data[metric].values
        
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def get_all_nodes_latest_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get latest metrics for all nodes.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping node_id to latest metrics
        """
        latest_metrics = {}
        for node_id in self.node_ids:
            latest_metrics[node_id] = self.get_latest_metrics(node_id)
        
        return latest_metrics
    
    def get_time_range_metrics(self, start_time: datetime, end_time: datetime, 
                             node_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get metrics within a specific time range.
        
        Args:
            start_time (datetime): Start of time range
            end_time (datetime): End of time range
            node_id (str, optional): Specific node ID. If None, returns all nodes.
            
        Returns:
            pd.DataFrame: Filtered metrics data
        """
        mask = (self.df['timestamp'] >= start_time) & (self.df['timestamp'] <= end_time)
        filtered_df = self.df[mask]
        
        if node_id:
            filtered_df = filtered_df[filtered_df['node_id'] == node_id]
        
        return filtered_df
    
    def get_metric_correlation(self, node_id: str) -> pd.DataFrame:
        """
        Calculate correlation matrix between PLR, CPU, and RTT for a node.
        
        Args:
            node_id (str): The node identifier
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        node_data = self.get_node_metrics(node_id)
        metrics = node_data[['plr', 'cpu', 'rtt']]
        return metrics.corr()
    
    def detect_anomalies(self, node_id: str, metric: str, threshold_std: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalies in a metric based on standard deviation.
        
        Args:
            node_id (str): The node identifier
            metric (str): Metric name ('plr', 'cpu', 'rtt')
            threshold_std (float): Standard deviation threshold for anomalies
            
        Returns:
            pd.DataFrame: Rows containing anomalies
        """
        node_data = self.get_node_metrics(node_id)
        values = node_data[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Calculate z-scores
        z_scores = np.abs((values - mean_val) / std_val)
        
        # Find anomalies
        anomalies = node_data[z_scores > threshold_std].copy()
        anomalies['z_score'] = z_scores[z_scores > threshold_std]
        
        return anomalies
    
    def export_node_metrics(self, node_id: str, output_file: str) -> None:
        """
        Export metrics for a specific node to a CSV file.
        
        Args:
            node_id (str): The node identifier
            output_file (str): Output file path
        """
        node_data = self.get_node_metrics(node_id)
        node_data.to_csv(output_file, index=False)
        print(f"Exported metrics for node '{node_id}' to {output_file}")
    
    def plot_metric_timeseries(self, node_id: str, metrics: List[str] = None, 
                             save_path: Optional[str] = None) -> None:
        """
        Plot timeseries of metrics for a node (requires matplotlib).
        
        Args:
            node_id (str): The node identifier
            metrics (List[str]): List of metrics to plot. If None, plots all.
            save_path (str, optional): Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if metrics is None:
                metrics = ['plr', 'cpu', 'rtt']
            
            node_data = self.get_node_metrics(node_id)
            
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3*len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                axes[i].plot(node_data['timestamp'], node_data[metric], marker='o', markersize=2)
                axes[i].set_title(f'{node_id} - {metric.upper()} over time')
                axes[i].set_ylabel(metric.upper())
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")

# Example usage and demonstration
def demonstrate_processor():
    """Demonstrate the NodeMetricsProcessor class functionality."""
    
    # Initialize processor
    processor = NodeMetricsProcessor('node_metrics.csv')
    
    print("\n" + "="*50)
    print("NODE METRICS PROCESSOR DEMONSTRATION")
    print("="*50)
    
    # 1. Show available nodes
    print(f"\n1. Available nodes: {processor.node_ids}")
    
    # 2. Get latest metrics for each node
    print("\n2. Latest metrics for all nodes:")
    latest_metrics = processor.get_all_nodes_latest_metrics()
    for node_id, metrics in latest_metrics.items():
        print(f"   {node_id}: PLR={metrics['plr']:.4f}, CPU={metrics['cpu']:.4f}, RTT={metrics['rtt']:.4f}")
    
    # 3. Get detailed metrics for a specific node
    sample_node = processor.node_ids[0] if processor.node_ids else 'CloudDBServer'
    print(f"\n3. Detailed analysis for node '{sample_node}':")
    
    # Get timeseries data
    timestamps, plr_values = processor.get_metric_timeseries(sample_node, 'plr')
    print(f"   - Collected {len(plr_values)} data points")
    
    # Get statistics
    plr_stats = processor.get_metric_statistics(sample_node, 'plr')
    cpu_stats = processor.get_metric_statistics(sample_node, 'cpu')
    rtt_stats = processor.get_metric_statistics(sample_node, 'rtt')
    
    print(f"   - PLR stats: mean={plr_stats['mean']:.4f}, std={plr_stats['std']:.4f}")
    print(f"   - CPU stats: mean={cpu_stats['mean']:.4f}, std={cpu_stats['std']:.4f}")
    print(f"   - RTT stats: mean={rtt_stats['mean']:.4f}, std={rtt_stats['std']:.4f}")
    
    # 4. Get correlation matrix
    correlation_matrix = processor.get_metric_correlation(sample_node)
    print(f"\n4. Correlation matrix for '{sample_node}':")
    print(correlation_matrix)
    
    # 5. Detect anomalies
    anomalies = processor.detect_anomalies(sample_node, 'plr', threshold_std=2.0)
    print(f"\n5. Anomalies detected in PLR for '{sample_node}': {len(anomalies)}")
    
    return processor
