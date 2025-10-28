"""
Node Metrics Processor - Runs simulations and collects node metrics data.
modules/node_operations/metrics_pre_processor.py
"""

import os
import yaml
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path


class NodeMetricsProcessor:
    """
    Executes node-level simulations, aggregates metrics,
    and stores the results in a CSV file.
    """

    def __init__(self, fault_template_path: str, output_filename: str = "node_metrics.csv"):
        """
        Initialise the node metrics processor.

        Args:
            fault_template_path: Path to the fault templates YAML file.
            output_filename: Name/path for the output CSV file.

        Raises:
            FileNotFoundError: If the fault template file doesn't exist.
            ValueError: If the file has invalid or unreadable format.
        """
        self.fault_template_path: str = fault_template_path
        self.output_filename: str = output_filename
        self.df: pd.DataFrame = pd.DataFrame()
        self.metrics_collected: int = 0
        self.nodes_processed: int = 0

        self._configure_logger()
        self._validate_fault_template()

    # -------------------------------------------------------------------------
    # Configuration & Validation
    # -------------------------------------------------------------------------

    def _configure_logger(self) -> None:
        """Configure console logger for simulation reporting."""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _validate_fault_template(self) -> None:
        """Validate that the fault template file exists and is readable."""
        if not os.path.exists(self.fault_template_path):
            raise FileNotFoundError(f"Fault template file not found: {self.fault_template_path}")

        if not os.path.isfile(self.fault_template_path):
            raise ValueError(f"Path is not a file: {self.fault_template_path}")

        # Validate YAML format
        try:
            with open(self.fault_template_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            if not isinstance(content, dict):
                raise ValueError("YAML fault template should define a mapping (dict).")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in fault template: {e}") from e

    def _validate_output_directory(self) -> None:
        """Ensure the output directory exists."""
        output_dir = os.path.dirname(self.output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # Simulation and Data Collection
    # -------------------------------------------------------------------------

    def collect_node_metrics(
        self,
        all_node_ids: List[str],
        run_simulation_initialisation: Callable[..., Tuple[Any, ...]],
        healthMetricCalculator: Callable[..., Any],
        default_weights: Dict[str, Any],
        layer_profiles: Dict[str, Any],
        max_values: Dict[str, Any],
        static_thresholds: Dict[str, Any],
        steps: int = 2,
        seed: int = 1
    ) -> "NodeMetricsProcessor":
        """
        Run simulations for all nodes and collect metrics.

        Args:
            all_node_ids: List of node IDs to process.
            run_simulation_initialisation: Function to execute the node simulation.
            healthMetricCalculator: Callable class to compute health metrics.
            default_weights: Default weight configuration.
            layer_profiles: Layer profile configuration.
            max_values: Maximum values configuration.
            static_thresholds: Static threshold configuration.
            steps: Number of simulation steps (default: 2).
            seed: Random seed for reproducibility (default: 1).

        Returns:
            Self for method chaining.
        """
        self._validate_output_directory()

        total_nodes = len(all_node_ids)
        self.logger.info("=" * 60)
        self.logger.info(f"Starting simulation for {total_nodes} nodes")
        self.logger.info(f"Fault template: {self.fault_template_path}")
        self.logger.info(f"Output file: {self.output_filename}")
        self.logger.info("=" * 60)

        for idx, node in enumerate(all_node_ids, 1):
            try:
                self.logger.info(f"Processing node {idx}/{total_nodes}: {node}")

                results, injector, data_returned, history, tendency_data = run_simulation_initialisation(
                    node,
                    default_weights,
                    layer_profiles,
                    max_values,
                    steps=steps,
                    seed=seed,
                    fault_templates=self.fault_template_path
                )

                self.logger.info(f"  ✓ Simulation completed for {node}")

                calculator = healthMetricCalculator(
                    node,
                    tendency_data,
                    default_weights,
                    static_thresholds
                )
                health_metric = calculator.healthMetric()

                dataset = health_metric[1]
                new_df = pd.DataFrame(dataset)

                if self.df.empty:
                    self.df = new_df
                    self.logger.info(f"  ✓ Initialised DataFrame with {len(new_df)} records")
                else:
                    self.df = pd.concat([self.df, new_df], ignore_index=True)
                    self.logger.info(f"  ✓ Added {len(new_df)} records")

                self.metrics_collected += len(new_df)
                self.nodes_processed += 1
                self._save_current_metrics()

            except Exception as e:
                self.logger.error(f"Error processing node {node}: {e}", exc_info=True)
                continue

        return self

    # -------------------------------------------------------------------------
    # Data Handling
    # -------------------------------------------------------------------------

    def _save_current_metrics(self) -> None:
        """Save current metrics to CSV file."""
        self.df.to_csv(self.output_filename, index=False)

    def save_metrics_to_csv(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Save collected metrics to CSV.

        Args:
            filename: Optional output filename (uses default if omitted).

        Returns:
            The DataFrame that was saved.
        """
        output_file = filename or self.output_filename
        self.df.to_csv(output_file, index=False)
        self.logger.info(f"Final metrics saved to: {output_file}")
        return self.df

    def get_metrics_dataset(self) -> pd.DataFrame:
        """Return the collected metrics as a DataFrame."""
        return self.df

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics about the collection process."""
        return {
            "nodes_processed": self.nodes_processed,
            "total_metrics_collected": self.metrics_collected,
            "output_filename": self.output_filename,
            "fault_template_used": self.fault_template_path,
            "dataframe_shape": self.df.shape if not self.df.empty else (0, 0),
            "columns": list(self.df.columns) if not self.df.empty else []
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the collection process."""
        summary = self.get_summary()
        self.logger.info("=" * 60)
        self.logger.info("SIMULATION COLLECTION SUMMARY")
        self.logger.info("=" * 60)
        for k, v in summary.items():
            self.logger.info(f"{k.replace('_', ' ').title()}: {v}")
        self.logger.info("=" * 60)


# Example usage
if 1==2:
    #if __name__ == "__main__":
    """
    Example usage of NodeMetricsProcessor.
    
    Replace the placeholder functions and data with your actual implementations.
    """
    
    # Example configuration (replace with actual data)
    all_node_ids = ['CloudDBServer', 'L1N_01', 'L2N_01', 'L2N_02']
    default_weights = {}  # Your weight configuration
    layer_profiles = {}   # Your layer profiles
    max_values = {}       # Your max values
    static_thresholds = {}  # Your thresholds
    
    # Initialise processor
    processor = NodeMetricsProcessor(
        fault_template_path='data/fault_templates_zero.yaml',
        output_filename='node_metrics.csv'
    )
    
    # Run collection (uncomment when you have the actual functions)
    """
    processor.collect_node_metrics(
        all_node_ids=all_node_ids,
        run_simulation_initialisation=run_simulation_initialisation,
        healthMetricCalculator=healthMetricCalculator,
        default_weights=default_weights,
        layer_profiles=layer_profiles,
        max_values=max_values,
        static_thresholds=static_thresholds,
        steps=2,
        seed=1
    )
    
    # Print summary
    processor.print_summary()
    
    # Get the final dataset
    final_df = processor.get_metrics_dataset()
    print(f"Final dataset has {len(final_df)} rows")
    """
    
    print("NodeMetricsProcessor initialised successfully!")
    print(f"Ready to process nodes and save to: {processor.output_filename}")