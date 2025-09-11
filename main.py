#main.py
from fault_injection.fault_injection_sim import run_complete_simulation


#baseline_values = {'cpu': 0.2, 'rtt': 100, 'plr': 0.01}
#baseline_values = {'cpu': 0.2, 'rtt': 30, 'plr': 0.01}
#baseline_values = {"cpu":0.145, "rtt":26.4, "plr":0.0084}
#baseline_values = {"cpu":0.090, "rtt":22.8, "plr":0.0068}
#max_values = {'cpu': 1.0, 'rtt': 500, 'plr': 0.5}  # Maximum reasonable values

#baseline_values = {'cpu': 0.2, 'rtt': 100, 'plr': 0.01}
#baseline_values = {'cpu': 0.2, 'rtt': 30, 'plr': 0.01}
#baseline_values = {"cpu":0.145, "rtt":26.4, "plr":0.0084}

baseline_values = {"cpu":0.090, "rtt":22.8, "plr":0.0068}
max_values = {'cpu': 0.75, 'rtt': 150, 'plr': 0.5}

try:
        # Run simulation
        results, injector, data_returned = run_complete_simulation(baseline_values, max_values, steps=50, seed=10)
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {len(results)}")
        print(f"Fault periods: {sum(1 for r in results if r['any_fault_active'])}")
        print("data_returned: ", data_returned)
        
except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
