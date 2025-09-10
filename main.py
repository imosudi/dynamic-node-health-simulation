from fault_injection.fault_injection_sim import run_complete_simulation

#baseline_values = {'cpu': 0.2, 'rtt': 100, 'plr': 0.01}
#baseline_values = {'cpu': 0.2, 'rtt': 30, 'plr': 0.01}
#baseline_values = {"cpu":0.145, "rtt":26.4, "plr":0.0084}

baseline_values = {"cpu":0.090, "rtt":22.8, "plr":0.0068}

try:
        # Run simulation
        results, injector = run_complete_simulation(baseline_values, steps=1, seed=10)
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {len(results)}")
        print(f"Fault periods: {sum(1 for r in results if r['any_fault_active'])}")
        
except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()