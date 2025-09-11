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

#fault_templates = 'fault_injection/fault_templates_zero.yaml',
#fault_templates = 'fault_injection/fault_templates.yaml',
fault_templates = 'fault_injection/templates.yaml'
try:
        # Run simulation
        results, injector, data_returned = run_complete_simulation(baseline_values, max_values, steps=111, seed=10, fault_templates=fault_templates)
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {len(results)}")
        print(f"Fault periods: {sum(1 for r in results if r['any_fault_active'])}")
        print("data_returned: ", data_returned)
        #print(f"results[0]: ", results[0]['cpu'] )
        if isinstance(results[0], dict) and 'cpu' in results[0]:
            print(results[0]['cpu'], results[0]['rtt'], results[0]['plr']  )
except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
