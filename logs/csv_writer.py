import csv


def write_detailed_csv(data, filename='health_monitoring_detailed.csv'):
    """
    Alternative version with more detailed breakdown
    """
    headers = [
        'step', 'baseline_cpu', 'baseline_rtt', 'baseline_plr',
        'observed_cpu', 'observed_rtt', 'observed_plr',
        'cpu_diff', 'rtt_diff', 'plr_diff',
        'active_faults_count', 'health_metric', 'adaptive_threshold', 'health_status'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for record in data:
            baseline = record['baseline']
            observed = record['observed']
            
            row = [
                record['step'],
                baseline['cpu'], baseline['rtt'], baseline['plr'],
                float(observed[0]), float(observed[1]), float(observed[2]),
                float(observed[0]) - baseline['cpu'],  # cpu difference
                float(observed[1]) - baseline['rtt'],  # rtt difference  
                float(observed[2]) - baseline['plr'],  # plr difference
                len(record['active_faults']),
                float(record['health_metric']),
                float(record['adaptive_threshold']),
                record['health_status']
            ]
            writer.writerow(row)
    
    print(f"Detailed data written to {filename}")

# write_detailed_csv(data)