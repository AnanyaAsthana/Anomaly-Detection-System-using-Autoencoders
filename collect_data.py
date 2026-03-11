import traci
import csv
import os

SUMO_BINARY = "sumo"
CONFIG_FILE = "highway.sumocfg"

def collect_normal_data(output_csv="normal_driving_data.csv"):
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "vehicle_id", "speed", "acceleration", "lane_position", "lane_index"])
        
        prev_speeds = {}
        step = 0
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            current_time = round(step * 0.1, 1)
            
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                lane_pos = traci.vehicle.getLanePosition(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                
                prev_speed = prev_speeds.get(veh_id, speed)
                acceleration = (speed - prev_speed) / 0.1
                prev_speeds[veh_id] = speed
                
                writer.writerow([current_time, veh_id,
                                  round(speed, 3),
                                  round(acceleration, 3),
                                  round(lane_pos, 3),
                                  lane_index])
            step += 1
    
    traci.close()
    print(f"Done! Data saved to {output_csv}")
    print(f"Total rows: {step}")

collect_normal_data()