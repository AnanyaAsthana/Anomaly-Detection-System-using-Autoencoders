import traci
import warnings
warnings.filterwarnings("ignore")

# Use sumo-gui instead of sumo to see visuals
traci.start(["sumo-gui", "-c", "highway.sumocfg"])

step = 0
bad_driver_id = None

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    current_time = round(step * 0.1, 1)
    active = traci.vehicle.getIDList()

    # Pick bad driver at t=50s
    if current_time >= 50 and bad_driver_id is None:
        if len(active) > 2:
            bad_driver_id = active[1]
            # Paint bad driver RED so you can see it
            traci.vehicle.setColor(bad_driver_id, (255, 0, 0, 255))
            print(f"BAD DRIVER: {bad_driver_id} (shown in RED)")

    # Make bad driver erratic
    if bad_driver_id and bad_driver_id in active:
        if int(step) % 10 == 0:
            traci.vehicle.setSpeed(bad_driver_id, 40.0)
        elif int(step) % 10 == 5:
            traci.vehicle.setSpeed(bad_driver_id, 0.0)

    step += 1

traci.close()
print("Done")