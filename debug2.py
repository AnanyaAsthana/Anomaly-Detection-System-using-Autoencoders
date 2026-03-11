import traci
import warnings
warnings.filterwarnings("ignore")

traci.start(["sumo", "-c", "highway.sumocfg"])
step = 0
bad_driver_id = None

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    current_time = round(step * 0.1, 1)
    active = traci.vehicle.getIDList()

    # Pick first available car as bad driver at t=20s
    if current_time >= 20 and bad_driver_id is None and len(active) > 0:
        bad_driver_id = active[0]
        print(f"t={current_time}s | Picked '{bad_driver_id}' as BAD DRIVER")

    # Make it erratic
    if bad_driver_id and bad_driver_id in active:
        if int(step) % 10 == 0:
            traci.vehicle.setSpeed(bad_driver_id, 40.0)
        elif int(step) % 10 == 5:
            traci.vehicle.setSpeed(bad_driver_id, 0.0)
        
        if step % 20 == 0:
            speed = traci.vehicle.getSpeed(bad_driver_id)
            print(f"t={current_time}s | BAD_DRIVER ({bad_driver_id}) speed={speed:.2f}")
    elif bad_driver_id and bad_driver_id not in active:
        print(f"t={current_time}s | BAD_DRIVER left, picking new one...")
        if len(active) > 0:
            bad_driver_id = active[0]
            print(f"t={current_time}s | New BAD_DRIVER: {bad_driver_id}")

    if current_time > 150:
        break
    step += 1

traci.close()
print("Done")