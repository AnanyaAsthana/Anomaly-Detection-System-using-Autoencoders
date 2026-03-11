import traci
import warnings
warnings.filterwarnings("ignore")

traci.start(["sumo", "-c", "highway.sumocfg"])
bad_driver_injected = False
step = 0

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    current_time = round(step * 0.1, 1)

    # Inject at t=10s
    if current_time >= 10 and not bad_driver_injected:
        try:
            traci.vehicle.add("BAD_DRIVER", routeID="route_0", typeID="polite_car")
            bad_driver_injected = True
            print(f"BAD DRIVER injected at t={current_time}s")
        except Exception as e:
            print(f"Injection error: {e}")

    if bad_driver_injected:
        active = traci.vehicle.getIDList()
        if "BAD_DRIVER" in active:
            if int(step) % 10 == 0:
                traci.vehicle.setSpeed("BAD_DRIVER", 40.0)
            elif int(step) % 10 == 5:
                traci.vehicle.setSpeed("BAD_DRIVER", 0.0)
            if step % 50 == 0:
                speed = traci.vehicle.getSpeed("BAD_DRIVER")
                print(f"t={current_time}s | BAD_DRIVER speed={speed:.2f} | active vehicles={len(active)}")
        else:
            print(f"t={current_time}s | BAD_DRIVER LEFT the simulation!")
            break

    if current_time > 200:
        break

    step += 1

traci.close()