import os
import shutil
from xml.dom import minidom

from traffic_generator import TrafficGenerator, Scenario
from sim_config import CONFIG_4WAY_160M

class SumoSimulation:
    def __init__(self, libsumo, sim_config, sim_step, episode_duration, log_folder, rank=0, enable_measure=False):
        self.libsumo = libsumo
        self.sim_config = sim_config
        self.rank = rank
        self.sim_step = sim_step
        self.episode_duration = episode_duration
        self.log_folder = log_folder
        self.measure_enabled = enable_measure

        self.template_xml_path = "sumo_xml_template_files"
        self.workspace_path = os.path.join("sumo_workspace", f"env_{self.rank}")
        self.sumo_config_path = os.path.join(
            self.workspace_path, 
            CONFIG_4WAY_160M.name, 
            CONFIG_4WAY_160M.name + ".sumocfg"
        )
        self._setup_workspace()
        
        self.active_vehicles = set()
        self.vehicle_dict = dict()
        self.traffic_gen = TrafficGenerator(self.sim_config, sim_step)

    def _setup_workspace(self):
        if os.path.exists(self.workspace_path):
            shutil.rmtree(self.workspace_path, ignore_errors=True)
        shutil.copytree(self.template_xml_path, self.workspace_path)

    def _generateVehicleTypesXML(self, vehicle_dict, output_folder):
        rootXML = minidom.Document()
        routes = rootXML.createElement('routes')
        rootXML.appendChild(routes)

        for v in vehicle_dict.values():
            vtype = rootXML.createElement('vType')
            vtype.setAttribute('id', 'vtype-'+v.vehicleID)
            vtype.setAttribute('length', str(v.length))
            vtype.setAttribute('mass', str(v.weight))
            vtype.setAttribute('maxSpeed', str(v.maxSpeed))
            vtype.setAttribute('accel', str(v.acceleration))
            vtype.setAttribute('decel', str(v.brakingAcceleration))
            vtype.setAttribute('emergencyDecel', str(v.fullBrakingAcceleration))
            vtype.setAttribute('minGap', str(v.minGap))
            vtype.setAttribute('tau', str(v.driverProfile.tau))
            vtype.setAttribute('sigma', str(v.driverProfile.sigma))
            vtype.setAttribute('speedFactor', str(v.driverProfile.speedLimitComplianceFactor))
            vtype.setAttribute('vClass', str(v.vClass))
            vtype.setAttribute('emissionClass', str(v.emissionClass))
            vtype.setAttribute('color', str(v.color))
            vtype.setAttribute('guiShape', str(v.shape))
            routes.appendChild(vtype)

        output_path = os.path.join(output_folder, "vehicletypes.rou.xml")
        with open(output_path, 'w') as fd:
            fd.write(rootXML.toprettyxml(indent="    "))

    def _log_scenario(self, log_folder, episode_index, vehicle_num, scenario):
        os.makedirs(log_folder, exist_ok=True)
        episode_info_file = os.path.join(log_folder, f"episode_info_ep{episode_index}.txt")

        episode_info = (
            f"==================================================\n"
            f" EPISODE INFO\n"
            f"==================================================\n"
            f" Episode ID  : {episode_index}\n"
            f" Scenario Type  : {scenario}\n"
            f" Total Vehicles : {vehicle_num}\n"
            f"==================================================\n"
        )

        with open(episode_info_file, 'w') as f:
            f.write(episode_info)

    def _startSumo(self, episode_index):
        try:
            self.libsumo.close()
        except:
            pass

        sumo_log_file = os.path.join(self.log_folder, f"sumo_output_ep{episode_index}.txt")
        self.libsumo.start([
            "sumo", 
            "-c", self.sumo_config_path, 
            "--waiting-time-memory", "3600", 
            "--start", 
            "--quit-on-end", 
            "--step-length", str(self.sim_step),
            "--log", sumo_log_file,
            "--time-to-teleport", "-1"
        ])

    def _addVehiclesToSimulation(self, vehicle_dict):
        for v in vehicle_dict.values():
            self.libsumo.vehicle.add(vehID=v.vehicleID, routeID=v.routeID, typeID='vtype-'+v.vehicleID, depart=v.depart, departSpeed=v.initialSpeed, departLane=v.departLane)

    def _resolve_scenario(self, scenario):
        if scenario is None:
            raise ValueError("Scenario must be provided ")
        if isinstance(scenario, Scenario):
            return scenario
        if isinstance(scenario, str):
            return Scenario[scenario.upper()]
        raise TypeError("Unsupported scenario type")

    def initialize_episode(self, episode_id, scenario, n_vehicles):
        self.active_vehicles = set()

        resolved_scenario = self._resolve_scenario(scenario)
        vehicle_dict, vehicle_num, scenario_used = self.traffic_gen.generate_traffic(
            episode_id,
            resolved_scenario,
            n_vehicles,
        )
        self.vehicle_dict = vehicle_dict
        self._generateVehicleTypesXML(self.vehicle_dict, output_folder=self.workspace_path)
        self._log_scenario(self.log_folder, episode_id, vehicle_num, scenario_used)

        self._startSumo(episode_id)
        self._addVehiclesToSimulation(self.vehicle_dict)
        self.libsumo.trafficlight.setProgram(self.sim_config.tl_id, self.sim_config.tl_program)

        for v in self.vehicle_dict.values():
            v.resetMeasures()

    def simulation_step(self):
        self.libsumo.simulationStep()

        if self.measure_enabled:
            self.active_vehicles.update(self.libsumo.simulation.getDepartedIDList())
            self.active_vehicles.difference_update(self.libsumo.simulation.getArrivedIDList())

            for vehicle in self.active_vehicles:
                self.vehicle_dict[vehicle].doMeasures()

    def get_measures(self):
        mesaured_vehicle_data = []
        for v in self.vehicle_dict.values():
            data = {
                "vehicleID": v.vehicleID, 
                "totalDistance": v.totalDistance, 
                "totalTravelTime": v.totalTravelTime, 
                "totalWaitingTime": v.totalWaitingTime, 
                "meanSpeed": v.meanSpeed, 
                "totalCO2Emissions": v.totalCO2Emissions, 
                "totalCOEmissions": v.totalCOEmissions, 
                "totalHCEmissions": v.totalHCEmissions, 
                "totalPMxEmissions": v.totalPMxEmissions, 
                "totalNOxEmissions": v.totalNOxEmissions, 
                "totalFuelConsumption": v.totalFuelConsumption, 
                "totalElectricityConsumption": v.totalElectricityConsumption, 
                "totalNoiseEmission": v.totalNoiseEmission
            }
            mesaured_vehicle_data.append(data)
        return mesaured_vehicle_data

    def close(self):
        try:
            self.libsumo.close()
        except:
            pass

