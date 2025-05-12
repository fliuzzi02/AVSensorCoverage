import logging
import time
import yaml

# This class represents a set of sensors
class SensorSet:
    def __init__(self, sensors):
        self.sensors = sensors

    def get_sensors(self):
        return self.sensors
    
    def calculate_coverage(self, grid, vehicle):
        start_time = time.time()
        ix = 1
        max_ix = len(self.sensors)
        for sensor in self.sensors:
            logging.info(f"Calculating Single Sensor {ix} of {max_ix}")
            sensor.calculate_coverage(grid, vehicle)
            ix += 1
        logging.info(f"Sensor coverage calculated in {time.time() - start_time} seconds")

    def print(self):
        for sensor in self.sensors:
            sensor.print()

    def copy(self):
        return SensorSet([sensor.copy() for sensor in self.sensors])
    
    def to_yaml(self):
        """
        Convert the sensor set to a yaml representation.
        :return: yaml representation of the sensor set
        """
        return {
            "lidars": [sensor.to_yaml() for sensor in self.sensors if sensor.type == "lidar"],
            "radars": [sensor.to_yaml() for sensor in self.sensors if sensor.type == "radar"],
            "cameras": [sensor.to_yaml() for sensor in self.sensors if sensor.type == "camera"],
        }
    
    def save(self, path):
        """
        Save the sensor set characteristics to a yaml file in indented format.
        :param path: path to the yaml file
        """
        with open(path, 'w') as file:
            yaml.dump(self.to_yaml(), file, default_flow_style=False)