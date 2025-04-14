import logging

# This class represents a set of sensors
class SensorSet:
    def __init__(self, sensors):
        self.sensors = sensors

    def get_sensors(self):
        return self.sensors
    
    def calculate_coverage(self, grid, vehicle):
        ix = 1
        max_ix = len(self.sensors)
        for sensor in self.sensors:
            logging.info(f"Calculating Single Sensor {ix} of {max_ix}")
            sensor.calculate_coverage(grid, vehicle)
            ix += 1

    def print(self):
        for sensor in self.sensors:
            sensor.print()