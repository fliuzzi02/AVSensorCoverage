import logging
import time

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