import logging
import time
import pyvista as pv
import numpy as np
import cma
import cma.constraints_handler as ch
import matplotlib.pyplot as plt

from args import args
from environment.grid import Grid
from environment.slice import Slice
from plotting.report import create_report
from plotting.plots import create_plots
from plotting.plot_helpers import metrics, setup_plot_args, output_folder
from sensors.sensor_helpers import load_sensorset
from sensors.sensor_set import SensorSet
from utils.gui import GUI
from tqdm import tqdm

# TODO: Add sensors list not to optimize, so the fixed positions

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

## GLOBAL VARIABLES
xMin = 0
xMax = 1.5
yMin = -0.65
yMax = 0.65
zMin = 1.5
zMax = 1.8
pitchMin = -10
pitchMax = 10
rollMin = -10
rollMax = 10
yawMin = -180
yawMax = 180

population_size = 20
max_iterations = 400

variables_per_sensor = 4

prototype_sensor_set, grid, vehicle, feasible_positions, sensors_per_set = None, None, None, None, None

## LOGGING OPTIONS
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO, handlers=[TqdmLoggingHandler()])
progress = tqdm(total=population_size*max_iterations, desc="CMA-ES", dynamic_ncols=True, leave=True)

def plot(args, grid, vehicle, sensor_set):
    logging.info("Starting plotting")
    sensor_set.calculate_coverage(grid, vehicle)
    grid.combine_data(sensor_set.get_sensors())

    # Calculate the metrics for the grid
    grid.set_metrics_no_condition()
    grid.set_metrics_condition(
        n1=args.conditions.N1,
        n2=args.conditions.N2,
        n6=args.conditions.N6,
        n7=args.conditions.N7,
        n8=args.conditions.N8,
    )

    # Here i can edit the height of the slices that will be analyzed and plotted
    slices = [
        Slice(grid, 0, normal="x"),
        Slice(grid, 0, normal="y"),
        Slice(grid, 0.5, normal="z"),
    ]
    slices.extend(
        [
            Slice(grid, i * args.slice.distance)
            for i in range(1, args.slice.number, 1)
        ]
    )

    logging.info("creating plots")
    create_plots(
        grid,
        sensor_set.get_sensors(),
        vehicle,
        args.save_path,
        args.folder_name,
        slices[0],
        slices[1],
        slices[2],
    )
    logging.info("Report created -> finished")

def plot_feasible_area(feasible_area, vehicle):
    """
    Plot the feasible area and the vehicle mesh.

    :param feasible_area: The feasible area mesh (PyVista PolyData).
    :param vehicle: The vehicle mesh (PyVista PolyData).
    """
    plotter = pv.Plotter()
    plotter.add_mesh(feasible_area, color="blue", opacity=0.5, show_edges=True)
    plotter.add_mesh(vehicle, color="red", opacity=0.5, show_edges=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

def get_points_inside_mesh(mesh, resolution):
    """
    Get all points inside a PyVista mesh.

    :param mesh: The PyVista mesh (PolyData) to check.
    :param resolution: Number of points along each axis in the bounding box.
    :return: A numpy array of points inside the mesh.
    """
    # Get the bounding box of the mesh
    bounds = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]

    # Create a grid of points within the bounding box
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    z = np.linspace(bounds[4], bounds[5], resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]  # Flatten into (N, 3)

    # Create a PointSet from the grid points
    point_set = pv.PointSet(grid_points)

    # Check which points are inside the mesh
    selected = point_set.select_enclosed_points(mesh)
    inside_points = selected.points[selected.point_data["SelectedPoints"] == 1]

    # Return the inside points as a numpy array
    return inside_points

def calculate_feasible_positions(feasible_area, resolution=100):
    """
    Calculate feasible positions for the sensors based on the vehicle and the feasible area.
    It subtracts the vehicle mesh from the feasible area and returns a set of discrete points
    inside the feasible area.
    :param feasible_area: The feasible area mesh (PyVista PolyData).
    :param vehicle: The vehicle mesh (PyVista PolyData).
    :param resolution: The resolution for the grid of points inside the feasible area.
    :return: A numpy array of feasible positions (x, y, z).
    """
    feasible_area = feasible_area.boolean_difference(vehicle).triangulate().clean()
    feasible_positions = get_points_inside_mesh(feasible_area, resolution)
    logging.info(f"Number of points inside the feasible area: {len(feasible_positions)}")
    return feasible_positions

def is_feasible(x):
    """
    Check if the given sensor positions are feasible.

    :param x: The input parameters for the objective function. They are interpreted as the sensor poses. [x1, y1, z1, roll1, pitch1, yaw1, ...].
    :param feasible_positions: The feasible positions to check against.
    :return: True if the positions are feasible, False otherwise.
    """
    # Check if all sensor positions are within the feasible area
    for i in range(0, len(x), variables_per_sensor):
        position = np.array([x[i], x[i+1], x[i+2]])
        if not np.any(np.all(np.isclose(feasible_positions, position), axis=1)):
            # TODO: This is always returning True, need to implement the actual check
            return True
    return True

def scale_solution(x):
    """
    Scale the solution to the range of the feasible area.
    :param x: The input parameters for the objective function. They are interpreted as the sensor poses. [x1, y1, z1, roll1, pitch1, yaw1, ...].
    :return: The scaled solution.
    """
    # Scale the solution to the range of the feasible area
    x_scaled = np.copy(x)
    for i in range(0, len(x), variables_per_sensor):
        # Rescale the x, y, z coordinates to the feasible area
        x_scaled[i] = x[i] * (xMax - xMin) + xMin
        x_scaled[i+1] = x[i+1] * (yMax - yMin) + yMin
        x_scaled[i+2] = x[i+2] * (zMax - zMin) + zMin
        # Rescale the pitch, yaw, roll angles to the feasible range
        # x_scaled[i+3] = x[i+3] * (pitchMax - pitchMin) + pitchMin
        x_scaled[i+3] = x[i+3] * (yawMax - yawMin) + yawMin
        # x_scaled[i+5] = x[i+5] * (rollMax - rollMin) + rollMin
    return x_scaled

def my_objective_function(x):
    """
    Objective function to be minimized. This function should return a single value
    representing the fitness of the solution.

    :param x: The input parameters for the objective function. They are interpreted as the sensor poses. [x1, y1, z1, pitch1, yaw1, roll1, ...].
    :return: The fitness value.
    """
    # Copy the input x to avoid modifying the original
    x_test = scale_solution(x)

    # Position sensors in 3D space
    i = 0
    sensor_set = prototype_sensor_set.copy()
    for sensor in sensor_set.get_sensors():
        if sensor.get_name() == "Camera_F":
            # Skip the front sensor
            continue
        sensor.set_pose(x_test[i], x_test[i+1], x_test[i+2], 0, x_test[i+3], 0)
        i += variables_per_sensor

    # Calculate the fitness value based on the sensor positions
    # This is just a placeholder; replace with actual fitness calculation
    sensor_set.calculate_coverage(grid, vehicle)
    grid.combine_data(sensor_set.get_sensors())
    fitness_value = grid.get_coverage()

    progress.update(1)

    return 1-fitness_value

def run(args):
    global prototype_sensor_set, grid, vehicle, feasible_positions, sensors_per_set

    logging.info("Starting Programm")
    plt.ion()

    if args.gui_mode:
        gui_instance = GUI()
        gui_instance.run()
        args.update(gui_instance.get_inputs())
    
    prototype_sensor_set = SensorSet(load_sensorset(args.sensor_setup))
    # TODO: Here i am optimizing all the sensors except the one in the front
    sensors_per_set = len(prototype_sensor_set.get_sensors()) - 1
    logging.info("Sensor set loaded -> now setting sensor pose")

    vehicle = pv.read(args.vehicle_path).triangulate().clean()
    logging.info("Vehicle loaded -> creating grid")

    grid = Grid(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        dim_z=args.dim_z,
        spacing=args.spacing,
        advanced=args.advanced,
        car=vehicle,
        center=args.origin,
        dist=args.nearfield_dist,
    )
    logging.info("Grid created -> Calculating feasible area")

    # Create the feasible positions
    feasible_area = pv.Box(bounds=[xMin, xMax, yMin, yMax, zMin, zMax]).triangulate().clean()
    # Plot the feasible area and the vehicle
    plot_feasible_area(feasible_area, vehicle)
    # feasible_positions = calculate_feasible_positions(feasible_area)
    logging.info("Feasible area calculated -> Initating optimization algorithm")

    ### This is where the optimization algorithm begins
    x0 = [0.5] * variables_per_sensor * sensors_per_set
    sigma0 = 0.9

    # Define CMA options
    options = {
        'bounds': [0, 1],
        'maxiter': max_iterations,
        'popsize': population_size,
        'verb_disp': 0
    }

    result = cma.fmin(
        my_objective_function,
        x0,
        sigma0,
        options=options
    )
    logging.info("Optimization finished -> Saving results")

    # Access final result
    best_solution = result[0]
    best_fitness = result[1]

    # Apply the best solution to the sensor set
    sensor_set = prototype_sensor_set.copy()
    best_solution = scale_solution(best_solution)
    i = 0
    for sensor in sensor_set.get_sensors():
        if sensor.get_name() == "Camera_F":
            # Skip the front sensor
            continue
        sensor.set_pose(best_solution[i], best_solution[i+1], best_solution[i+2], 0, best_solution[i+3], 0)
        i += variables_per_sensor
    
    logging.info(f"Best solution: {str(best_solution)}")
    logging.info(f"Best fitness: {str(best_fitness)}")

    # Save the sensor set to a file
    sensor_set.save("test/optimizedSensors.yaml")

    # Plot cma results
    cma.plot()
    plt.show(block=True)

    # Do the rest of the visualization stuff...
    plot(args, grid, vehicle, sensor_set)
    
if __name__ == "__main__":
    start_time = time.time()
    logging.info("Starting main")
    run(args)
    logging.info("Success")   
    # logging.info(f"Results saved: {output_folder(args.save_path, args.folder_name)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time is {elapsed_time}")
