import logging
import time
import pyvista as pv
import numpy as np
import cma

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

# PROGRAM OPTIONS
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def plot(args, grid, vehicle, sensor_set):
    logging.info("Starting plotting")
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

def get_points_inside_mesh(mesh, resolution=100):
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
    selected = point_set.select_enclosed_points(mesh, progress_bar=True)
    inside_points = selected.points[selected.point_data["SelectedPoints"] == 1]

    # Return the inside points as a numpy array
    return inside_points

def my_objective_function(x, sensor_set, grid, vehicle):
    """
    Objective function to be minimized. This function should return a single value
    representing the fitness of the solution.

    :param x: The input parameters for the objective function. They are interpreted as the sensor poses. [x1, y1, z1, roll1, pitch1, yaw1, ...].
    :param sensor_set: The sensor set to be evaluated.
    :param grid: The grid to be used for coverage calculation.
    :return: The fitness value.
    """
    # Unpack the input parameters
    # Position sensors in 3D space
    i = 0
    for sensor in sensor_set.get_sensors():
        sensor.set_pose(x[i], x[i+1], x[i+2], x[i+3], x[i+4], x[i+5])
        i += 1

    # Calculate the fitness value based on the sensor positions
    # This is just a placeholder; replace with actual fitness calculation
    sensor_set.calculate_coverage(grid, vehicle)
    grid.combine_data(sensor_set.get_sensors())
    fitness_value = grid.get_coverage()

    return fitness_value

def run(args):
    logging.info("Starting Programm")

    if args.gui_mode:
        gui_instance = GUI()
        gui_instance.run()
        args.update(gui_instance.get_inputs())
    
    prototype_sensor_set = SensorSet(load_sensorset(args.sensor_setup))
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

    # Create the feasible area
    feasible_area = pv.Box(bounds=[0, 1.5, -0.75, 0.75, 1.25, 1.75]).triangulate().clean()
    feasible_area = feasible_area.boolean_difference(vehicle).triangulate().clean()
    feasible_positions = get_points_inside_mesh(feasible_area, resolution=100)
    logging.info(f"Number of points inside the feasible area: {len(feasible_positions)}")
    logging.info("Feasible area calculated -> initating optimization algorithm")

    ### This is where the optimization algorithm begins
    population_size = 1
    population = []
    fitness_values = []
    
    # Create a random population of sensor sets
    for _ in range(population_size):
        # Create a new sensor set with the same sensors but different positions
        new_sensor_set = prototype_sensor_set.copy()
        for sensor in new_sensor_set.get_sensors():
            # Randomly select a point inside the feasible area
            random_point = feasible_positions[np.random.choice(feasible_positions.shape[0])]
            sensor.set_pose(random_point[0], random_point[1], random_point[2])
        population.append(new_sensor_set)

    # Calculate the coverage of all sensor sets in the population
    start_time = time.time()
    for i, sensor_set in enumerate(population):
        sensor_set.calculate_coverage(grid, vehicle)
        grid.combine_data(sensor_set.get_sensors())
        fitness_values.append(grid.get_coverage())
        logging.info(f"Sensor set {i + 1}/{population_size} coverage: {fitness_values[i]}")
    logging.info(f"Coverage calculation took {time.time() - start_time} seconds")
    logging.info("Fitness values: " + str(fitness_values))

    # Test saving the population
    population[0].save("test/pickle.yaml")

    # Do the rest of the visualization stuff...
    # plot(args, grid, vehicle, population[0])
    


if __name__ == "__main__":
    start_time = time.time()
    logging.info("Starting main")
    run(args)
    logging.info("Success")
    logging.info(f"Results saved: {output_folder(args.save_path, args.folder_name)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time is {elapsed_time}")
