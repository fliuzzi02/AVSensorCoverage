import logging
import pickle
import time
import pyvista as pv

from args import args
from environment.grid import Grid
from environment.slice import Slice
from plotting.report import create_report
from plotting.plots import create_plots
from plotting.plot_helpers import metrics, setup_plot_args, output_folder
from sensors.sensor_helpers import load_sensorset
from sensors.sensor_set import SensorSet
from utils.gui import GUI

# PROGRAM OPTIONS
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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
    logging.info("Grid coverage calculated -> preparing report and plots")

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


def run(args):
    logging.info("Starting Programm")

    if args.gui_mode:
        gui_instance = GUI()
        gui_instance.run()
        args.update(gui_instance.get_inputs())
    
    sensor_set = SensorSet(load_sensorset(args.sensor_setup))
    logging.info("Sensor set loaded -> now setting sensor pose")

    vehicle = pv.read(args.vehicle_path).triangulate_contours().clean()
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
    logging.info("Grid created -> starting single sensor coverage calculation")

    # Create de feasible area

    feasible_area = pv.Box(bounds=[-0.5, 0.5, -0.5, 0.5, 1, 1.5]).triangulate().clean()


    # Plot the feasible area and the car so that i can deactivate the visualization of the car and see the feasible area
    # This is useful to see if the feasible area is correct
    plotter = pv.Plotter()
    plotter.add_mesh(vehicle, color="red", show_edges=True, edge_color="black")
    plotter.add_mesh(feasible_area, color="green", show_edges=True, edge_color="black")
    plotter.show()


    # calculate the coverage of each sensor in the grid
    sensor_set.calculate_coverage(grid, vehicle)
    logging.info("Finished single sensor calculation -> calculating grid coverage")

    # Combine the data of all sensors in the grid
    grid.combine_data(sensor_set.get_sensors())
    logging.info("Grid coverage calculated -> Value: " + str(grid.get_coverage()))

    # Initialize the population

    # Do the rest of the visualization stuff...
    # plot(args, grid, vehicle, sensor_set)
    


if __name__ == "__main__":
    start_time = time.time()
    logging.info("Starting main")
    run(args)
    logging.info("Success")
    logging.info(f"Results saved: {output_folder(args.save_path, args.folder_name)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time is {elapsed_time}")
