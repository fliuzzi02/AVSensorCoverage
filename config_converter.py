## This code converts a sensor configuration file from one format to another.
## Available conversions:
## - Scenario JSON to yaml
## - yaml to Scenario JSON
import json
import yaml
import os
import math

def quaternion_to_euler(obj):
    q_x = obj.get("orientation", {}).get("e_x")
    q_y = obj.get("orientation", {}).get("e_y")
    q_z = obj.get("orientation", {}).get("e_z")
    q_w = obj.get("orientation", {}).get("e_0")

    # Roll (x-axis rotation)
    roll = math.atan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x**2 + q_y**2))
    
    # Pitch (y-axis rotation)
    pitch = math.asin(2 * (q_w * q_y - q_z * q_x))
    
    # Yaw (z-axis rotation)
    yaw = math.atan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2))

    # Convert radians to degrees
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    
    return roll, pitch, yaw

def extract_common_parameters(sensor):
    roll, pitch, yaw = quaternion_to_euler(sensor)
    return {
        "name": sensor.get("name"),
        "type": sensor.get("description"),
        "uuid": sensor.get("uuid"),
        "position": {
            "x": sensor.get("location", {}).get("x"),
            "y": sensor.get("location", {}).get("y"),
            "z": sensor.get("location", {}).get("z")
        },
        "orientation": {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        },
        "update_hz": sensor.get("update_hz")
    }

def convert_scenario_json_to_yaml(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    actors = data.get("actors", [])
    vehicles = actors.get("vehicles", [])
    # Select vehicle with name "ego"
    ego_vehicle = next((v for v in vehicles if v.get("name") == "ego"), None)
    sensors = ego_vehicle.get("sensors", [])
    cameras = sensors.get("cams", [])
    lidars = sensors.get("pcs", [])
    radars = sensors.get("radars", [])

    yaml_data = {
        "lidars": [],
        "cameras": [],
        "radars": [],
    }

    # Convert cameras to YAML format
    for camera in cameras:
        camera_data = extract_common_parameters(camera)
        camera_data.update({
            "fov": camera.get("fov_deg"),
            "max_dist": 100,
            "min_dist": 0.1,
            "n_pixels": {
                "width": camera.get("width_px"),
                "height": camera.get("height_px")
            },
            "encoding": camera.get("encoding"),
            "lens_dirt": camera.get("lens_dirt"),
        })
        yaml_data["cameras"].append(camera_data)

    # Convert lidars to YAML format
    for lidar in lidars:
        lidar_data = extract_common_parameters(lidar)
        lidar_data.update({
            "pps": lidar.get("pps"),
            "rps": lidar.get("rps"),
            "fov_h": 360 if lidar.get("full_range") else 120,
            "fov_v": abs(lidar.get("pitches_deg", [])[0])*2 if lidar.get("pitches_deg") else 60,
            "detection_range": lidar.get("max_dist"),
            "min_range": 0.1
        })
        yaml_data["lidars"].append(lidar_data)

    # Convert radars to YAML format
    for radar in radars:
        radar_data = extract_common_parameters(radar)
        radar_data.update({
            "fov_h": radar.get("horizontal_fov_deg"),
            "fov_v": radar.get("vertical_fov_deg"),
            "detection_range": 60,
            "min_range": 0.1,
            "source": radar.get("source"),
        })
        yaml_data["radars"].append(radar_data)

    # Write the YAML data to the output file with tab indentation
    with open(output_file, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, indent=2)

if __name__ == "__main__":
    input_file = "test/neighborhood.json"
    output_file = "test/scenario.yaml"

    if os.path.exists(input_file):
        convert_scenario_json_to_yaml(input_file, output_file)
        print(f"Converted {input_file} to {output_file}")
    else:
        print(f"{input_file} does not exist.")