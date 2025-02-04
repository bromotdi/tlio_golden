mport os
import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

import glob

import pandas as pd
import numpy as np
from pyproj import Transformer
from geographiclib.geodesic import Geodesic

def llh_to_ecef(latitude, longitude, altitude):
    """Convert latitude, longitude, and altitude to ECEF coordinates."""
    transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
    x, y, z = transformer.transform(longitude, latitude, altitude)
    return x, y, z

def ned_to_ecef(lat0, lon0, h0, north, east, down):
    """Convert NED coordinates to ECEF coordinates."""
    transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
    x0, y0, z0 = transformer.transform(lon0, lat0, h0)

    # Calculate ECEF coordinates from NED
    sin_lat0 = np.sin(np.radians(lat0))
    cos_lat0 = np.cos(np.radians(lat0))
    sin_lon0 = np.sin(np.radians(lon0))
    cos_lon0 = np.cos(np.radians(lon0))

    dx = -sin_lat0 * cos_lon0 * north - sin_lon0 * east - cos_lat0 * cos_lon0 * down
    dy = -sin_lat0 * sin_lon0 * north + cos_lon0 * east - cos_lat0 * sin_lon0 * down
    dz = cos_lat0 * north - sin_lat0 * down

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz
    return x, y, z

def ecef_to_ned(lat0, lon0, h0, x, y, z):
    """Convert ECEF coordinates to NED frame relative to a given origin."""
    transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
    x0, y0, z0 = transformer.transform(lon0, lat0, h0)

    # Calculate NED coordinates from ECEF
    dx = x - x0
    dy = y - y0
    dz = z - z0

    sin_lat0 = np.sin(np.radians(lat0))
    cos_lat0 = np.cos(np.radians(lat0))
    sin_lon0 = np.sin(np.radians(lon0))
    cos_lon0 = np.cos(np.radians(lon0))

    north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    east = -sin_lon0 * dx + cos_lon0 * dy
    down = -cos_lat0 * cos_lon0 * dx - cos_lat0 * sin_lon0 * dy - sin_lat0 * dz
    return north, east, down

def convert_csv_to_odometry(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Define the initial position for conversion (first entry in the CSV)
    initial_latitude = df['latitude'].iloc[0]
    initial_longitude = df['longitude'].iloc[0]
    initial_altitude = df['altitude'].iloc[0]

    # Lists to store the odometry data
    timestamps = []
    x_vals = []
    y_vals = []
    z_vals = []
    qx_vals = []
    qy_vals = []
    qz_vals = []
    qw_vals = []

    # Loop through each row in the CSV
    for index, row in df.iterrows():
        timestamp = row['header_stamp']
        latitude = row['latitude']
        longitude = row['longitude']
        altitude = row['altitude']
        
        if latitude is None or longitude is None or altitude is None:
            continue

        # Convert LLH to ECEF
        x_ecef, y_ecef, z_ecef = llh_to_ecef(latitude, longitude, altitude)

        # Convert ECEF to NED relative to the initial position
        north, east, down = ecef_to_ned(initial_latitude, initial_longitude, initial_altitude, x_ecef, y_ecef, z_ecef)

        # Append the results to the lists
        timestamps.append(timestamp)
        x_vals.append(east)
        y_vals.append(north)
        z_vals.append(-down)  # Converting down to up 

        qx_vals.append(row["quaternion_x"])
        qy_vals.append(row["quaternion_y"])
        qz_vals.append(row["quaternion_z"])
        qw_vals.append(row["quaternion_w"])

    # Create a new DataFrame with the odometry data
    odometry_df = pd.DataFrame({
        'timestamp': timestamps,
        'x': x_vals,
        'y': y_vals,
        'z': z_vals,
        'Qx': qx_vals,
        'Qy': qy_vals,
        'Qz': qz_vals,
        'Qw': qw_vals
    })

    # Write the odometry DataFrame to a CSV file
    odometry_df.to_csv(output_csv, index=False)


class BagToImuAndGps(Node):
    def __init__(self, bag_path, imu_csv_file, gps_csv_file, quat_csv_file):
        super().__init__('bag_to_imu_and_gps')
        self.bag_path = bag_path

        # Set up the reader to use MCAP
        self.reader = SequentialReader()
        storage_options = StorageOptions(uri=self.bag_path, storage_id='mcap')
        converter_options = ConverterOptions('', '')
        self.reader.open(storage_options, converter_options)

        # -- Prepare IMU CSV output --
        self.imu_csv_file = imu_csv_file
        imu_dir = os.path.dirname(self.imu_csv_file)
        if imu_dir and not os.path.exists(imu_dir):
            os.makedirs(imu_dir)
        self.imu_file = open(self.imu_csv_file, 'w')

        # Write IMU CSV header row (no covariance fields)
        self.imu_file.write(
            "header_stamp,"
            "angular_velocity_x,angular_velocity_y,angular_velocity_z,"
            "linear_acceleration_x,linear_acceleration_y,linear_acceleration_z\n"
        )

        # -- Prepare GPS CSV output --
        self.gps_csv_file = gps_csv_file
        gps_dir = os.path.dirname(self.gps_csv_file)
        if gps_dir and not os.path.exists(gps_dir):
            os.makedirs(gps_dir)
        self.gps_file = open(self.gps_csv_file, 'w')

        # Write GPS CSV header row
        self.gps_file.write(
            "header_stamp,"
            "latitude,longitude,altitude\n"
        )

        # --- Prepare Quaternion CSV output ---
        self.quat_csv_file = quat_csv_file
        quat_dir = os.path.dirname(self.quat_csv_file)
        if quat_dir and not os.path.exists(quat_dir):
            os.makedirs(quat_dir)
        self.quat_file = open(self.quat_csv_file, 'w')
        self.quat_file.write(
            "timestamp,"
            "quaternion_x,quaternion_y,quaternion_z,quaternion_w\n"
        )

    def convert(self):
        """Iterate over all messages, saving IMU (/ubuntu/cube_imu/data) and GPS (/ubuntu/gps) data."""

        # 1) Figure out which IMU topic to read by inspecting the bag’s metadata
        topics_and_types = self.reader.get_all_topics_and_types()
        topic_names = [t.name for t in topics_and_types]

        # Decide which IMU topic to use
        if '/ubuntu/cube_imu/data' in topic_names:
            imu_topic = '/ubuntu/cube_imu/data'     # primary
        elif '/ubuntu/icm20948_imu/data_raw' in topic_names:
            imu_topic = '/ubuntu/icm20948_imu/data_raw'  # fallback
        else:
            # If neither is present, we won't parse IMU at all.
            imu_topic = None


        while self.reader.has_next():
            topic, data, t = self.reader.read_next()
            
               # Handle IMU if this is our chosen IMU topic
            if imu_topic and topic == imu_topic:
                msg_type = get_message('sensor_msgs/msg/Imu')
                msg = deserialize_message(data, msg_type)
                self.save_imu(msg)
        
            elif topic == '/ubuntu/gps':
                # Deserialize as sensor_msgs/msg/NavSatFix
                msg_type = get_message('sensor_msgs/msg/NavSatFix')
                msg = deserialize_message(data, msg_type)
                self.save_gps(msg)
            
            elif topic == '/ubuntu/gps/quaternion':
                # Deserialize as geometry_msgs/msg/Quaternion
                msg_type = get_message('geometry_msgs/msg/Quaternion')
                msg = deserialize_message(data, msg_type)
                self.save_quaternion(msg, t)

    def save_imu(self, msg):
        """Write IMU fields (excluding covariances) to CSV."""
        hdr = msg.header
        stamp_sec = hdr.stamp.sec
        stamp_nsec = hdr.stamp.nanosec

        # Create one CSV line (without covariance)
        line = (
            f"{stamp_sec}.{stamp_nsec},"
            f"{msg.angular_velocity.x},{msg.angular_velocity.y},{msg.angular_velocity.z},"
            f"{msg.linear_acceleration.x},{msg.linear_acceleration.y},{msg.linear_acceleration.z}\n"
        )
        self.imu_file.write(line)

    def save_gps(self, msg):
        """Write NavSatFix fields to CSV."""
        hdr = msg.header
        stamp_sec = hdr.stamp.sec
        stamp_nsec = hdr.stamp.nanosec

        line = (
            f"{stamp_sec}.{stamp_nsec},"
            f"{msg.latitude},{msg.longitude},{msg.altitude}\n"
        )
        self.gps_file.write(line)

    def save_quaternion(self, msg, bag_time):
        # Convert bag_time_ns (nanoseconds) to seconds if you like:
        line = (
            f"{bag_time},"
            f"{msg.x},{msg.y},{msg.z},{msg.w}\n"
        )
        self.quat_file.write(line)

    def __del__(self):
        # Close the files if they were opened
        if hasattr(self, 'imu_file') and self.imu_file:
            self.imu_file.close()
        if hasattr(self, 'gps_file') and self.gps_file:
            self.gps_file.close()
        if hasattr(self, 'quat_file') and self.quat_file:
            self.quat_file.close()


def merge_lat_and_quat(gps_csv_file, quat_csv_file, lat_data_with_quat):
    df_lat  = pd.read_csv(gps_csv_file)
    df_quat = pd.read_csv(quat_csv_file)

    # Усечём оба DataFrame до одинаковой длины, если они не совпадают
    min_len = min(len(df_lat), len(df_quat))
    df_lat  = df_lat.iloc[:min_len].reset_index(drop=True)
    df_quat = df_quat.iloc[:min_len].reset_index(drop=True)

    # Добавляем колонки из df_quat
    df_lat["quaternion_x"] = df_quat["quaternion_x"]
    df_lat["quaternion_y"] = df_quat["quaternion_y"]
    df_lat["quaternion_z"] = df_quat["quaternion_z"]
    df_lat["quaternion_w"] = df_quat["quaternion_w"]

    # Сохраняем результат
    df_lat.to_csv(lat_data_with_quat, index=False)


def merge_gps_and_imu(
    gps_data_csv: str,
    imu_data_csv: str,
    merged_output_csv: str
):

    df_gps = pd.read_csv(gps_data_csv)
    df_imu = pd.read_csv(imu_data_csv)

    # Rename 'header_stamp' to 'timestamp' in the IMU data for consistent merging
    imu_data_renamed = df_imu.rename(columns={'header_stamp': 'timestamp'})

    # Merging the dataframes on the 'timestamp' column using the closest timestamp
    merged_data = pd.merge_asof(df_gps.sort_values('timestamp'), imu_data_renamed.sort_values('timestamp'), on='timestamp', direction='nearest')

    merged_data.to_csv(merged_output_csv, index=False)


def process_single_bag(bag_path: str, output_folder: str):
    """
    Processes a single .mcap bag file:
      1. Reads data into IMU, GPS, and quaternion CSVs
      2. Merges GPS & quaternion data
      3. Converts lat-lon-alt to NED
      4. Merges resulting GPS data with IMU data
    """
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define all CSV output inside this folder
    imu_csv_file        = os.path.join(output_folder, "imu_data.csv")
    gps_csv_file        = os.path.join(output_folder, "lat_data.csv")
    quat_csv_file       = os.path.join(output_folder, "quat_data.csv")
    lat_data_with_quat  = os.path.join(output_folder, "lat_data_with_quat.csv")
    final_gps_odometry  = os.path.join(output_folder, "gps_data.csv")
    merged_gps_imu_file = os.path.join(output_folder, "gps_imu_merged.csv")

    # 1) Initialize and read from bag
    rclpy.init()
    node = BagToImuAndGps(bag_path, imu_csv_file, gps_csv_file, quat_csv_file)
    node.convert()
    node.destroy_node()
    rclpy.shutdown()

    # 2) Merge lat_data.csv & quat_data.csv → lat_data_with_quat.csv
    merge_lat_and_quat(gps_csv_file, quat_csv_file, lat_data_with_quat)

    # 3) Convert lat_data_with_quat → NED (gps_data.csv)
    convert_csv_to_odometry(lat_data_with_quat, final_gps_odometry)

    # 4) Merge gps_data.csv & imu_data.csv → gps_imu_merged.csv
    merge_gps_and_imu(final_gps_odometry, imu_csv_file, merged_gps_imu_file)
    
    print(f"Done processing bag: {bag_path}")
    print(f"Created files:\n  {imu_csv_file}\n  {gps_csv_file}\n  {quat_csv_file}\n  {lat_data_with_quat}\n  {final_gps_odometry}\n  {merged_gps_imu_file}")



def process_all_bags_in_folder(input_folder: str, output_root: str):
    """
    For each .mcap bag file found in input_folder, create an output subfolder
    and run the pipeline.
    
    :param input_folder: Directory containing .mcap files
    :param output_root:  Where to place the converted_data/ subfolders
    """
    # Make sure the folder for all outputs exists
    os.makedirs(output_root, exist_ok=True)

    # Find all .mcap bag files in 'input_folder'
    bag_files = sorted(glob.glob(os.path.join(input_folder, '*.mcap')))
    if not bag_files:
        print(f"No .mcap files found in {input_folder}")
        return
    
    print(f"Found {len(bag_files)} .mcap files in {input_folder}")

    for bag_path in bag_files:
        # Example naming: 
        #   if bag_path = "/home/ubuntu/dataset/run1.bag.mcap",
        #   bag_name = "run1.bag"
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]
        
        # Create a subfolder inside output_root specifically for this bag
        # e.g. "output_root/run1.bag_converted_data/"
        output_folder = os.path.join(output_root, f"{bag_name}_converted_data")
        os.makedirs(output_folder, exist_ok=True)

        # Now process the single bag
        process_single_bag(bag_path, output_folder)



def main():
    """
    Example main function that processes multiple input folders,
    each containing multiple .mcap files.
    """
    # List of folders that contain .mcap files
    input_folders = [
        "/home/ubuntu/rnin-vio/tali_dataset/1",
        "/home/ubuntu/rnin-vio/tali_dataset/2",
        "/home/ubuntu/rnin-vio/tali_dataset/3",
        "/home/ubuntu/rnin-vio/tali_dataset/3rd",
        "/home/ubuntu/rnin-vio/tali_dataset/4",
        "/home/ubuntu/rnin-vio/tali_dataset/5",
        "/home/ubuntu/rnin-vio/tali_dataset/6",
        "/home/ubuntu/rnin-vio/tali_dataset/7",
        "/home/ubuntu/rnin-vio/tali_dataset/8",
        "/home/ubuntu/rnin-vio/tali_dataset/gps7_dji_ar0234_two_triangles_360",
        "/home/ubuntu/rnin-vio/tali_dataset/gps8_dji_jetson_two_wierd_squares_17_12_180"
    ]
    
    # Where to store the final outputs, maybe a common root.
    # Inside this, we will create subfolders for each input folder,
    # then sub-subfolders for each bag.
    output_root = "/home/ubuntu/rnin-vio/ALL_CONVERTED"
    
    for folder in input_folders:
        # e.g. /home/ubuntu/rnin-vio/tali_dataset/1 -> folder_name = "1"
        folder_name = os.path.basename(folder)
        folder_output = os.path.join(output_root, folder_name)

        print(f"\n=== Processing folder: {folder} ===")
        process_all_bags_in_folder(folder, folder_output)
        print(f"=== Done folder: {folder} ===")

if __name__ == '__main__':
    main()
