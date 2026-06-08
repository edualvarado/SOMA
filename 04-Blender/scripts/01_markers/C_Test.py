import json
from pathlib import Path

def copy_marker_values(source_file_path, destination_file_path, output_file_path):
    """
    Copies marker coordinate values from a source JSON file to a destination JSON file,
    assuming the markers are in the same order in both files, and saves the result
    to a new output JSON file.

    Args:
        source_file_path (str or Path): Path to the source JSON file (e.g., with "marker_0" keys).
        destination_file_path (str or Path): Path to the destination JSON file (e.g., with "marker_459_0_0" keys).
        output_file_path (str or Path): Path where the modified JSON data will be saved.
    """
    source_file_path = Path(source_file_path)
    destination_file_path = Path(destination_file_path)
    output_file_path = Path(output_file_path)

    if not source_file_path.exists():
        print(f"Error: Source file not found at {source_file_path}")
        return
    if not destination_file_path.exists():
        print(f"Error: Destination file not found at {destination_file_path}")
        return

    # Load source data
    with open(source_file_path, 'r') as f:
        source_data = json.load(f)

    # Load destination data (this will be modified and saved to output_file_path)
    with open(destination_file_path, 'r') as f:
        destination_data = json.load(f)

    # Ensure both files have the expected structure (a "0" key)
    if "0" not in source_data or "0" not in destination_data:
        print("Error: Both JSON files must contain a top-level key '0'.")
        return

    source_markers = source_data["0"]
    destination_markers = destination_data["0"]

    # Extract ordered values from the source
    # We assume the order of items() is consistent for this operation
    source_values = [value for key, value in source_markers.items()]
    
    # Extract ordered keys from the destination
    destination_keys = [key for key, value in destination_markers.items()]

    if len(source_values) != len(destination_keys):
        print(f"Warning: Number of markers in source ({len(source_values)}) "
              f"does not match number of markers in destination ({len(destination_keys)}). "
              f"Copying up to the minimum count.")
    
    # Copy values based on order
    num_markers_to_copy = min(len(source_values), len(destination_keys))
    for i in range(num_markers_to_copy):
        dest_key = destination_keys[i]
        source_value = source_values[i]
        destination_markers[dest_key] = source_value

    # Ensure the output directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the modified data to the new output file
    try:
        with open(output_file_path, 'w') as f:
            json.dump(destination_data, f, indent=2)
        print(f"Successfully copied marker values from {source_file_path} "
              f"to {destination_file_path} and saved the result to {output_file_path}")
    except Exception as e:
        print(f"Error: Failed to save modified data to {output_file_path}. Error: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define your file paths
    file_A_path = f"canonical_data_tpose_new.json"
    file_B_path = f"canonical_data_tpose.json"
    output_C_path = f"canonical_data_tpose_merged.json" # New output file

    copy_marker_values(file_A_path, file_B_path, output_C_path)