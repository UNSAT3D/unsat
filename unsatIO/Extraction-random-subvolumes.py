import csv
import tifffile
from random import randint

# Define the path where files are stored and CSV will be saved
path = "C:/Users/AnselmucciFAR/OneDrive - University of Twente/Bureaublad/UNSAT/" 

# Function to extract subvolumes and save their details
def extract_and_save_subvolumes(matrix, subvol_size, num_subvolumes, filename, csv_writer):
    x_size, y_size, z_size = subvol_size
    max_x, max_y, max_z = matrix.shape
    extracted_info = []

    for _ in range(num_subvolumes):
        # Randomly select the starting point
        x_start = randint(0, max_x - x_size)
        y_start = randint(0, max_y - y_size)
        z_start = randint(0, max_z - z_size)
        
        # Extract the subvolume
        subvolume = matrix[x_start:x_start+x_size, y_start:y_start+y_size, z_start:z_start+z_size]

        # Create the file name
        output_name = f"{filename}_{x_size}x{y_size}x{z_size}_{x_start}x{y_start}x{z_start}.tif"
        
        # Save the subvolume as a TIFF file
        tifffile.imsave(path + output_name, subvolume)

        # Collect information for writing to the CSV file
        extracted_info.append([output_name, x_start, x_size, y_start, y_size, z_start, z_size])
    
    return extracted_info

# Main function to process the 3D images and generate the CSV
def main():
    # Open the CSV file to write
    with open(path + 'subvolumes_extraction_log.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(["file name", "xstart", "xsize", "ystart", "ysize", "zstart", "zsize"])
        
        # Loop through each file and extract subvolumes
        for i in range(1, 7): # range to be updates based on the tot nomber of scan for the specific set
            filename_template = 'coarse-sand-sample-CLM04-day-0{}-raw' # to be modified
            filename = filename_template.format(i)
            matrix = tifffile.imread(path + filename + ".tif")
            subvol_size = (10, 10, 10)  # The desired subvolume size (x, y, z)
            num_subvolumes = 5  # The number of subvolumes to extract

            # Extract the subvolumes and get their information
            extracted_info = extract_and_save_subvolumes(matrix, subvol_size, num_subvolumes, filename, csv_writer)

            # Write the extraction info to the CSV file
            for info in extracted_info:
                csv_writer.writerow(info)

if __name__ == "__main__":
    main()
