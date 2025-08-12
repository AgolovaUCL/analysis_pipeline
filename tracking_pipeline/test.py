import glob
import os

tr = 1
folder_directory= r"Z:\Eylon\Data\Spatiotemporal_Task\rawdata\sub-003_id_2V\ses-01_date-30072025\tracking"
pattern = os.path.join(folder_directory, f"T{tr}_*.avi")
matches = glob.glob(pattern)
print(matches)
