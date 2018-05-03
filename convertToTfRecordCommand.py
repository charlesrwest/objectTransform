import sys
import convertToTfRecord

folder_path = sys.argv[1]
output_file_path = sys.argv[2]
target_width = int(sys.argv[3])
target_height = int(sys.argv[4])

convertToTfRecord.ConvertToTfRecord(folder_path, output_file_path, target_width, target_height)
