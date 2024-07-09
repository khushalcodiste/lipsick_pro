file_path = "/mnt/c/Users/ADMIN/Khushal/lipsick/venv/lib/python3.10/site-packages/basicsr/data/degradations.py"
replacement_line = 'from torchvision.transforms.functional import rgb_to_grayscale\n'

# Read the content of the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Find and replace the desired line
for i, line in enumerate(lines):
    if "from torchvision.transforms.functional_tensor import rgb_to_grayscale" in line:
        lines[i] = replacement_line

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.writelines(lines)

print("Replacement done successfully.")
# C:\Users\ADMIN\Khushal\lipsick\lipsick_pro\app_1.py