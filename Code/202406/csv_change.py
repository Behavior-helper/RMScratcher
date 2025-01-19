# Read the file
with open('wrong_region.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Initialize variables
new_lines = []
current_group = []

# Process the lines
for line in lines:
    if line.startswith('data'):
        pass
        #current_group.append(line.strip())
    else:
        current_group.append(line.strip())

if current_group:
    for data_line in current_group:
        values = data_line.split(',')
        print(values)
        diff = round(float(values[1]) - float(values[0]), 3)
        new_line = f'{values[0]},{values[1]},{diff},{values[2]}'
        new_lines.append(new_line)
    current_group = []

# Write the updated data back to the file
with open('updated_data.txt', 'w', encoding='utf-8') as file:
    for line in new_lines:
        file.write(f'{line}\n')