input_path = "../../data/KISA_TBC_VIEWS_UNIQ.csv"
output_path = "../../data/short.csv"

with open(input_path, "r") as file:
    input = file.readlines()

dict = {}

for i, line in enumerate(input):
    input[i] = line.split(",")[:2]
    if not input[i][0] in dict.keys():
        dict[input[i][0]] = [input[i][1]]
    else:
        dict[input[i][0]].append(input[i][1])

with open(output_path, "w") as file:
    for v in dict.values():
        line = ",".join(v)

        line += "\n"

        file.write(line)