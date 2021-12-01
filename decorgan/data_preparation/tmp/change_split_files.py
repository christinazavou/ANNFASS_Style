import random


def run(in_file, keep):
    out_file = in_file.replace('.txt', '_balanced.txt')

    def get_building_from_filename(file):
        return file.split("/")[0]

    def get_element_from_filename(file):
        return file.split("__")[0].split("_")[-1]

    with open(in_file) as fin:
        lines = fin.readlines()

    buildings = set()
    lines_per_element = {}
    for line in lines:
        el = get_element_from_filename(line.strip())
        lines_per_element.setdefault(el, [])
        lines_per_element[el] += [line]
        b = get_building_from_filename(line.strip())
        buildings.add(b)
    print(f"total buildings : {len(buildings)}")

    frequencies = {}
    for el, lines in lines_per_element.items():
        frequencies[el] = len(lines)
    print(frequencies)

    with open(out_file, "w") as fout:
        for el in lines_per_element:
            elements = lines_per_element[el]
            random.shuffle(elements)
            fout.writelines(elements[0:keep])


def get_buildings(in_file):
    def get_building_from_filename(file):
        return file.split("/")[0]

    with open(in_file) as fin:
        lines = fin.readlines()

    buildings = set()
    for line in lines:
        b = get_building_from_filename(line.strip())
        buildings.add(b)
    print(f"total buildings : {len(buildings)}")


run("/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/from_gypsum/buildnet_component_refined_v2_max_train.txt", 500)
run("/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/from_gypsum/buildnet_component_refined_v2_max_val.txt", 100)
# get_buildings("/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/from_gypsum/buildnet_component_max_train.txt")
# get_buildings("/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/from_gypsum/buildnet_component_max_train_balanced.txt")
