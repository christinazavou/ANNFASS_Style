import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, help="result_dir", required=True)
ARGS = parser.parse_args()
print(ARGS)

# Merge .csv
overall_distances = {}
for job_dir in os.listdir(ARGS.result_dir):
    file = os.path.join(ARGS.result_dir, job_dir, "distances.csv")
    assert os.path.exists(file)
    with open(file, 'r') as fin:
        line = fin.readline().strip().split(',')
        assert (line[0] == "source_models")
        assert (line[1] == "target_models")
        assert (line[2] == "equal_vertices")
        assert (line[3] == "equal_faces")
        assert (line[4] == "distances")
        for line in fin:
            line = line.strip().split(' ')
            if line[0] not in overall_distances:
                overall_distances[line[0]] = (line[1], line[2], line[3], line[4])
            elif line[4] < overall_distances[line[0]][3]:
                overall_distances[line[0]] = (line[1], line[2], line[3], line[4])
with open(os.path.join(ARGS.result_dir, "distances.csv"), 'w') as fout:
    # Write header
    fout.write("source_models target_models equal_vertices equal_faces distances\n")
    for query_name, distance in overall_distances.items():
        fout.write("{} {} {} {} {}\n".format(query_name,
                                             distance[0],
                                             distance[1],
                                             distance[2],
                                             distance[3]))
