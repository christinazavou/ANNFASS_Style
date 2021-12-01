import os


def merge_distances(result_dir):
    # Merge .csv
    overall_distances = {}
    for f in os.listdir(result_dir):
        if f.endswith(".csv"):
            with open(os.path.join(result_dir, f), 'r') as fin:
                line = fin.readline().strip().split(',')
                assert (line[0] == "source_models")
                assert (line[1] == "target_models")
                assert (line[2] == "equal_vertices")
                assert (line[3] == "equal_faces")
                assert (line[4] == "distances")
                for line in fin:
                    line = line.strip().split(',')
                    if line[0] not in overall_distances:
                        overall_distances[line[0]] = (line[1], line[2], line[3], line[4])
                    elif line[4] < overall_distances[line[0]][3]:
                        overall_distances[line[0]] = (line[1], line[2], line[3], line[4])
    with open(os.path.join(result_dir, "distances.csv"), 'w') as fout:
        # Write header
        fout.write("source_models,target_models,equal_vertices,equal_faces,distances\n")
        for query_name, distance in overall_distances.items():
            fout.write("{} {} {} {} {}\n".format(query_name,
                                                 distance[0],
                                                 distance[1],
                                                 distance[2],
                                                 distance[3]))
        # r_data = models_data[row[1]]
        # q_data = query_models_data[row[0]]
        # write_ply_v_f(r_data['vertices'], r_data['faces'], row[1]+".ply")
        # write_ply_v_f(q_data['vertices'], q_data['faces'], row[0]+".ply")


if __name__ == '__main__':
    jobs_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_YU_LUN_HU/duplicates/jobs_buildings_yu"
    for job_dir in os.listdir(jobs_dir):
        out_path = os.path.join(jobs_dir, job_dir, "distances.csv")
        if not os.path.exists(out_path):
            merge_distances(os.path.join(jobs_dir, job_dir))
