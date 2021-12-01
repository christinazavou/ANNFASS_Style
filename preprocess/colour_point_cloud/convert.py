
import os

rd="/home/graphicslab/Desktop/samplePoints_refinedTextures/withColorPratheba"
rd_new="/home/graphicslab/Desktop/samplePoints_refinedTextures/withColorPratheba_new"
os.makedirs(rd_new, exist_ok=True)
for f in os.listdir(rd):
    if f.endswith(".ply"):
        with open(os.path.join(rd, f), "r") as fin:
            lines = fin.readlines()

        with open(os.path.join(rd_new, f), "w") as fout:
            header = True
            for line in lines:
                if header:
                    if "float r" in line or "float g" in line or "float b" in line or "float alpha" in line:
                        fout.write(line.replace("float", "uchar"))
                    else:
                        fout.write(line)
                else:
                    d = line.strip().split(" ")
                    fout.write(" ".join(d[:-4])
                               + f" {int(float(d[-4]) * 255)} {int(float(d[-3]) * 255)} {int(float(d[-2]) * 255)} {int(float(d[-1]) * 255)}\n")
                if "end_header" in line:
                    header=False
