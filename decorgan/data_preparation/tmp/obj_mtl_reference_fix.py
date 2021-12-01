import os
from shutil import copyfile
from shutil import copytree


objs = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj_refinedTextures"

out_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj_refinedTextures_Local"
os.makedirs(out_dir, exist_ok=True)


for building_dir in os.listdir(objs):
    if not 'religious' in building_dir.lower():
        continue
    print(building_dir)
    in_mtl = os.path.join(objs, building_dir, f"{building_dir}.mtl")
    if not os.path.exists(in_mtl):
        continue
    os.makedirs(os.path.join(out_dir, building_dir), exist_ok=True)
    out_mtl = os.path.join(out_dir, building_dir, f"{building_dir}.mtl")
    with open(in_mtl, "r") as fin:
        lines = fin.read()
    with open(out_mtl, "w") as fout:
        fout.write(lines.replace("/mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/normalizedObj/",
                                 "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj/"))
    copyfile(os.path.join(objs, building_dir, f"{building_dir}.obj"),
             os.path.join(out_dir, building_dir, f"{building_dir}.obj"))
    copytree(os.path.join(objs.replace("_refinedTextures", ""), building_dir, building_dir),
             os.path.join(out_dir, building_dir, building_dir))
