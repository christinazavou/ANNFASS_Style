notes:
- in mesh sampling the output points using K samples will be: 0.01 K <= points <= 1.01 K
- if you are running point cloud generation (e.g. rnv generation) a temp .txt file is being generated under 
  normalizedObj, thus you can't run the same code at the same time ...

preprocessing Annfass buildings (no colour):
```
cd preprocess/scripts/local
make triangulate-annfass ANNFASS_REPO=ANNFASS_Buildings_march
make normalize-annfass ANNFASS_REPO=ANNFASS_Buildings_march
Create a buildings.csv with the buildings ..
```

preprocessing Annfass buildings (with colour, i.e. refined textures):
```
1. manual steps:
First change materials of building 7 within Blender to not have reflex map and create the new fbx..
Also change canopy of building 11 and create the new fbx (merge canopy 20-37 and 38+39+41-119) ..
(note: save blend, save textures, export fbx)

2. automatic steps:
cd preprocess/scripts/local
make triangulate-annfass ANNFASS_REPO=ANNFASS_Buildings_may
make normalize-annfass ANNFASS_REPO=ANNFASS_Buildings_may

3. manual steps:
Open objs from normalizedObj and change some uvs maps and some materials:
 1. Open the main view, a shade editor, and a uv editor.
 2. In Object mode select the objects to be uv unwraped together.
 3. In Edit mode, with face selection mode, click on Select --> All.
 4. Right click --> Unwrap the faces --> UV Smart Project
 5. You will see change in the uv editor.
 6. Go to object mode.
 7. In Shade Editor, add a Texture Coordinate node, and a Mapping Node.
 8. Connect the UV from Texture Coordinate to Vector of Mapping and output into the texture file that goes in base colour
 of principled bsdf.
 9. In Mapping Node adjust scaling / rotation.
 10. Copy the initial obj folder and rename it with suffix "_refinedTextures" 
 11. Export obj in new folder with the filename suffix "_refinedTextures" 
 12. Create a buildings.csv with the buildings replaced with the suffix..
 ```
 ![blender img](resources/blender_re_texture.png?raw=true "Example how process looks in Blender interface")


To generate colour point clouds for Annfass buildings:
```
need to use the buildings_refinedTextures.csv both at point clouds generation and at colour extraction
because those objs have different amount of texture coordinates and faces than original objs.
```


To generate colour point clouds for Buildnet buildings:
```
cd preprocess/local or cluster/scripts
make retexture REPO=BUILDNET_Buildings PARTITION=titanx-long START_IDX=0 END_IDX=500 STEP_SIZE=100 
make point-clouds REPO=BUILDNET_Buildings SAMPLES=1000000 BUILDING_FILE=buildings OBJ_DIR=normalizedObj_refinedTextures SAMPLES_OUT_DIR=samplePoints_refinedTextures START_IDX=0 END_IDX=2000 STEP_SIZE=100 MEMORY=22GB
make colour-point-clouds REPO=BUILDNET_Buildings INIT_SAMPLES=1000K START_IDX=0 END_IDX=500 STEP_SIZE=100 BUILDING_FILE=buildings OBJ_DIR=normalizedObj_refinedTextures SAMPLES_OUT_DIR=samplePoints_refinedTextures
```

To have both annfass and buildnet component data together, annfass needs dummy grouping, and buildnet needs grouping:
```
cd preprocess/scripts/local
make groups-annfass

make annfass_style_stats (to see per building)

cd preprocess/scripts/cluster
make buildnet-group-triangles-as-component START_IDX=0 END_IDX=600 STEP_SIZE=100 PARTITION=m40-long BUILDING_FILE=buildings_religious GROUPS_DIR=group_triangles_to_component

if you want to delete the renderings made with cycles and do the ones with eevee locally:
cd /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_debug_cycles
rm */*_grouped_r*.png
cd preprocess/scripts/local
make buildnet-group-triangles-as-component START_IDX=0 END_IDX=600 STEP_SIZE=100 PARTITION=m40-long BUILDING_FILE=buildings_religious GROUPS_DIR=group_triangles_to_component
```

if you want to visualize the groups, you can upload them on google drive, e.g.:
``` 
cd preprocess/scripts/local
make upload-groups START_IDX=0 END_IDX=100 STEP_SIZE=30 GOOGLE_DRIVE_REPO=BUILDNET_Buildings_june BUILDING_FILE=buildings_religious GROUPS_DIR=group_triangles_to_component PARTITION=m40-long CONDA_PATH=/home/maverkiou/miniconda2
```

To use Annfass component point clouds:
```
make point-clouds REPO=ANNFASS_Buildings_may SAMPLES=500000 BUILDING_FILE=buildings END_IDX=30 PTS_ON_STYLE=True PTS_PROCESSES=6 MEMORY=18GB
make pts2ply_with_group REPO=ANNFASS_Buildings_may INIT_SAMPLES=500K_style_mesh BUILDING_FILE=buildings CUT_AT=10000 PLY_DIR_PREFIX=stylePly END_IDX=30
```
To use Buildnet component point clouds:
```
make point-clouds SAMPLES=500000 BUILDING_FILE=buildings_religious PTS_ON_STYLE=True END_IDX=500 PTS_PROCESSES=6 MEMORY=18GB
make pts2ply_with_group REPO=BUILDNET_Buildings INIT_SAMPLES=500K_style_mesh BUILDING_FILE=buildings_religious CUT_AT=10000 PLY_DIR_PREFIX=stylePly
```

Generate unique components (based on grouped components):
``` 
previous steps:
cd preprocess/local/scripts
make point-clouds ...
make pts2ply_with_group ...
e.g. make pts2ply_with_group START_IDX=0 END_IDX=500 STEP_SIZE=75 BUILDING_FILE=buildings_religious INIT_SAMPLES=10000K_style_mesh PLY_DIR_PREFIX=newgroupStylePly GROUPS_DIR=group_triangles_to_component CUT_AT=10000 SAMPLES_OUT_DIR=samplePoints.backup

current steps:
make find-unique-components END_IDX=3 STEP_SIZE=150 REPO=BUILDNET_Buildings SAMPLES_OUT_DIR=samplePoints PLY_DIR_PREFIX=stylePly_cut10.0K_pgc UNIQUE_DIR=newgroup_unique_point_clouds CONDA_PATH=/home/maverkiou/miniconda2 CONDA_ENV=py3-mink BUILDING_FILE=buildings_religious PARTITION=titanx-long
```

To render groups with eevee locally:
``` 
rm */_grouped*.png
and run again group....
make buildnet-group-triangles-as-component START_IDX=0 END_IDX=600 STEP_SIZE=100 PARTITION=m40-long BUILDING_FILE=buildings_religious GROUPS_DIR=group_triangles_to_component
```

To create plots of grouping rendering and unique components rendering per building:
``` 
first create pts2ply based on new grouping:
make pts2ply_with_group REPO=BUILDNET_Buildings END_IDX=3 STEP_SIZE=70 INIT_SAMPLES=500K_style_mesh BUILDING_FILE=buildings_religious CUT_AT=10000 GROUPS_DIR=groups_june17 PLY_DIR_PREFIX=groups_june17_stylePly
now make unique point clouds:
make find-unique-components END_IDX=3 STEP_SIZE=150 REPO=BUILDNET_Buildings SAMPLES_OUT_DIR=samplePoints PLY_DIR_PREFIX=groups_june17_stylePly_cut10.0K_pgc UNIQUE_DIR=groups_june17_unique_point_clouds CONDA_PATH=/home/maverkiou/miniconda2 CONDA_ENV=py3-mink BUILDING_FILE=buildings_religious PARTITION=titanx-long
now render unique components
cd preprocess/scripts/local
change locations in mtl files (using change_mtl_locations.py) ...
rm */*/img*png if already done wrong
make buildnet-render-components END_IDX=3 BUILDING_FILE=buildings_religious GROUPS_DIR=groups_june17 OBJCOMPONENTS_DIR=unifiedgroups17 UNIQUE_DIR=groups_june17_unique_point_clouds STEP_SIZE=70

in unique point cousd : find -name "*.ply" | wc -l
in uni_nor_components: find -name "*.png" | wc -l (must be 6 times the previous amount)
```

prerequisite step (for evaluation) for any style classification method:
```
cd splits
python find_missing_components_in_encodings.py
python make_stats.py
make classification_cross_val_splits
```


steps to generate point cloud on whole buildings annotated with style label and component id:
```
cd preprocess/scripts/local
make point-clouds SAMPLES=10000000 PTS_ON_STYLE=False
make pts2ply REPO=ANNFASS_Buildings_march INIT_SAMPLES=10000K CUT_AT=100000 PLY_SAMPLES=ply_100K_cnsc START_IDX=0 END_IDX=30 STEP_SIZE=30
(output will contain x,y,z,nx,ny,nz,style_label,component_id)
```

steps to generate point cloud on whole buildings annotated with style label, component id and ridge/valley:
```
cd preprocess/scripts/local
make point-clouds SAMPLES=10000000 PTS_ON_STYLE=False
make ridge-valley PTS_PROCESSES=4 ANNFASS_REPO=ANNFASS_Buildings_march INIT_SAMPLES=10000K
make pts2ply ANNFASS_REPO=ANNFASS_Buildings_march INIT_SAMPLES=10000K CUT_AT=100000 PLY_SAMPLES=ply_100K_cnscr
(output will contain x,y,z,nx,ny,nz,style_label,component_id, ridgevalley)
```

steps to generate point cloud per stylistic component annotated with style label:
``` 
cd preprocess/scripts/local
make point-clouds SAMPLES=10000000 PTS_ON_STYLE=True
make pts2ply-per-stylistic-component INIT_SAMPLES=10000K_style_mesh PLY_PER_COMPONENT=style1000Kply_pc
```
#fixme: rename into style_ply_100K_cns
for buildnet religious buildings (on cluster)
```
make point-clouds SAMPLES=10000000 PTS_ON_STYLE=True BUILDING_FILE=buildings_religious STEP_SIZE=100 JOBS_DIR=buildnetjobs REPO=BUILDNET_Buildings PTS_PROCESSES=8 START_IDX=0 END_IDX=500

[comment]: <> (make pts2ply REPO=BUILDNET_Buildings INIT_SAMPLES=10000K_style_mesh PLY_SAMPLES=style1000Kply_pc_u CUT_AT=-1 BUILDING_FILE=buildings_religious NUM_PROCESSES=12 PER_COMPONENT=True)
make pts2ply_buildnet_tmp REPO=BUILDNET_Buildings INIT_SAMPLES=10000K_style_mesh PLY_SAMPLES=style1000Kply_pc_g CUT_AT=-1 BUILDING_FILE=buildings_religious NUM_PROCESSES=12 PER_COMPONENT=True
```
steps to generate point cloud per stylistic component annotated with style label and ridge/valley:
``` 
cd preprocess/scripts/local
make point-clouds SAMPLES=10000000 PTS_ON_STYLE=True
make ridge-valley PTS_PROCESSES=4 ANNFASS_REPO=ANNFASS_Buildings_march INIT_SAMPLES=10000K_style_mesh
make pts2ply-per-stylistic-component INIT_SAMPLES=10000K_style_mesh PLY_PER_COMPONENT=style_ply_100K_cnsr
```


python make_component_csv.py --component_dirs
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_uni_nor_components,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/groups_june17_uni_nor_components
--unique_dirs
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds
--buildings_csv
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style.csv
--components_csv
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv
--override_labels
True,False
--parts
column,dome,door,window,tower

python make_component_csv.py --component_dirs
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_uni_nor_components,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/groups_june17_uni_nor_components
--unique_dirs
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds
--buildings_csv
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style_refinedTextures.csv
--components_csv
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style_refinedTextures.csv
--override_labels
True,False
--parts
column,dome,door,window,tower
