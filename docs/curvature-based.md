Baseline SVM with curvatures
```
either
  cd preprocess/scripts/local
  make point-clouds-curvature END_IDX=30 SAMPLES=15000 PTS_ON_STYLE=True
  make pts2ply-per-stylistic-component ANNFASS_REPO=ANNFASS_Buildings_march INIT_SAMPLES=15K_style_mesh_wc PLY_PER_COMPONENT=ply15Kwcpercomponent
or
  cd preprocess/scripts/cluster
  make point-clouds-annfass-curvature SAMPLES=30000 NUM_PROCESSES=3 PTS_ON_STYLE=True
  cd preprocess/scripts/local
  make pts2ply-per-stylistic-component ANNFASS_REPO=ANNFASS_Buildings_march INIT_SAMPLES=30K_style_mesh_wc PLY_PER_COMPONENT=ply30Kwcpercomponent

cd splits
make ply_splits REPO=ANNFASS_Buildings_march PLY_SAMPLES=ply15Kwcpercomponent SPLIT_DIR=annfass_splits_march

cd sklearn_impl
make run-svm-curvatures PLY_SAMPLES=ply15Kwcpercomponent
```