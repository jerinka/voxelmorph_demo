# Deep Deformable Registration

Deformable registration is basically calculation a dense vector field 
or mapping for every pixel in moving image to fixed image. This is better
suited for deeplearning based registration compared to estimation of 
transformation matrix end to end. Voxelmorph have model creation classes
for deformable registration learning. 

[More on Voxelmorph](https://github.com/learn2reg/tutorials2019/blob/master/slides/Learn2reg_tutorial_unsupervided_AdrianDalca.pdf)

# Clone
```git clone https://github.com/jerinka/voxelmorph_demo```

# Install
```virtualenv venv3 --python=python3```\
```source venv3/bin/activate```\
```pip3 install -r requirements.txt```


# Run Demo

## Using voxelmorph class (Preferred)
```python3 simple_voxelmorph.py```

## From scratch demo in tf
```python3 register_basics.py```

## Detailed code from voxelmorph tutorial
```python voxelmorph_tutorial.py```


