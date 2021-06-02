## Data Generation

Running DataGeneration.py with the specification will give you the datas.

`scf` is the configuration for the synthesizer class, should be left alone for this case.

The function `generator_sampling` will sample the images `sampling_times` times. This functions calls `generate_voronoi` which creates the base voronoi diagram.

The function takes a config for the voronoi diagram, min cells, max cells etc. You can play around with this to get different results.


