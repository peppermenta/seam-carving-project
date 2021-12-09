# Energy Maps for Seam Carving

We experiment with various energy maps for seam carving in this repository. The four kinds of energy maps covered are:

1. Gradient Energy Map
2. Major Blob Map
3. Self-Attention Map
4. Saliency Map

Standard seam carving algorithm used in the original paper is employed here.

- To observe the difference between Gradient Energy and Major Blob seam carving methods, run `seam_carving.py`
- For deeper look at the Gradient Energy and Major Blob maps, check out `energy_maps.py`
- Check out `generate_dino_maps.py` to understand how self-attention maps are generated.
- Navigate to `dino_maps` folder and run `seam_carving_dino.py` to observe the difference between conventional and DINO self-attention energy map methods.
- Navigate to `dino_maps` folder and run `seam_carving_combined_maps.py` to generate output by combining major blob and self-attention energy maps.

Contributors:
- Tarun Ram - Original Seam carving algorithm, Original Gradient Energy Map, Integrated Gradients based Saliency Energy Map, and Combination of Energy Maps
- Agraj Srivastava - Major Blob energy map and alternative seam carving implementation with faster execution
- Mansi Nanavati - DINO Self Attention energy map, Combining Energy Maps using major blob method and Self-Attention.
- Soumi Chakraborty - Researching combination of energy maps and implementing evaluation metric to compare energy maps
