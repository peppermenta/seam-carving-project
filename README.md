# Energy Maps for Seam Carving

We experiment with various energy maps for seam carving in this repository. The three kinds of energy maps covered are:

1. Gradient Energy Map
2. Major Blob Map
3. Self-Attention Map

Standard seam carving algorithm used in the first paper is employed here.

- To observe the difference between Gradient Energy and Major Blob seam carving methods, run `seam_carving.py`
- For deeper look at the Gradient Energy and Major Blob maps, check out `energy_maps.py`
- Navigate to `dino_maps` folder and run `seam_carving_dino` to observe the difference between conventional and DINO self-attention energy map methods.
- Check out `generate_dino_maps.py` to understand how self-attention maps are generated.

Contributors:
- Tarun Ram - Seam carving algorithm and gradient energy map
- Agraj Srivastava - Major Blob energy map and alternative seam carving implementation with faster execution
- Mansi Nanavati - DINO Self Attention energy map
- Soumi Chakraborty - Researching combination of energy maps 