# FractalsFun

Some Python code for generating some interesting fractal images.

Generate images of fractals of type:
 - Sierpinski Gasket (a.k.a Sierpinksi Triangles)
 - Bedford-McMullen Carpet
 - 'Rotating Squares'


Example usage:
--------------
To generate an array of Sierpinski triangles with transparent background
```
python fractals.py --fractal 'triangles' --max_iters '[[0,1],[2,3]]' -t
```

To generate a single Bedford-McMullen carpet and display on screen
```
python fractals.py --fractal 'carpet' --max_iters '[[3]]' --remove '[3]' -s
```