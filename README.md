# FractalsFun

Some Python code for generating some interesting fractal images.

Generate images of fractals of type:
 - [Sierpinski Gasket](https://en.wikipedia.org/wiki/Sierpi%C5%84ski_triangle) (a.k.a Sierpinksi Triangles)
 - [Bedford-McMullen Carpet](https://demonstrations.wolfram.com/BedfordMcMullenCarpets/)
 - 'Rotating Squares'

---

### Example usage:

To generate an array of Sierpinski triangles.
```
python fractals.py --fractal 'triangles' --max_iters '[[0,1],[2,3]]'
```

![Resulting Sierpinski gasket image.](triangles_display.png "Sierpinksi Gasket")

To generate a single Bedford-McMullen carpet and display on screen.
```
python fractals.py --fractal 'carpet' --max_iters '[[5]]' --remove '[3]' -s
```

![Resulting Bedford-McMullen Carpet image.](carpet_display.png "Bedford-McMullen Carpet")