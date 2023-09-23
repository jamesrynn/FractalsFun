"""
Script for generating images of Fractals, see README.md for details.
"""

##  -----  MODULES  -----  ##

import argparse
import ast
import math

from datetime import datetime
from PIL import Image, ImageColor, ImageDraw

import numpy as np


##  -----  GLOBAL PARAMETERS  -----  ##

# per-shape/image bounding box edge lengths in pixels
SIZE_X, SIZE_Y = 2500, 2500
SIZE_MIN = min(SIZE_X, SIZE_Y)

# spacing between shapes
SPACING_X, SPACING_Y = int(SIZE_X/20), int(SIZE_Y/20)

FILL_RATIO = 0.65

DPI = (450,450)

# colours
BLACK = '#000000'
GREY = '#808080'
WHITE = '#ffffff'
RED = '#ff0000'


##  -----  FUNCTIONS  -----  ##

def draw_gasket(centre_x, centre_y, radius, image, iteration, max_iterations, fill_colour=BLACK):
    """
    Produces 'Sierpinksi Gasket' fractal by recursively drawing triangles.

    Parameters:
    -----------
    centre_x : float
        x co-ordinate triangle centre
    centre_y : float
        y co-ordinate of triangle centre
    radius : float
        radius of triangle bounding circle
    image : PIL.ImageDraw.Draw
        image object to be mutated
    iteration : int
        current iteration
    max_iterations : int
        max number of iterations
    fill_colour : str (OPTIONAL, default='#000000')
        hex code of fill colour
    """

    # define sub-triangles in terms of (centre_x, centre_y, radius)
    sub_triangles = [
        (centre_x-math.sqrt(3)*radius/4, centre_y+radius/4, radius/2),
        (centre_x, centre_y-radius/2, radius/2),
        (centre_x+math.sqrt(3)*radius/4, centre_y+radius/4, radius/2)
    ]

    # iterate on sub-triangles
    if iteration < max_iterations:

        # white central sub-triangle
        image.regular_polygon((centre_x, centre_y, radius/2), n_sides=3, fill=WHITE, rotation=180)

        # filled sub-triangles
        for (t_x, t_y, t_r) in sub_triangles:
            draw_gasket(
                centre_x = t_x,
                centre_y = t_y,
                radius = t_r,
                image = image,
                iteration = iteration+1,
                max_iterations = max_iterations,
                fill_colour = fill_colour
            )


def draw_carpet(x_0, y_0, width, height, image, iteration, max_iterations, remove=None, keep=None, m=3, n=2, empty_colour=WHITE):
    """
    Produces 'Bedford-McMullen Carpet' fractal by recusively drawing rectangles.

    Parameters:
    -----------
    x_0 : float
        x co-ordinate of top left of rectangle
    y_0 : float
        y co-ordinate of top left of rectangle
    width : float
        rectangle width
    height : float
        rectangle height
    image : PIL.ImageDraw.Draw
        image object to be mutated
    iteration : int
        current iteration
    max_iterations : int
        max number of iterations
    remove : list[int] | None (default=None)
        list of rectangles to remove at each step
    keep : list[int] | None (default=None)
        list of rectangles to retain at each step
    m : int (default=3)
        number of rows in carpet
    n : int (default=2)
        number of columns in carpet
    empty_colour : str (default='#ffffff')
        colour for removed rectangles
    """

    # number of grid elements
    num_el = m*n

    ## check inputs
    if num_el < len(remove):
        raise ValueError(f'Number of elements to remove ({len(remove)}) must be less than number of grid elements ({num_el})')

    if not set(remove) < set(range(num_el)):
        raise ValueError(f'Elements to remove ({remove}) must be a (strict) subset of the number of elements ({num_el}).')

    if keep and remove:
        raise ValueError('Must pass either keep or remove parameter, not both.')
    if remove:
        # rectangles to retain
        keep = [i for i in range(num_el) if i not in remove]
    elif keep:
        # rectangles to remove
        remove = [i for i in range(num_el) if i not in keep]
    else:
        raise ValueError('Must pass either keep or remove parameter.')

    # sub-rectangle dimensions
    sub_h = height/m
    sub_w = width/n

    # define sub-rectangles by top-left corner co-ordinates
    rectangles = [(x_0+i*sub_w, y_0+j*sub_h) for i in range(n) for j in range(m)]

    if iteration < max_iterations:

        # remove designated rectangles by filling white
        for i in remove:
            (x,y) = rectangles[i]
            image.rectangle([(x,y), (x+sub_w,y+sub_h)], fill=empty_colour)

        # repeat on subrectangles that are retained
        for i in keep:
            (x,y) = rectangles[i]
            draw_carpet(
                x_0 = x,
                y_0 = y,
                width = sub_w,
                height = sub_h,
                image = image,
                iteration = iteration+1,
                max_iterations = max_iterations,
                remove = remove,
                m = m,
                n = n,
                empty_colour = empty_colour
            )

def draw_square(centre, side_length, radians_rotate, thickness, colour, pixels, square_ratio, iteration, max_iterations):
    """
    Produces a fractal of self-similar sets with dense (irrational) rotations in R^2 by recursively
    drawing squares.

    Parameters:
    -----------
    centre : list[int]
        [y,x] point within the array of pixels where the centre of the square lies
    side_length : float
        length of each side of the square
    radians_rotate  : float
        how many radians square is to be rotated about the centre
    thickness : int
        edge line thickness
    colour : list[int]
        [r,g,b] colour of lines
    pixels : np.array (dimension n_rows x ncols x 3)
        3d array representing the pixels
    square_ratio : float
        ratio to shrink chosen sub-square by with each iteration
    max_iterations : int
        maximum number of iterations
    """

    corner_top_left     = [centre[0] - side_length/2, centre[1] - side_length/2]
    corner_top_right    = [centre[0] + side_length/2, centre[1] - side_length/2]
    corner_bottom_right = [centre[0] + side_length/2, centre[1] + side_length/2]
    corner_bottom_left  = [centre[0] - side_length/2, centre[1] + side_length/2]

    if (radians_rotate != 0):
        corner_bottom_right = _rotate_coordinate_around_point(corner_bottom_right, centre, radians_rotate)
        corner_bottom_left = _rotate_coordinate_around_point(corner_bottom_left, centre, radians_rotate)
        corner_top_left = _rotate_coordinate_around_point(corner_top_left, centre, radians_rotate)
        corner_top_right = _rotate_coordinate_around_point(corner_top_right, centre, radians_rotate)

    lines = [
        [corner_top_left, corner_top_right],       # top
        [corner_top_right, corner_bottom_right],   # right
        [corner_bottom_right, corner_bottom_left], # bottom
        [corner_bottom_left, corner_top_left]      # left
    ]

    for line in lines:
        _plot_line(line[0], line[1], thickness, colour, pixels)

    new_side_length_1 = side_length/2
    new_side_length_2 = square_ratio * new_side_length_1

    new_centre_1 = _rotate_coordinate_around_point([centre[0] + side_length/4, centre[1] - side_length/4], centre, radians_rotate)
    new_centre_2 = _rotate_coordinate_around_point([centre[0] - side_length/4, centre[1] + side_length/4], centre, radians_rotate)

    # rotational angle is irrational multiple of pi
    alpha = math.exp(1)*math.pi/8

    if iteration < max_iterations:
        draw_square(new_centre_1, new_side_length_1, radians_rotate,       thickness, colour, pixels, square_ratio, iteration+1, max_iterations)
        draw_square(new_centre_2, new_side_length_2, radians_rotate+alpha, thickness, colour, pixels, square_ratio, iteration+1, max_iterations)


def _plot_line(from_coordinates, to_coordinates, thickness, colour, pixels):
    """
    Plot a line between two coordinates.

    Parameters:
    -----------
    from_coordinates : list[int]
        [y,x] point within the array of pixels defining line start point
    to_coordinates : list[int]
        [y,x] point within the array of pizels defining line end point
    thickness : int
        line thickness
    colour : list[int]
        [r,g,b] colour for line
    pixels : np.array (dimension n_rows x ncols x 3)
        3d array representing the pixels
    """

    # boundaries of pixel array
    max_x_coordinate = len(pixels[0])
    max_y_coordinate = len(pixels)

    # istances along the x and y axis between the 2 points
    horizontal_distance = to_coordinates[1] - from_coordinates[1]
    vertical_distance = to_coordinates[0] - from_coordinates[0]

    # total distance between the two points, used to calculate pixel positions
    distance = math.sqrt((to_coordinates[1] - from_coordinates[1])**2 + (to_coordinates[0] - from_coordinates[0])**2)
    step = round(distance)

    # line step widths
    horizontal_step = horizontal_distance/step
    vertical_step = vertical_distance/step

    # draw line with pixel at each step
    for i in range(step):

        # line centre
        current_x_coordinate = round(from_coordinates[1] + (horizontal_step*i))
        current_y_coordinate = round(from_coordinates[0] + (vertical_step*i))

        if (0 < current_x_coordinate < max_x_coordinate and 0 < current_y_coordinate < max_y_coordinate):
            pixels[current_y_coordinate][current_x_coordinate] = colour

        # draw a 'point' (a square) of size <thickness> around the coordinates
        for x in range (-thickness, thickness):
            for y in range (-thickness, thickness):
                x_value = current_x_coordinate + x
                y_value = current_y_coordinate + y

                if (0 < x_value < max_x_coordinate and 0 < y_value < max_y_coordinate):
                    pixels[y_value][x_value] = colour

def _rotate_coordinate_around_point(coordinate, centre_point, radians):
    """
    Rotate 'coordinate' around the 'centre_point' by specified angle (in radians).

    Paramaters:
    -----------
    coordinate : list[int,int]
        [y,x] point to rotate
    centre_point : list[int,int]
        [y,x] point to rotate around
    radians : float
        radians to rotate by
    """

    # subtract the point to be rotated around from coordinate to remove the offset from 0,0
    x = (coordinate[0] - centre_point[0])
    y = (coordinate[1] - centre_point[1])

    # calculate rotated points
    new_x = (x * math.cos(radians)) - (y * math.sin(radians))
    new_y = (y * math.cos(radians)) + (x * math.sin(radians))

    # re-add offset to rotated points and return
    return [new_x + centre_point[0], new_y + centre_point[1]]


if __name__ == '__main__':

    ## parse input arguments
    ## ---------------------

    parser = argparse.ArgumentParser()

    # type of fractal to produce
    parser.add_argument('--fractal', required=False, type=str, default='sierpinski')

    # number of iterations per fractal in format list[list[int]] where length of outer list is
    # number of rows in full image and length of each sublist is the number of columns in row
    parser.add_argument('--max_iters',  required=False,  type=str,  default='[[0, 1], [2, 3]]')

    # colour of fractals specifed as HEX code
    parser.add_argument('--fill_colour', required=False, type=str,  default=BLACK)

    # filename
    parser.add_argument('--filename', required=False, type=str, default='')

    # list of sub-rectangles to remove (only appropriate for BM carpets)
    parser.add_argument('--remove', required=False, type=str, default='[1,3]')

    # angle of rotation (only appropriate for rotating squares)
    parser.add_argument('--angle', required=False, type=float, default=math.exp(1)*math.pi/8)

    # switch for setting transparency of PNG image produced
    parser.add_argument('-t', '--transparent', action='store_true')

    # switch for displaying image
    parser.add_argument('-s', '--show_image',  action='store_true')

    args = parser.parse_args()


    # convert max_depths parameter from string to list[list[int]] format
    max_iters = ast.literal_eval(args.max_iters)

    # convert remove parameter from string to list[int] format
    remove = ast.literal_eval(args.remove)


    ## build initial image
    ## -------------------

    n_rows = len(max_iters)                                # number of rows in full image
    n_cols = max(len(max_iters[i]) for i in range(n_rows)) # number of columns in full image

    w_ = n_cols*SIZE_X # combined width of sub-images
    h_ = n_rows*SIZE_Y # combined height of sub-images

    w = w_ + (n_cols+1)*SPACING_X # width of full image (with spacings)
    h = h_ + (n_rows+1)*SPACING_Y # height of full image (with spacings)

    # creating new image object
    base_img = Image.new('RGB', (w,h), ImageColor.getcolor(WHITE, 'RGB'))
    img = ImageDraw.Draw(base_img)

    # max depths matrix as list
    max_iters = [max_iters[i][j] for i in range(n_rows) for j in range(len(max_iters[i]))]


    ## build fractals
    ## --------------

    match args.fractal:
        # Sierpinksi triangles
        case 'triangles' | 'triangle' | 'sierpinski':
            # initial triangle radius
            r0 = FILL_RATIO*2*SIZE_MIN/3

            # initial triangle centres
            triangle_centres = [
                (
                    (j+1)*SPACING_X + (2*j+1)*SIZE_X/2,
                    (i+1)*SPACING_Y + (0.6+i)*SIZE_Y
                )
                for i in range(n_rows) for j in range(n_cols)
            ]

            for (t0_x, t0_y), max_iter in zip(triangle_centres, max_iters):

                # if no max iterations value, skip
                if max_iter is None:
                    continue

                # colour in initial triangle
                img.regular_polygon((t0_x, t0_y, r0), n_sides=3, fill=args.fill_colour)

                # call sub function on each initial triangle
                draw_gasket(
                    centre_x = t0_x,
                    centre_y = t0_y,
                    radius = r0,
                    image = img,
                    iteration = 0,
                    max_iterations = max_iter,
                    fill_colour = args.fill_colour
                )

        # Bedford-McMullen carpet
        case 'rectangles' | 'rectangle' | 'carpet' | 'BedfordMcMullen':

            # colour in initial rectangle with background colour highlight space between sub-images
            img.rectangle([(0,0), (w,h)], fill=GREY)

            rectangle_corners = [
                (
                    (j+1)*SPACING_X + j*SIZE_X,
                    (i+1)*SPACING_Y + i*SIZE_Y
                )
                for i in range(n_rows) for j in range(n_cols)
            ]

            for (r_x0, r_y0), max_iter in zip(rectangle_corners, max_iters):

                # draw initial sub_rectangle
                img.rectangle([(r_x0,r_y0),(r_x0+SIZE_X,r_y0+SIZE_Y)], fill=args.fill_colour)

                # if no max iterations value, skip
                if max_iter is None:
                    continue

                draw_carpet(
                    x_0 = r_x0,
                    y_0 = r_y0,
                    width = SIZE_X,
                    height = SIZE_Y,
                    remove = remove,
                    image = img,
                    iteration = 0,
                    max_iterations = max_iter
                )

        # rotating squares
        case 'squares' | 'square':

            img_pixels = 255 * np.ones((SIZE_MIN,SIZE_MIN,3), dtype=np.uint8)

            # TODO: generalise to allow grid of fractals
            if len(max_iters)!=1:
                raise ValueError('Only a single max iterations value may be provided. e.g. [[1]]')

            draw_square(
                centre = [SIZE_X/2, SIZE_Y/2],
                side_length = 0.85*SIZE_MIN,
                radians_rotate = 0,
                thickness = 4,
                colour = ImageColor.getcolor(args.fill_colour,'RGB'),
                pixels = img_pixels,
                square_ratio = FILL_RATIO,
                iteration = 0,
                max_iterations = max_iters[0]
            )

            # generate image from pixels array
            base_img = Image.fromarray(img_pixels)


        # handle unknown fractal requests
        case _:
            raise ValueError(f'unrecognised fractal type: {args.fractal}.')


    ## display image (if requested)
    ## ----------------------------
    if args.show_image:
        base_img.show()


    ## save file as .PNG file
    ## ----------------------
    if args.filename:
        filename = f'{args.filename.split(".")[0]}.png'
    else:
        filename = f'{args.fractal}_{datetime.now().strftime("%Y-%m-%dT%H%M%SZ")}.png'

    if args.transparent:
        # save image with transparent background
        base_img.save(filename, format='PNG', dpi=DPI, transparency=ImageColor.getcolor(WHITE,'RGB'))
    else:
        # save image with white background
        base_img.save(filename, format='PNG', dpi=DPI)
