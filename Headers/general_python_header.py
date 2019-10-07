#############################################################################
#
# Author            : Martin Lopez Jr.
#
# Purpose           : Create a python header with my commonly imported
#                     packages and general figure preferences.
#
#############################################################################


########################################
#                HEADER
########################################

import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import importlib as il
import general.common as com


########################################
#                CONSTANTS
########################################

### Physical Constants
G = 6.674e-8           # Gravitational Constant [g^-1 s^-2 cm^3]
c = 2.99792458e10      # Speed of Light [cm s^-1]
mp = 1.6726219e-24     # Mass of Proton [g]
me = 9.10938356e-28    # Mass of Electron [g]
kB = 1.380658e-16      # Boltzmann Constant

### Astronomical Constants
AU = 1.496e13          # Astronomical Unit [cm]
pc = 3.085677581e18    # Parsec [cm]

### Solar Constants
Mo = 1.989e33                        # Solar Mass [g]
Ro= 6.955e10                         # Solar Radius [cm]
Lo = 3.9e33                          # Solar Luminosity [erg s^-1]
To = 5.78e3                          # Solar Temperature [K]
Vo= np.sqrt(2*G*Mo/Ro)               # Solar Escape Velocity [cm s^-1]
Rhoo = Mo/((4./3.) * np.pi * Ro**3)  # Solar Average Density [g cm^-3]


########################################
#             CONVERSION FACTORS
########################################

sec_per_year = 3.1536e7
sec_per_day = 60.*60.*24
sec_per_hrs = 60.*60.
sec_per_min = 60.
sec_per_tdyn = np.sqrt(Ro**3/(G*Mo))

########################################
#                RC PARAMS
########################################

### LINES
# See http://matplotlib.org/api/artist_api.html#module-matplotlib.lines for more
# information on line properties.

plt.rc('lines',
        linewidth = 3,               # line width in points
        linestyle = '-',               # solid line
        marker = 'None',               # the default marker
        markeredgewidth = 1.0,         # the line width around the marker symbol
        markersize = 6,                # markersize, in points
        dash_joinstyle = 'miter',      # miter|round|bevel
        dash_capstyle = 'butt',        # butt|round|projecting
        solid_joinstyle = 'miter',     # miter|round|bevel
        solid_capstyle = 'projecting', # butt|round|projecting
        antialiased = True)            # render lines in antialiased (no jaggies)


### AXES
# default face and edge color, default tick sizes,
# default fontsizes for ticklabels, and so on.  See
# http://matplotlib.org/api/axes_api.html#module-matplotlib.axes

plt.rc('axes',
        facecolor = 'white',    # axes background color
        edgecolor = 'black',    # axes edge color
        linewidth = 0.8,        # edge linewidth
        grid = False,           # display grid or not
        titlesize = 25,    # fontsize of the axes title
        titlepad = 6.0,         # pad between axes and title in points
        labelsize = 20,   # fontsize of the x any y labels
        labelpad = 8.0,         # space between label and axis
        labelweight = 'normal', # weight of the x and y labels
        labelcolor = 'black',
        axisbelow = 'line',     # draw axis gridlines and ticks below
                                # patches (True); above patches but below
                                # lines ('line'); or above all (False)
        xmargin = .05,          # x margin.
        ymargin = .05,          # y margin
        prop_cycle = plt.cycler('color', ['#00BFFF', # color cycle for plot lines
                                          '#FF4500', # as list of string colorspecs:
                                          '#2DC22D', # single letter, long name, or
                                          '#E60B0D', # web-style hex
                                          '#801ED9',
                                          '#F527B6',
                                          '#8C8C8C',
                                          '#FFD700',
                                          '#1f77b4',
                                          '#ff7f0e',
                                          '#2ca02c',
                                          '#d62728',
                                          '#9467bd',
                                          '#8c564b',
                                          '#e377c2',
                                          '#7f7f7f',
                                          '#bcbd22',
                                          '#17becf']))

## POLAR AXES
plt.rc('polaraxes',
        grid = True)    # display grid on polar axes

## 3D AXES
plt.rc('axes3d',
        grid = True)    # display grid on 3d axes


### TICKS
# see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick

## x ticks
plt.rc('xtick',
        top = False,          # draw ticks on the top side
        bottom = True,        # draw ticks on the bottom side
        color = 'k',          # color of the tick labels
        labelsize = 12, # fontsize of the tick labels
        direction = 'out')    # direction: in, out, or inout

## x major ticks
plt.rc('xtick.major',
        size = 5,      # major tick size in points
        width = 0.8,     # major tick width in points
        pad = 3.5,       # distance to major tick label in points
        top = True,      # draw x axis top major ticks
        bottom = True)   # draw x axis bottom major ticks

## x minor ticks
plt.rc('xtick.minor',
        size = 2,          # minor tick size in points
        width = 0.6,       # minor tick width in points
        pad = 3.4,         # distance to the minor tick label in points
        visible = False, # visibility of minor ticks on x-axis
        top = True,      # draw x axis top minor ticks
        bottom = True)   # draw x axis bottom minor ticks

## y ticks
plt.rc('ytick',
        right = False,        # draw ticks on the right side
        left = True,        # draw ticks on the left side
        color = 'k',           # color of the tick labels
        labelsize = 12,  # fontsize of the tick labels
        direction = 'out')     # direction: in, out, or inout

## y major ticks
plt.rc('ytick.major',
        size = 5,       # major tick size in points
        width = 0.8,      # major tick width in points
        pad = 3.5,        # distance to major tick label in points
        left = True,      # draw y axis left major ticks
        right = True)     # draw y axis right major ticks

## y minor ticks
plt.rc('ytick.minor',
        size = 2,           # minor tick size in points
        width = 0.6,        # minor tick width in points
        pad = 3.4,          # distance to the minor tick label in points
        visible = False,    # visibility of minor ticks on y-axis
        left = True,        # draw y axis left minor ticks
        right = True)       # draw y axis right minor ticks

### GRIDS

plt.rc('grid',
        color = 'b0b0b0',  # grid color
        linestyle = '-',   # solid
        linewidth = 0.8,   # in points
        alpha = 1.0)       # transparency, between 0.0 and 1.0


### LEGEND
plt.rc('legend',
        loc = 'best',
        frameon = False,         # if True, draw the legend on a background patch
        framealpha = 0.8,       # legend patch transparency
        facecolor = 'inherit',  # inherit from axes.facecolor; or color spec
        edgecolor = '0.8',      # background patch boundary color
        fancybox = True,        # if True, use a rounded box for the
                                # legend background, else a rectangle
        shadow = False,         # if True, give background a shadow effect
        numpoints = 1,          # the number of marker points in the legend line
        scatterpoints = 1,      # number of scatter points
        markerscale = 1.0,      # the relative size of legend markers vs. original
        fontsize = 16,
        # Dimensions as fraction of fontsize:
        borderpad =  0.4,       # border whitespace
        labelspacing =  0.5,    # the vertical space between the legend entries
        handlelength =  2.0,    # the length of the legend lines
        handleheight =  0.7,    # the height of the legend handle
        handletextpad = 0.8,    # the space between the legend line and legend text
        borderaxespad = 0.5,    # the border between the axes and legend edge
        columnspacing = 2.0)    # column separation

### FONT
# See http://matplotlib.org/api/font_manager_api.html for more
# information on font properties.

# font = {'family': 'serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 20,
#         }

plt.rc('font',
        family = 'serif',
        style = 'normal',
        variant = 'normal',
        weight = 'medium',
        stretch = 'normal',
        # note that font.size controls default text sizes.  To configure
        # special text sizes tick labels, axes, labels, title, etc, see the rc
        # settings for axes and ticks. Special text sizes can be defined
        # relative to font.size, using the following values: xx-small, x-small,
        # small, medium, large, x-large, xx-large, larger, or smaller
        size = 20.0)

## SANS-SERIF
plt.rcParams['font.sans-serif'] = ['Verdana']

### FIGURE
# See http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure

plt.rc('figure',
        titlesize = 'large',       # size of the figure title (Figure.suptitle())
        titleweight = 'normal',    # weight of the figure title
        figsize = (8,5),        # figure size in inches
        dpi = 100,                 # figure dots per inch
        facecolor = 'white',       # figure facecolor; 0.75 is scalar gray
        edgecolor = 'white',       # figure edgecolor
        autolayout = False,        # When True, automatically adjust subplot
                                   # parameters to make the plot fit the figure
        max_open_warning = 20)     # The maximum number of figures to open through
                                   # the pyplot interface before emitting a warning.
                                   # If less than one this feature is disabled.

## SUBPLOT
plt.rc('figure.subplot',  #All dimensions are a fraction of the
        left = 0.125,     # the left side of the subplots of the figure
        right = 0.9,      # the right side of the subplots of the figure
        bottom = 0.11,    # the bottom of the subplots of the figure
        top = 0.88,       # the top of the subplots of the figure
        wspace = 0.2,     # the amount of width reserved for blank space between subplots,
                          # expressed as a fraction of the average axis width
        hspace = 0.2)     # the amount of height reserved for white space between subplots
                          # expressed as a fraction of the average axis height

### IMAGES
plt.rc('image',
        aspect = 'equal',           # equal | auto | a number
        interpolation = 'nearest',  # see help(imshow) for options
        cmap = 'viridis',           # A colormap name, gray etc...
        lut  = 256,                 # the size of the colormap lookup table
        origin = 'upper',             # lower | upper
        resample = True,
        composite_image = True)     # When True, all the images on a set of axes are
                                    # combined into a single composite image before
                                    # saving a figure as a vector graphics file,
                                    # such as a PDF.

### CONTOUR PLOTS
plt.rc('contour',
        negative_linestyle = 'dashed', # dashed | solid
        corner_mask = True)            # True | False | legacy

### ERRORBAR PLOTS
plt.rc('errorbar',
        capsize = 0)              # length of end cap on error bars in pixels

### HISTOGRAM PLOTS
plt.rc('hist',
        bins = 10)                # The default number of histogram bins.
                                  # If Numpy 1.11 or later is
                                  # installed, may also be `auto`

### SCATTER PLOTS
plt.rc('scatter',
        marker = 'o')               # The default marker type for scatter plots.

### SAVING FIGURES
# the default savefig params can be different from the display params
# e.g., you may want a higher resolution, or to make the figure
# background white

plt.rc('savefig',
        dpi = 'figure',         # figure dots per inch or 'figure'
        facecolor = 'white',    # figure facecolor when saving
        edgecolor = 'white',    # figure edgecolor when saving
        format = 'png',         # png, ps, pdf, svg
        bbox = 'standard',      # 'tight' or 'standard'.
                                # 'tight' is incompatible with pipe-based animation
                                # backends but will workd with temporary file based ones:
                                # e.g. setting animation.writer to ffmpeg will not work,
                                # use ffmpeg_file instead
        pad_inches = 0.1,       # Padding to be used when bbox is set to 'tight'
        jpeg_quality = 95,      # when a jpeg is saved, the default quality parameter.
        directory = '~',        # default directory in savefig dialog box,
                                # leave empty to always use current working directory
        transparent = False)    # setting that controls whether figures are saved with a
                                # transparent background by default
