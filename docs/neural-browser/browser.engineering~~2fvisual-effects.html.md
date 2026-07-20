Adding Visual Effects | Web Browser Engineering

Adding Visual Effects
=====================

[Twitter](https://twitter.com/browserbook) ·
[Blog](https://browserbook.substack.com/) ·
[Discussions](https://github.com/browserengineering/book/discussions)

Chapter 11 of [Web Browser Engineering](https://browser.engineering/index.html "Table of Contents").
[<](https://browser.engineering/security.html "Previous chapter")
[>](https://browser.engineering/scheduling.html "Next chapter")

![The cover for Web Browser Engineering, published by Oxford University Press. Click the cover to buy a copy.](https://browser.engineering/im/cover.jpg)
[Buy a copy »](https://global.oup.com/academic/product/web-browser-engineering-9780198913863)

*Web Browser Engineering* is now available.
[Buy a copy »](https://global.oup.com/academic/product/web-browser-engineering-9780198913863)

* [Installing Skia and SDL](https://browser.engineering/visual-effects.html#installing-skia-and-sdl)
* [SDL Creates the Window](https://browser.engineering/visual-effects.html#sdl-creates-the-window)
* [Surfaces
  and Pixels](https://browser.engineering/visual-effects.html#surfaces-and-pixels)
* [Rasterizing with Skia](https://browser.engineering/visual-effects.html#rasterizing-with-skia)
* [Browser
  Compositing](https://browser.engineering/visual-effects.html#browser-compositing)
* [Transparency](https://browser.engineering/visual-effects.html#transparency)
* [Blending and Stacking](https://browser.engineering/visual-effects.html#blending-and-stacking)
* [Compositing Pixels](https://browser.engineering/visual-effects.html#compositing-pixels)
* [Clipping and Masking](https://browser.engineering/visual-effects.html#clipping-and-masking)
* [Optimizing Surface Use](https://browser.engineering/visual-effects.html#optimizing-surface-use)
* [Summary](https://browser.engineering/visual-effects.html#summary)
* [Outline](https://browser.engineering/visual-effects.html#outline)
* [Exercises](https://browser.engineering/visual-effects.html#exercises)

Right now our browser can only draw colored rectangles and
text—pretty boring! Real browsers support all kinds of *visual
effects* that change how pixels and colors blend together. To
implement those effects, and also make our browser faster, we’ll need
control over *surfaces*, the key low-level feature behind fast
scrolling, visual effects, animations, and many other browser
capabilities. To get that control, we’ll also switch to using the Skia
graphics library and even take a peek under its hood.

Installing Skia and SDL
=======================

While Tkinter is great for basic shapes and input handling, it
doesn’t give us control over surfacesThat’s because Tk, the graphics library that Tkinter uses,
dates from the early 1990s, before high-performance graphics cards and
GPUs became widespread. and lacks implementations of most
visual effects. Implementing them ourselves would be fun, but it’s
outside the scope of this book, so we need a new graphics library. Let’s
use [Skia](https://skia.org), the library that Chromium uses.
Unlike Tkinter, Skia doesn’t handle inputs or create graphical windows,
so we’ll pair it with the [SDL](https://www.libsdl.org/) GUI
library. Beyond new capabilities, switching to Skia will allow us to
control graphics and rasterization at a lower level.

Start by installing [Skia](https://kyamagu.github.io/skia-python/) and [SDL](https://pypi.org/project/PySDL2/):The 1st printed
edition of *Web Browser Engineering* used an older Skia version,
and if you’re reading this website together with a printed
1st edition, make sure to check the [porting notes](https://browser.engineering/porting.html) to
understand the differences between the two.

```
python3 -m pip install skia-python pysdl2 pysdl2-dll PyOpenGL
```

As elsewhere in this book, you may need to install the
`pip` package first, or use your IDE’s package installer. If
you’re on Linux, you’ll need to install additional dependencies, like
OpenGL and fontconfig. Also, you may not be able to install
`pysdl2-dll`; if so, you’ll need to find SDL in your system
package manager instead. Consult the [`skia-python`](https://kyamagu.github.io/skia-python/)
and [`pysdl2`](https://pypi.org/project/PySDL2/)
web pages for more details.

Once installed, remove the `tkinter` imports from browser
and replace them with these:

```
import ctypes
import sdl2
import skia
```

The `ctypes` module is a standard part of Python; we’ll
use it to convert between Python and C types. If any of these imports
fail, check that Skia and SDL were installed correctly.

The [`<canvas>`](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/canvas)
HTML element provides a JavaScript API that is similar to Skia and
Tkinter. Combined with [WebGL](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API),
it’s possible to implement basically all of SDL and Skia in JavaScript.
Alternatively, one can [compile Skia](https://skia.org/docs/user/modules/canvaskit/) to
[WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly)
to do the same.

SDL Creates the Window
======================

The first big task is to switch to using SDL to create the window and
handle events. The main loop of the browser first needs some boilerplate
to get SDL started:

```
if __name__ == "__main__":
    sdl2.SDL_Init(sdl2.SDL_INIT_EVENTS)
    browser = Browser()
    browser.new_tab(URL(sys.argv[1]))
    # ...
```

Next, we need to create an SDL window, instead of a Tkinter window,
inside the `Browser`. Here’s the SDL incantation:

```
class Browser:
    def __init__(self):
        self.sdl_window = sdl2.SDL_CreateWindow(b"Browser",
            sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED,
            WIDTH, HEIGHT, sdl2.SDL_WINDOW_SHOWN)
```

Now that we’ve created a window, we need to handle events sent to it.
SDL doesn’t have a `mainloop` or `bind` method; we
have to implement it ourselves:

```
def mainloop(browser):
    event = sdl2.SDL_Event()
    while True:
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                browser.handle_quit()
                sdl2.SDL_Quit()
                sys.exit()
            # ...
```

The details of `ctypes` and `PollEvent` aren’t
too important here, but note that `SDL_QUIT` is an event,
sent when the user closes the last open window. The
`handle_quit` method it calls just cleans up the window
object:

```
class Browser:
    def handle_quit(self):
        sdl2.SDL_DestroyWindow(self.sdl_window)
```

Call `mainloop` in place of
`tkinter.mainloop`:

```
if __name__ == "__main__":
    # ...
    mainloop(browser)
```

In place of all the `bind` calls in the
`Browser` constructor, we can just directly call methods for
various types of events, like clicks, typing, and so on. The SDL syntax
looks like this:

```
def mainloop(browser):
    while True:
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            # ...
            elif event.type == sdl2.SDL_MOUSEBUTTONUP:
                browser.handle_click(event.button)
            elif event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_RETURN:
                    browser.handle_enter()
                elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                    browser.handle_down()
            elif event.type == sdl2.SDL_TEXTINPUT:
                browser.handle_key(event.text.text.decode('utf8'))
```

I’ve changed the signatures of the various event handler methods. For
example, the `handle_click` method is now passed a
`MouseButtonEvent` object, which thankfully contains
`x` and `y` coordinates, while the
`handle_enter` and `handle_down` methods aren’t
passed any argument at all, because we don’t use that argument anyway.
You’ll need to change the `Browser` methods’ signatures to
match.

SDL is most popular for making games. Their site lists [a selection of books](https://wiki.libsdl.org/Books) about game
programming in SDL.

Surfaces and Pixels
===================

Let’s peek under the hood of these SDL calls. When we create an SDL
window, we’re asking SDL to allocate a *surface*, a chunk of
memory representing the pixels on the screen.A surface may or may not be
bound to the physical pixels on the screen via a window, and there can
be many surfaces. A *canvas* is an API interface that allows you
to draw into a surface with higher-level commands such as for rectangles
or text. Our browser uses separate Skia and SDL surfaces for simplicity,
but in a highly optimized browser, minimizing the number of surfaces is
important for good performance. Creating and managing
surfaces is going to be the big focus of this chapter. On today’s large
screens, surfaces take up a lot of memory, so handling surfaces well is
essential to good browser performance.

A *surface* is a representation of a graphics buffer into
which you can draw *pixels* (bits representing colors). We
implicitly created an SDL surface when we created an SDL window; let’s
also create a surface for Skia to draw to:

```
class Browser:
    def __init__(self):
        self.root_surface = skia.Surface.MakeRaster(
            skia.ImageInfo.Make(
                WIDTH, HEIGHT,
                ct=skia.kRGBA_8888_ColorType,
                at=skia.kUnpremul_AlphaType))
```

Each pixel has a color. Note the `ct` argument, meaning
“color type”, which indicates that each pixel of this surface should be
represented as *r*ed, *g*reen, *b*lue, and
*a*lpha values, each of which should take up eight bits. In other
words, pixels are basically defined like so:

```
class Pixel:
    def __init__(self, r, g, b, a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a
```

This `Pixel` definition is an illustrative example, not
actual code in our browser. It’s standing in for somewhat more complex
code within SDL and Skia themselves.Skia actually represents colors as 32-bit integers, with
the most significant byte representing the alpha value (255 meaning
opaque and 0 meaning transparent) and the next three bytes representing
the red, green, and blue color channels.

Defining colors via red, green, and blue components is fairly
standardIt’s formally
known as the [sRGB color
space](https://en.wikipedia.org/wiki/SRGB), and it dates back to [CRT (cathode-ray
tube) displays](https://en.wikipedia.org/wiki/Cathode-ray_tube), which had a pretty limited *gamut* of
expressible colors. New technologies like LCD, LED, and OLED can display
more colors, so CSS now includes [syntax](https://drafts.csswg.org/css-color-4/) for expressing
these new colors. Still, all color spaces have a limited gamut of
expressible colors. and corresponds to how computer
screens work.Actually,
some screens contain [lights
besides red, green, and blue](https://geometrian.com/programming/reference/subpixelzoo/index.php), including white, cyan, or yellow.
Moreover, different screens can use slightly different reds, greens, or
blues; professional color designers typically have to [calibrate their
screen](https://en.wikipedia.org/wiki/Color_calibration) to display colors accurately. For the rest of us, the
software still communicates with the display in terms of standard red,
green, and blue colors, and the display hardware converts them to
whatever pixels it uses. For example, in CSS, we refer to
arbitrary colors with a hash character and six hex digits, like
`#ffd700`, with two digits each for red, green, and
blue:Alpha is implicitly
255, meaning opaque, in this case.

```
def parse_color(color):
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return skia.Color(r, g, b)
```

The colors we’ve seen so far can just be specified in terms of this
syntax:

```
NAMED_COLORS = {
    "black": "#000000",
    "white": "#ffffff",
    "red":   "#ff0000",
    # ...
}

def parse_color(color):
    # ...
    elif color in NAMED_COLORS:
        return parse_color(NAMED_COLORS[color])
    else:
        return skia.ColorBLACK
```

You can add more named colors from [the
list](https://developer.mozilla.org/en-US/docs/Web/CSS/named-color) as you come across them; the demos in this book use
`blue`, `green`, `lightblue`,
`lightgreen`, `orange`, `orangered`,
and `gray`. Note that unsupported colors are interpreted as
black, so that at least something is drawn to the screen.This is not the
standards-required behavior—the invalid value should just not
participate in styling, so an element styled with an unknown color might
inherit a color other than black—but I’m doing it as a
convenience.

Let’s now use our understanding of surfaces and colors to copy from
the Skia surface, where we will draw the chrome and page content, to the
SDL surface, which actually appears on the screen. This is a little
hairy, because we are moving data between two low-level libraries, but
really we’re just copying pixels from one place to another. First, get
the sequence of bytes representing the Skia surface:

```
class Browser:
    def draw(self):
        # ...
        skia_image = self.root_surface.makeImageSnapshot()
        skia_bytes = skia_image.tobytes()
```

Next, we need to copy the data to an SDL surface. This requires
telling SDL what order the pixels are stored in and your computer’s [endianness](https://en.wikipedia.org/wiki/Endianness):

```
class Browser:
    def __init__(self):
        if sdl2.SDL_BYTEORDER == sdl2.SDL_BIG_ENDIAN:
            self.RED_MASK = 0xff000000
            self.GREEN_MASK = 0x00ff0000
            self.BLUE_MASK = 0x0000ff00
            self.ALPHA_MASK = 0x000000ff
        else:
            self.RED_MASK = 0x000000ff
            self.GREEN_MASK = 0x0000ff00
            self.BLUE_MASK = 0x00ff0000
            self.ALPHA_MASK = 0xff000000
```

The `CreateRGBSurfaceFrom` method then wraps the data in
an SDL surface (without copying the bytes):

```
class Browser:
    def draw(self):
        # ...
        depth = 32 # Bits per pixel
        pitch = 4 * WIDTH # Bytes per row
        sdl_surface = sdl2.SDL_CreateRGBSurfaceFrom(
            skia_bytes, WIDTH, HEIGHT, depth, pitch,
            self.RED_MASK, self.GREEN_MASK,
            self.BLUE_MASK, self.ALPHA_MASK)
```

Finally, we draw all this pixel data on the window itself by blitting
(copying) it from `sdl_surface` to `sdl_window`’s
surface:Note that since
Skia and SDL are C++ libraries, they are not always consistent with
Python’s garbage collection system. So the link between the output of
`tobytes` and `sdl_window` is not guaranteed to be
kept consistent when `skia_bytes` is garbage-collected. The
SDL surface could be left pointing at a bogus piece of memory, leading
to memory corruption or a crash. The code here is correct because all of
these are local variables that are garbage-collected together, but if
not you need to be careful to keep all of them alive at the same
time.

```
class Browser:
    def draw(self):
        # ...
        rect = sdl2.SDL_Rect(0, 0, WIDTH, HEIGHT)
        window_surface = sdl2.SDL_GetWindowSurface(self.sdl_window)
        # SDL_BlitSurface is what actually does the copy.
        sdl2.SDL_BlitSurface(sdl_surface, rect, window_surface, rect)
        sdl2.SDL_UpdateWindowSurface(self.sdl_window)
```

So now we can copy from the Skia surface to the SDL window. One last
step: we have to draw the browser to the Skia surface.

We take it for granted, but color standards like [CIELAB](https://en.wikipedia.org/wiki/CIELAB_color_space)
derive from attempts to [reverse-engineer
human vision](https://en.wikipedia.org/wiki/Opponent_process). Screens use red, green, and blue color channels to
match the three types of [cone cells](https://en.wikipedia.org/wiki/Cone_cell) in a human
eye. These cone cells vary between people: some have [more](https://en.wikipedia.org/wiki/Tetrachromacy#Humans) and
some [fewer](https://en.wikipedia.org/wiki/Color_blindness)
(typically an inherited condition carried on the X chromosome).
Moreover, different people have different ratios of cone types and those
cone types use different protein structures that vary in the exact
frequency of green, red, and blue that they respond to. The study of
color thus combines software, hardware, chemistry, biology, and
psychology.

Rasterizing with Skia
=====================

We want to draw text, rectangles, and so on to the Skia surface. This
step—coloring in the pixels of a surface to draw shapes on it—is called
“rasterization” and is one important task of a graphics library. In
Skia, rasterization happens via a *canvas* API. A canvas is just
an object that draws to a particular surface:

```
class Browser:
    def draw(self, canvas, offset):
        # ...
        canvas = self.root_surface.getCanvas()
        # ...
```

Let’s change the various classes to use Skia’s raster APIs.

The first thing we need to do is change the `rect` field
into a Skia `Rect` object. Get rid of the old
`Rect` class that was introduced in [Chapter 7](https://browser.engineering/chrome.html) in favor of `skia.Rect`.
Everywhere that a `Rect` was constructed, instead put
`skia.Rect.MakeLTRB` (for “make left-top-right-bottom”) or
`MakeXYWH` (for “make *x*-*y*-width-height”).
Everywhere that the sides of the rectangle (e.g., `left`)
were checked, replace them with the corresponding function on a Skia
`Rect` (e.g., `left()`). Also replace calls to
`containsPoint` with Skia’s `contains`.

For `DrawText` and `DrawLine` in particular,
it’s:

```
class DrawText:
    def __init__(self, x1, y1, text, font, color):
        self.rect = skia.Rect.MakeLTRB(
            x1, y1,
            x1 + font.measureText(text),
            y1 - font.getMetrics().fAscent \
                + font.getMetrics().fDescent)
        # ...

class DrawLine:
    def __init__(self, x1, y1, x2, y2, color, thickness):
        self.rect = skia.Rect.MakeLTRB(x1, y1, x2, y2)
        # ...
```

Our browser’s drawing commands will need to invoke Skia methods on
this canvas. To draw a line, you use Skia’s `Path`
object:Consult the [Skia](https://skia.org) and [skia-python](https://kyamagu.github.io/skia-python/)
documentation for more on the Skia API.

```
class DrawLine:
    def execute(self, scroll, canvas):
        path = skia.Path().moveTo(
            self.rect.left(), self.rect.top() - scroll) \
                .lineTo(self.rect.right(),
                    self.rect.bottom() - scroll)
        paint = skia.Paint(
            Color=parse_color(self.color),
            StrokeWidth=self.thickness,
            Style=skia.Paint.kStroke_Style,
        )
        canvas.drawPath(path, paint)
```

Note the steps involved here. We first create a `Path`
object, and then call `drawPath` to actually draw this path
to the canvas. This `drawPath` call takes a second argument,
`paint`, which defines how to actually perform this drawing.
We specify the color, but we also need to specify that we want to draw a
line *along* the path, instead of filling in the interior of the
path, which is the default. To do that we set the style to “stroke”, a
standard term referring to drawing along the border of some shape.The opposite is “fill”,
meaning filling in the interior of the shape.

We do something similar to draw text using
`drawString`:

```
class DrawText:
    def execute(self, scroll, canvas):
        paint = skia.Paint(
            AntiAlias=True,
            Color=parse_color(self.color),
        )
        baseline = self.rect.top() - scroll \
            - self.font.getMetrics().fAscent
        canvas.drawString(self.text, float(self.rect.left()),
            baseline, self.font, paint)
```

Note again that we create a `Paint` object identifying the
color and asking for anti-aliased text.“Anti-alias”ing just means
drawing some semi-transparent pixels to better approximate the shape of
the text. This is important when drawing shapes with fine details, like
text, but is less important when drawing large shapes like rectangles
and lines. We don’t specify the “style” because we want to
fill the interior of the text, the default.

Finally, for drawing rectangles you use `drawRect`:

```
class DrawRect:
    def execute(self, scroll, canvas):
        paint = skia.Paint(
            Color=parse_color(self.color),
        )
        canvas.drawRect(self.rect.makeOffset(0, -scroll), paint)
```

To create an outline, draw a rectangle but set the `Style`
parameter of the `Paint` to `Stroke_Style`:

```
class DrawOutline:
    def execute(self, scroll, canvas):
        paint = skia.Paint(
            Color=parse_color(self.color),
            StrokeWidth=self.thickness,
            Style=skia.Paint.kStroke_Style,
        )
        canvas.drawRect(self.rect.makeOffset(0, -scroll), paint)
```

Since we’re replacing Tkinter with Skia, we are also replacing
`tkinter.font`. In Skia, a font object has two pieces: a
`Typeface`, which is a type family with a certain weight,
style, and width; and a `Font`, which is a
`Typeface` at a particular size. It’s the
`Typeface` that contains data and caches, so that’s what we
need to cache:

```
def get_font(size, weight, style):
    key = (weight, style)
    if key not in FONTS:
        if weight == "bold":
            skia_weight = skia.FontStyle.kBold_Weight
        else:
            skia_weight = skia.FontStyle.kNormal_Weight
        if style == "italic":
            skia_style = skia.FontStyle.kItalic_Slant
        else:
            skia_style = skia.FontStyle.kUpright_Slant
        skia_width = skia.FontStyle.kNormal_Width
        style_info = \
            skia.FontStyle(skia_weight, skia_width, skia_style)
        font = skia.Typeface('Arial', style_info)
        FONTS[key] = font
    return skia.Font(FONTS[key], size)
```

Our browser also needs font metrics and measurements. In Skia, these
are provided by the `measureText` and `getMetrics`
methods. Let’s start with `measureText` replacing all calls
to `measure`. For example, in the `paint` method
in `InputLayout`, we must do:

```
class InputLayout:
    def paint(self):
        if self.node.is_focused:
            cx = self.x + self.font.measureText(text)
            # ...
```

There are `measure` calls in several other layout objects
(both in `paint` and `layout`), in
`DrawText`, in the `draw` method on
`Chrome`, in the `text` method in
`BlockLayout`, and in the `layout` method in
`TextLayout`. Update all of them to use
`measureText`.

Also, in the `layout` method of `LineLayout`
and in `DrawText` we make calls to the `metrics`
method on fonts. In Skia, this method is called `getMetrics`,
and to get the ascent and descent we need the `fAscent` and
`fDescent` fields on its result.

Importantly, in Skia the ascent needs to be negated. In Skia, ascent
and descent are positive if they go downward and negative if they go
upward, so ascents will normally be negative, the opposite of Tkinter.
There’s no analog for the `linespace` field that Tkinter
provides, but you can use descent minus ascent instead:

```
def linespace(font):
    metrics = font.getMetrics()
    return metrics.fDescent - metrics.fAscent
```

You should now be able to run the browser again. It should look and
behave just as it did in previous chapters, and it might feel faster on
complex pages, because Skia and SDL are in general faster than Tkinter.
If the transition felt easy—well, that’s one of the benefits to
abstracting over the drawing backend using a display list!

Finally, Skia also provides some new features. For example, Skia has
native support for rounded rectangles via `RRect` objects. We
can implement that by converting `DrawRect` to
`DrawRRect`:

```
class DrawRRect:
    def __init__(self, rect, radius, color):
        self.rect = rect
        self.rrect = skia.RRect.MakeRectXY(rect, radius, radius)
        self.color = color

    def execute(self, scroll, canvas):
        paint = skia.Paint(
            Color=parse_color(self.color),
        )
        canvas.drawRRect(self.rrect, paint)
```

Then we can draw these rounded rectangles for backgrounds:

```
class BlockLayout:
    def paint(self):
        if bgcolor != "transparent":
            radius = float(
                self.node.style.get(
                    "border-radius", "0px")[:-2])
            cmds.append(DrawRRect(
                self.self_rect(), radius, bgcolor))
```

With that, [this
example](https://browser.engineering/examples/example11-rounded-background.html):Note that
the example listed here, in common with other examples present in the
book, accesses a local resource (a CSS file in this case) that is also
present on [browser.engineering](https://browser.engineering/).

```
<link rel=stylesheet href="example11-longword.css">
<div>
Background is rounded
</div>
```

will round the corners of its background (see Figure 1).

![Figure 1: Example of a rounded background.](https://browser.engineering/examples/example11-rounded-background.png)

Figure 1: Example of a rounded
background.

Similar changes should be made to `InputLayout`. New
shapes, like rounded rectangles, is one way that Skia is a more advanced
rasterization library than Tk. More broadly, since Skia is also used by
Chromium, we know it has fast, built-in support for all of the shapes we
might need in a browser.

[Font
rasterization](https://en.wikipedia.org/wiki/Font_rasterization) is surprisingly deep, with techniques such as [subpixel
rendering](https://en.wikipedia.org/wiki/Subpixel_rendering) and [hinting](https://en.wikipedia.org/wiki/Font_hinting) used to
make fonts look better on lower-resolution screens. These techniques are
much less necessary on [high-pixel-density](https://en.wikipedia.org/wiki/Pixel_density)
screens, though. It’s likely that all screens will eventually be
high-density enough to retire these techniques.

Browser Compositing
===================

Skia and SDL have just made our browser more complex, but the
low-level control offered by these libraries is important because it
allows us to optimize common interactions like scrolling.

So far, any time the user scrolled a web page, we had to clear the
canvas and re-raster everything on it from scratch. This is
inefficient—we’re drawing the same pixels, just in a different place.
When the context is complex or the screen is large, rastering too often
produces a visible slowdown and drains laptop and mobile batteries. Real
browsers optimize scrolling using a technique I’ll call *browser
compositing*: drawing the whole web page to a hidden surface, and
only copying the relevant pixels to the window itself.

To implement this, we’ll need two new Skia surfaces: a surface for
browser chrome and a surface for the current `Tab`’s
contents. We’ll only need to re-raster the `Tab` surface if
page contents change, but not when (say) the user types into the address
bar. And we can scroll the `Tab` without any raster at all—we
just copy a different part of the current `Tab` surface to
the screen. Let’s call those surfaces `chrome_surface` and
`tab_surface`:We could even use a different surface for each
`Tab`, but real browsers don’t do this, since each surface
uses up a lot of memory, and typically users don’t notice the small
raster delay when switching tabs.

```
class Browser:
    def __init__(self):
        # ...
        self.chrome_surface = skia.Surface(
            WIDTH, math.ceil(self.chrome.bottom))
        self.tab_surface = None
```

I’m not explicitly creating `tab_surface` right away,
because we need to lay out the page contents to know how tall the
surface needs to be.

We’ll also need to split the browser’s `draw` method into
three parts:

* `raster_tab` will raster the page to the
  `tab_surface`;
* `raster_chrome` will raster the browser chrome to the
  `chrome_surface`;
* `draw` will composite the chrome and tab surfaces and
  copy the result from Skia to SDL.It might seem wasteful to copy from the chrome and tab
  surfaces to an intermediate Skia surface, instead of directly to the SDL
  surface. It is, but skipping that copy requires a lot of tricky
  low-level code. In [Chapter 13](https://browser.engineering/animations.html) we’ll avoid
  this copy in a different, better way.

Let’s start by doing the split:

```
class Browser:
    def raster_tab(self):
        canvas = self.tab_surface.getCanvas()
        canvas.clear(skia.ColorWHITE)
        # ...

    def raster_chrome(self):
        canvas = self.chrome_surface.getCanvas()
        canvas.clear(skia.ColorWHITE)
        # ...

    def draw(self):
        canvas = self.root_surface.getCanvas()
        canvas.clear(skia.ColorWHITE)
        # ...
```

Since we didn’t create the `tab_surface` on startup, we
need to create it at the top of `raster_tab`:For a very big web page,
`tab_surface` can be much larger than the size of the SDL
window, and therefore take up a very large amount of memory. We’ll
ignore that, but a real browser would only paint and raster surface
content up to a certain distance from the visible region, and
re-paint/raster as the user scrolls.

```
import math

class Browser:
    def raster_tab(self):
        tab_height = math.ceil(
            self.active_tab.document.height + 2*VSTEP)

        if not self.tab_surface or \
                tab_height != self.tab_surface.height():
            self.tab_surface = skia.Surface(WIDTH, tab_height)

        # ...
```

Note that we need to recreate the tab surface if the page’s height
changes. The way we compute the page bounds here, based on the layout
tree’s height, would be incorrect if page elements could stick out below
(or to the right) of their parents—but our browser doesn’t support any
features like that.

Next, `draw` should copy from the chrome and tab surfaces
to the root surface. Moreover, we need to translate the
`tab_surface` down by `chrome.bottom` and up by
`scroll`, and clip it to just the area of the window that
doesn’t overlap the browser chrome:

```
class Browser:
    def draw(self):
        # ...
        
        tab_rect = skia.Rect.MakeLTRB(
            0, self.chrome.bottom, WIDTH, HEIGHT)
        tab_offset = self.chrome.bottom - self.active_tab.scroll
        canvas.save()
        canvas.clipRect(tab_rect)
        canvas.translate(0, tab_offset)
        self.tab_surface.draw(canvas, 0, 0)
        canvas.restore()

        chrome_rect = skia.Rect.MakeLTRB(
            0, 0, WIDTH, self.chrome.bottom)
        canvas.save()
        canvas.clipRect(chrome_rect)
        self.chrome_surface.draw(canvas, 0, 0)
        canvas.restore()

        # ...
```

Note the `draw` calls: these copy the
`tab_surface` and `chrome_surface` to the
`canvas`, which is bound to `root_surface`. The
`clipRect` and `translate` calls make sure we copy
the right parts.

Finally, everywhere in `Browser` that we call
`draw`, we now need to call either `raster_tab` or
`raster_chrome` first. For example, in
`handle_click`, we do this:

```
class Browser:
    def handle_click(self, e):
        if e.y < self.chrome.bottom:
            # ...
            self.raster_chrome()
        else:
            # ...
            self.raster_tab()
        self.draw()
```

Notice how we don’t redraw the chrome when only the tab changes, and
vice versa. Likewise, in `handle_down`, we don’t need to call
`raster_tab` at all, since scrolling doesn’t change the
page.

However, clicking on a web page can cause it to navigate to a new
one, so we do need to detect that and raster the browser chrome if the
URL changed:

```
class Browser:
    def handle_click(self, e):
        if e.y < self.chrome.bottom:
            # ...
        else:
            # ...
            url = self.active_tab.url
            tab_y = e.y - self.chrome.bottom
            self.active_tab.click(e.x, tab_y)
            if self.active_tab.url != url:
                self.raster_chrome()
            self.raster_tab()
```

We also have some related changes in `Tab`. Let’s rename
`Tab`’s `draw` method to `raster`. In
it, we no longer need to pass around the scroll offset to the
`execute` methods, or account for `chrome.bottom`,
because we always draw the whole tab to the tab surface:

```
class Tab:
    def raster(self, canvas):
        for cmd in self.display_list:
            cmd.execute(canvas)
```

Likewise, we can remove the `scroll` parameter from each
drawing command’s `execute` method:

```
class DrawRect:
    def execute(self, canvas):
        paint = skia.Paint(
            Color=parse_color(self.color),
        )
        canvas.drawRect(self.rect, paint)
```

Our browser now uses composited scrolling, making scrolling faster
and smoother, all because we are now using a mix of intermediate
surfaces to store already-rastered content and avoid re-rastering unless
the content has actually changed.

Real browsers allocate new surfaces for various different situations,
such as implementing accelerated overflow scrolling and animations of
certain CSS properties such as [transform](https://developer.mozilla.org/en-US/docs/Web/CSS/transform)
and opacity that can be done without raster. They also allow scrolling
arbitrary HTML elements via [`overflow: scroll`](https://developer.mozilla.org/en-US/docs/Web/CSS/overflow)
in CSS. Basic scrolling for DOM elements is very similar to what we’ve
just implemented. But implementing it in its full generality, and with
excellent performance, is *extremely* challenging. Scrolling may
well be the single most complicated feature in a browser rendering
engine. The corner cases and subtleties involved are almost endless.

Transparency
============

Drawing shapes quickly is already a challenge, but with multiple
shapes there’s an additional question: what color should the pixel be
when two shapes overlap? So far, our browser has only handled opaque
shapes,It also hasn’t
considered subpixel geometry or anti-aliasing, which also rely on color
mixing. and the answer has been simple: take the color of
the top shape. But now we need more nuance.

Consider partially transparent colors in CSS. These use a hex color
with eight hex digits, with the last two indicating the level of
transparency. For example, the color `#00000080` is 50%
transparent black. Over a white background, that looks gray, but over an
orange background it looks like Figure 2.

Test

Figure 2: Example of black semi-transparent text blending into an
orange background.

Note that the text is a kind of dark orange, because its color is a
mix of 50% black and 50% orange. Many objects in the real world are
partially transparent: frosted glass, clouds, or colored paper, for
example. Looking through one, you see multiple colors *blended*
together. That’s also why computer screens work: the red, green, and
blue lights [blend
together](https://en.wikipedia.org/wiki/Color_mixing) and appear to our eyes as another color. Designers use this
effectMostly. Some more
advanced blending modes on the web are difficult, or perhaps impossible,
in real-world physics. in overlays, shadows, and tooltips,
so our browser needs to support color mixing.

Skia supports this kind of transparency by setting the “alpha” field
on the parsed color:

```
def parse_color(color):
    # ...
    elif color.startswith("#") and len(color) == 9:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        a = int(color[7:9], 16)
        return skia.Color(r, g, b, a)
    # ...
```

Check that your browser renders dark-orange text for the example
above. That shows that it’s actually mixing the black color with the
existing orange color from the background.

However, there’s another, subtly different way to create transparency
with CSS. Here, 50% transparency is applied to the whole element using
the `opacity` property, as in Figure 3.

Test

Figure 3: Example of black text on an orange background, then blended
semi-transparently into its ancestor.

Now the opacity applies to both the background and the text, so the
background is now a little lighter. But note that the text is now gray,
not dark orange. The black and orange pixels are no longer blended
together!

That’s because opacity introduces what CSS calls a [stacking
context](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Positioning/Understanding_z_index/The_stacking_context). Most of the details aren’t important right now, but the
order of operations is. In the first example, the black pixels were
first made transparent, then blended with the background. Thus, 50%
transparent black pixels were blending with orange pixels, resulting in
a dark-orange color. In the second example, the black pixels were first
blended with the background, then the result was made transparent. Thus,
fully black pixels replaced fully orange ones, resulting in just black
pixels, which were later made 50% transparent.

Applying blending in the proper order, as is necessary to implement
effects like `opacity`, requires more careful handling of
surfaces.

Mostly, elements [form
a stacking context](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Positioning/Understanding_z_index/The_stacking_context) because of CSS properties that have something to
do with layering (like `z-index`) or visual effects (like
`mix-blend-mode`). On the other hand, the
`overflow` property, which can make an element scrollable,
does not induce a stacking context, which I think was a mistake.While we’re at it, perhaps
scrollable elements should also be a [containing
block](https://developer.mozilla.org/en-US/docs/Web/CSS/Containing_block) for descendants. Otherwise, a scrollable element can have
non-scrolling children via properties like `position`. This
situation is very complicated to handle in real browsers.
The reason is that inside a modern browser, scrolling is done on the GPU
by offsetting two surfaces. Without a stacking context the browser might
(depending on the web page structure) have to move around multiple
independent surfaces with complex paint orders, in lockstep, to achieve
scrolling. Fixed- and sticky-positioned elements also form stacking
contexts because of their interaction with scrolling.

Blending and Stacking
=====================

To handle the order of operations properly, browsers apply blending
not to individual shapes but to a tree of surfaces (see Figure 4).
Conceptually, each shape is drawn to its own surface, and then blended
into its parent surface. Different structures of intermediate surfaces
create different visual effects.You can see a more detailed discussion of how the tree
structure affects the final image, and how that impacted the CSS
specifications, on [David
Baron’s blog](https://dbaron.org/log/20130306-compositing-blending). Rastering a web page requires a
bottom-up traversal of this conceptual tree: to raster a surface you
first need to raster its contents, including its child surfaces, and
then the contents need to be blended together into the parent.This tree of surfaces is an
implementation strategy and not something required by any specific web
API. However, the concept of a [*stacking
context*](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Positioning/Understanding_z_index/The_stacking_context) is related. A stacking context is technically a
mechanism to define groups and ordering during paint, and stacking
contexts need not correspond to a surface (e.g. ones created via [`z-index`](https://developer.mozilla.org/en-US/docs/Web/CSS/z-index)
do not). However, for ease of implementation, all visual effects in CSS
that generally require surfaces to implement are specified to go
hand-in-hand with a stacking context, so the tree of stacking contexts
is very related to the tree of surfaces.

![Figure 4: A rendered web page is actually the result of stacking and blending a series of different surfaces.](https://browser.engineering/im/visual-effects-surfaces.jpg)

Figure 4: A rendered web page is actually
the result of stacking and blending a series of different
surfaces.

To match this use pattern, in Skia, surfaces form a stack. You can
pus
... (truncated)