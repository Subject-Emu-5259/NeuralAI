Supporting Embedded Content | Web Browser Engineering

Supporting Embedded Content
===========================

[Twitter](https://twitter.com/browserbook) ·
[Blog](https://browserbook.substack.com/) ·
[Discussions](https://github.com/browserengineering/book/discussions)

Chapter 15 of [Web Browser Engineering](https://browser.engineering/index.html "Table of Contents").
[<](https://browser.engineering/accessibility.html "Previous chapter")
[>](https://browser.engineering/invalidation.html "Next chapter")

![The cover for Web Browser Engineering, published by Oxford University Press. Click the cover to buy a copy.](https://browser.engineering/im/cover.jpg)
[Buy a copy »](https://global.oup.com/academic/product/web-browser-engineering-9780198913863)

*Web Browser Engineering* is now available.
[Buy a copy »](https://global.oup.com/academic/product/web-browser-engineering-9780198913863)

* [Images](https://browser.engineering/embeds.html#images)
* [Embedded
  layout](https://browser.engineering/embeds.html#embedded-layout)
* [Modifying Image Sizes](https://browser.engineering/embeds.html#modifying-image-sizes)
* [Interactive Widgets](https://browser.engineering/embeds.html#interactive-widgets)
* [Iframe
  Rendering](https://browser.engineering/embeds.html#iframe-rendering)
* [Iframe
  Input Events](https://browser.engineering/embeds.html#iframe-input-events)
* [Iframe
  Scripts](https://browser.engineering/embeds.html#iframe-scripts)
* [Communicating Between
  Frames](https://browser.engineering/embeds.html#communicating-between-frames)
* [Isolation and Timing](https://browser.engineering/embeds.html#isolation-and-timing)
* [Summary](https://browser.engineering/embeds.html#summary)
* [Outline](https://browser.engineering/embeds.html#outline)
* [Exercises](https://browser.engineering/embeds.html#exercises)

While our browser can render complex styles, visual effects, and
animations, all of those apply basically just to text. Yet web pages
contain a variety of non-text *embedded content*, from images to
other web pages. Support for embedded content has powerful implications
for browser architecture, performance, security, and open information
access, and has played a key role throughout the web’s history.

Images
======

Images are certainly the most popular kind of embedded content on the
web,So it’s a little
ironic that images only make their appearance in Chapter 15 of this
book! It’s because Tkinter doesn’t support many image formats or proper
sizing and clipping, so I had to wait for the introduction of
Skia. dating back to [early
1993](http://1997.webhistory.org/www.lists/www-talk.1993q1/0182.html).This history is
also [the
reason behind](http://1997.webhistory.org/www.lists/www-talk.1993q1/0196.html) a lot of inconsistencies, like `src` versus
`href` or `img` versus
`image`. They’re included on web pages via the
`<img>` tag, which looks like this:

```
<img src="https://browser.engineering/im/hes.jpg">
```

This particular example renders as shown in Figure 1.

![Figure 1: A computer operator using the Hypertext Editing System in 1969. (Gregory Lloyd from Wikipedia, CC BY-SA 4.0 International.)](https://browser.engineering/im/hes.jpg)

Figure 1: A computer operator using the
Hypertext Editing System in 1969. (Gregory Lloyd from [Wikipedia](https://commons.wikimedia.org/wiki/File:HypertextEditingSystemConsoleBrownUniv1969.jpg),
[CC
BY-SA 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/deed.en).)

Luckily, implementing images isn’t too hard, so let’s just get
started. There are four steps to displaying images in our browser:

1. Download the image from a URL.
2. Decode the image into a buffer in memory.
3. Lay the image out on the page.
4. Paint the image in the display list.

Let’s start with downloading images from a URL. Naturally, that
happens over HTTP, which we already have a `request` function
for. However, while all of the content we’ve downloaded so far—HTML,
CSS, and JavaScript—has been textual, images typically use binary data
formats. We’ll need to extend `request` to support binary
data.

The change is pretty minimal: instead of passing the `"r"`
flag to `makefile`, pass a `"b"` flag indicating
binary mode:

```
class URL:
    def request(self, referrer, payload=None):
        # ...
        response = s.makefile("b")
        # ...
```

Now every time we read from `response`, we will get
`bytes` of binary data, not a `str` with textual
data, so we’ll need to change some HTTP parser code to explicitly
`decode` the data:

```
class URL:
    def request(self, referrer, payload=None):
        # ...
        statusline = response.readline().decode("utf8")
        # ...
        while True:
            line = response.readline().decode("utf8")
            # ...
        # ...
```

Note that I *didn’t* add a `decode` call when we
read the body; that’s because the body might actually be binary data,
and we want to return that binary data directly to the browser. Now,
every existing call to `request`, which wants textual data,
needs to `decode` the response. For example, in
`load`, you’ll want to do something like this:

```
class Tab:
    def load(self, url, payload=None):
        # ...
        headers, body = url.request(self.url, payload)
        body = body.decode("utf8", "replace")
        # ...
```

By passing `replace` as the second argument to
`decode`, I tell Python to replace any invalid characters by
a special � character instead of throwing an exception.

Make sure to make this change everywhere in your browser that you
call `request`, including inside
`XMLHttpRequest_send` and in several other places in
`load`.

When we download images, however, we *won’t* call
`decode`; we’ll just use the binary data directly.

```
class Tab:
    def load(self, url, payload=None):
        # ...
        images = [node
            for node in tree_to_list(self.nodes, [])
            if isinstance(node, Element)
            and node.tag == "img"]
        for img in images:
            src = img.attributes.get("src", "")
            image_url = url.resolve(src)
            assert self.allowed_request(image_url), \
                "Blocked load of " + str(image_url) + " due to CSP"
            header, body = image_url.request(url)
```

Once we’ve downloaded the image, we need to turn it into a Skia
`Image` object. That requires the following code:

```
class Tab:
    def load(self, url, payload=None):
        for img in images:
            # ...
            img.encoded_data = body
            data = skia.Data.MakeWithoutCopy(body)
            img.image = skia.Image.MakeFromEncoded(data)
```

There are two tricky steps here: the requested data is turned into a
Skia `Data` object using the `MakeWithoutCopy`
method, and then into an image with `MakeFromEncoded`.

Because we used `MakeWithoutCopy`, the `Data`
object just stores a reference to the existing `body` and
doesn’t own that data. That’s essential, because encoded image data can
be large—maybe megabytes—and copying that data wastes memory and time.
But that also means that the `data` will become invalid if
`body` is ever garbage-collected; that’s why I save the
`body` in an `encoded_data` field.This is a bit of a hack.
Perhaps a better solution would be to write the response directly into a
Skia `Data` object using the `writable_data` API.
That would require some refactoring of the rest of the browser which is
why I’m choosing to avoid it.

These download and decode steps can both fail; if that happens we’ll
load a “broken image” placeholder (I used [one from
Wikipedia](https://commons.wikimedia.org/wiki/File:Broken_Image.png)):

```
BROKEN_IMAGE = skia.Image.open("Broken_Image.png")

class Tab:
    def load(self, url, payload=None):
        for img in images:
            try:
                # ...
                assert img.image, \
                    "Failed to recognize image format for " + \
                        str(image_url)
            except Exception as e:
                print("Image", img.attributes.get("src", ""),
                    "crashed", e)
                img.image = BROKEN_IMAGE
```

Now that we’ve downloaded and saved the image, we need to use it.
That just requires calling Skia’s `drawImageRect`
function:

```
class DrawImage(PaintCommand):
    def __init__(self, image, rect):
        super().__init__(rect)
        self.image = image

    def execute(self, canvas):
        canvas.drawImageRect(self.image, self.rect)
```

The internals of `drawImageRect`, however, are a little
complicated and worth expanding on. Recall that the `Image`
object is created using a `MakeFromEncoded` method. That name
reminds us that the image we’ve downloaded isn’t raw image bytes. In
fact, all of the image formats you know—JPG, PNG, and the many more
obscure ones—encode the image data using various sophisticated
algorithms. The image therefore needs to be *decoded* before it
can be used.And with
much more complicated algorithms than just `utf8`
conversion.

Skia applies a variety of clever optimizations to decoding, such as
directly decoding the image to its eventual size and caching the decoded
image as long as possible.There’s also an [HTML
API](https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement/decoding) to control decoding, so that the web page author can indicate
when to pay that cost. That’s because raw image data can
be quite large:Decoding
costs both a lot of memory and also a lot of time, since just writing
out all of those bytes can take a big chunk of our render budget.
Optimizing image handling is essential to a performant
browser. a pixel is usually stored as 4 bytes, so a 12
megapixel camera (as you can find on phones these days) produces 48
megabytes of raw data for a single image.

Because image decoding can be so expensive, Skia also has several
algorithms available for decoding, some of which are faster but result
in a worse-looking image.Image formats like JPEG are also [*lossy*](https://en.wikipedia.org/wiki/Lossy_compression),
meaning that they don’t faithfully represent all of the information in
the original picture, so there’s a time/quality trade-off going on
before the file is saved. Typically these formats try to drop “noisy
details” that a human is unlikely to notice, just like different
resizing algorithms might. For example, there’s the fast,
simple “nearest neighbor” algorithm and the slower but higher-quality
“bilinear” or even “[Lanczos](https://en.wikipedia.org/wiki/Lanczos_resampling)”
algorithms.Specifically,
these algorithms decide how to decode an image when the image size and
the destination size are different and the image therefore needs to be
resized. The faster algorithms tend to result in choppier, more jagged
images.

To give web page authors control over this performance bottleneck,
there’s an [`image-rendering`](https://developer.mozilla.org/en-US/docs/Web/CSS/image-rendering)
CSS property that indicates which algorithm to use. Let’s add that as an
argument to `DrawImage`:The 1st printed edition of *Web Browser
Engineering* used an older API for image rendering quality,
described in the [porting
notes](https://browser.engineering/porting.html).

```
def parse_image_rendering(quality):
   if quality == "high-quality":
       return skia.SamplingOptions(skia.CubicResampler.Mitchell())
   elif quality == "crisp-edges":
       return skia.SamplingOptions(
           skia.FilterMode.kNearest, skia.MipmapMode.kNone)
   else:
       return skia.SamplingOptions(
           skia.FilterMode.kLinear, skia.MipmapMode.kLinear)

class DrawImage(PaintCommand):
    def __init__(self, image, rect, quality):
        # ...
        self.quality = parse_image_rendering(quality)

    def execute(self, canvas):
        canvas.drawImageRect(self.image, self.rect, self.quality)
```

But to talk about where this argument comes from, or more generally
to actually see downloaded images in our browser, we first need to add
images into our browser’s layout tree.

The HTTP `Content-Type` header lets the web server tell
the browser whether a document contains text or binary data. The header
contains a value called a [MIME
type](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types), such as `text/html`, `text/css`, and
`text/javascript` for HTML, CSS, and JavaScript;
`image/png` and `image/jpeg` for PNG and JPEG
images; and [many
others](https://www.iana.org/assignments/media-types/media-types.xhtml) for various font, video, audio, and data formats.“MIME” stands for Multipurpose
Internet Mail Extensions, and was originally intended for enumerating
all of the acceptable data formats for email attachments. These days the
loop has basically closed: most email clients are now “webmail” clients,
accessed through your browser, and most emails are now HTML, encoded
with the `text/html` MIME type, though typically there is
still a plain-text option. Interestingly, we didn’t need
to specify the image format in the code above. That’s because many image
formats start with [“magic
bytes”](https://www.netspi.com/blog/technical/web-application-penetration-testing/magic-bytes-identifying-common-file-formats-at-a-glance/); for example, PNG files always start with byte 137 followed
by the letters “PNG”. These magic bytes are often more reliable than
web-server-provided MIME types, so such “format sniffing” is common
inside browsers and their supporting libraries.

Embedded layout
===============

Based on your experience with prior chapters, you can probably guess
how to add images to our browser’s layout and paint process. We’ll need
to create an `ImageLayout` class; add a new
`image` case to `BlockLayout`’s
`recurse` method; and generate a `DrawImage`
command from `ImageLayout`’s `paint` method.

As we do this, you might recall doing something very similar for
`<input>` elements. In fact, text areas and buttons are
very similar to images: both are leaf nodes of the DOM, placed into
lines, affected by text baselines, and painting custom content.Images aren’t quite like
*text* because a text node is potentially an entire run of text,
split across multiple lines, while an image is an [atomic
inline](https://drafts.csswg.org/css-display-3/#atomic-inline). The other types of embedded content in this chapter are also
atomic inlines. Since they are so similar, let’s try to
reuse the same code for both.

Let’s split the existing `InputLayout` into a superclass
called `EmbedLayout`, containing most of the existing code,
and a new subclass with the input-specific code,
`InputLayout`:In a real browser, input elements are usually called
*widgets* because they have a lot of [special
rendering rules](https://html.spec.whatwg.org/multipage/rendering.html#widgets) that sometimes involve CSS.

```
class EmbedLayout:
    def __init__(self, node, parent, previous, frame):
        # ...

    def layout(self):
        self.zoom = self.parent.zoom
        self.font = font(self.node.style, self.zoom)
        if self.previous:
            space = self.previous.font.measureText(" ")
            self.x = \
                self.previous.x + space + self.previous.width
        else:
            self.x = self.parent.x
```

```
class InputLayout(EmbedLayout):
    def __init__(self, node, parent, previous):
        super().__init__(node, parent, previous)

    def layout(self):
        super().layout()
```

The idea is that `EmbedLayout` should provide common
layout code for all kinds of embedded content, while its subclasses like
`InputLayout` should provide the custom code for that type of
content. Different types of embedded content might have different widths
and heights, so that should happen in each subclass, as should the
definition of `paint`:

```
class InputLayout(EmbedLayout):
    def layout(self):
        # ...
        self.width = dpx(INPUT_WIDTH_PX, self.zoom)
        self.height = linespace(self.font)
        self.ascent = -self.height
        self.descent = 0

    def paint(self):
        # ...
```

`ImageLayout` can now inherit most of its behavior from
`EmbedLayout`, but take its width and height from the image
itself:

```
class ImageLayout(EmbedLayout):
    def __init__(self, node, parent, previous):
        super().__init__(node, parent, previous)

    def layout(self):
        super().layout()
        self.width = dpx(self.node.image.width(), self.zoom)
        self.img_height = dpx(self.node.image.height(), self.zoom)
        self.height = max(self.img_height, linespace(self.font))
        self.ascent = -self.height
        self.descent = 0
```

Notice that the height of the image depends on the font size of the
element. Though odd, this is how image layout actually works: a line
with a single, very small, image on it will still be tall enough to
contain text.In fact, a
page with only a single image and no text or CSS at all still has its
layout affected by a font—the default font. This is a common source of
confusion for web developers. In a real browser, it can be avoided by
forcing an image into a block or other layout mode via the
`display` CSS property. The underlying reason
for this is because, as a type of inline layout, images are designed to
flow along with related text, which means the bottom of the image should
line up with the [text
baseline](https://browser.engineering/text.html#text-of-different-sizes). That’s also why we save `img_height` in the
code above.

Also, in the code above I introduced new `ascent` and
`descent` fields on `EmbedLayout` subclasses. This
is meant to be used in `LineLayout` layout in place of the
existing layout code for ascent and descent. It also requires
introducing those fields on `TextLayout`:

```
class LineLayout:
    def layout(self):
        # ...
        max_ascent = max([-child.ascent 
                          for child in self.children])
        baseline = self.y + max_ascent

        for child in self.children:
            if isinstance(child, TextLayout):
                child.y = baseline + child.ascent / 1.25
            else:
                child.y = baseline + child.ascent
        max_descent = max([child.descent
                           for child in self.children])
        self.height = max_ascent + max_descent

class TextLayout:
    def layout(self):
        # ...
        self.ascent = self.font.getMetrics().fAscent * 1.25
        self.descent = self.font.getMetrics().fDescent * 1.25
```

Painting an image is also straightforward:

```
class ImageLayout(EmbedLayout):
    def paint(self):
        cmds = []
        rect = skia.Rect.MakeLTRB(
            self.x, self.y + self.height - self.img_height,
            self.x + self.width, self.y + self.height)
        quality = self.node.style.get("image-rendering", "auto")
        cmds.append(DrawImage(self.node.image, rect, quality))
        return cmds
```

Now we need to create `ImageLayout`s in
`BlockLayout`. Input elements are created in an
`input` method, so we create a largely similar
`image` method. But `input` is itself largely a
duplicate of `word`, so this would be a lot of duplication.
The only part of these methods that differs is the part that computes
the width of the new inline child; most of the rest of the logic is
shared.

Let’s instead refactor the shared code into new methods which
`text`, `image`, and `input` can call.
First, all of these methods need a font to determine how much spaceYes, this is how real browsers
do it too. to leave after the inline; let’s make a
function for that:

```
def font(style, zoom):
    weight = style["font-weight"]
    variant = style["font-style"]
    size = float(style["font-size"][:-2]) * 0.75
    font_size = dpx(size, zoom)
    return get_font(font_size, weight, variant)
```

There’s also shared code that handles line layout; let’s put that
into a new `add_inline_child` method. We’ll need to pass in
the HTML node, the element, and the layout class to instantiate (plus a
`word` parameter that’s just for
`TextLayout`s):

```
class BlockLayout:
    def add_inline_child(self, node, w, child_class, word=None):
        if self.cursor_x + w > self.x + self.width:
            self.new_line()
        line = self.children[-1]
        previous_word = line.children[-1] if line.children else None
        if word:
            child = child_class(node, word, line, previous_word)
        else:
            child = child_class(node, line, previous_word)
        line.children.append(child)
        self.cursor_x += w + \
            font(node.style, self.zoom).measureText(" ")
```

We can redefine `word` and `input` in a
satisfying way now:

```
class BlockLayout:
    def word(self, node, word):
        node_font = font(node.style, self.zoom)
        w = node_font.measureText(word)
        self.add_inline_child(node, w, TextLayout, word)

    def input(self, node):
        w = dpx(INPUT_WIDTH_PX, self.zoom)
        self.add_inline_child(node, w, InputLayout)
```

Adding `image` is easy:

```
class BlockLayout:
    def recurse(self, node):
            # ...
            elif node.tag == "img":
                self.image(node)
    
    def image(self, node):
        w = dpx(node.image.width(), self.zoom)
        self.add_inline_child(node, w, ImageLayout)
```

And of course, images also get the same inline layout mode as input
elements:

```
class BlockLayout:
    def layout_mode(self):
        # ...
        elif self.node.tag in ["input", "img"]:
            return "inline"

    def should_paint(self):
        return isinstance(self.node, Text) or \
            (self.node.tag not in \
                ["input", "button", "img"])
```

Now that we have `ImageLayout` nodes in our layout tree,
we’ll be painting `DrawImage` commands to our display list
and showing the image on the screen!

But what about our second output modality, screen readers? That’s
what the `alt` attribute is for. It works like this:

```
<img src="https://browser.engineering/im/hes.jpg"
  alt="An operator using the Hypertext Editing System in 1969">
```

Implementing this in `AccessibilityNode` is very easy:

```
class AccessibilityNode:
    def __init__(self, node):
        else:
            # ...
            elif node.tag == "img":
                self.role = "image"

    def build(self):
        # ...
        elif self.role == "image":
            if "alt" in self.node.attributes:
                self.text = "Image: " + self.node.attributes["alt"]
            else:
                self.text = "Image"
```

As we continue to implement new features for the web platform, we’ll
always need to think about how to make features work in multiple
modalities.

Videos are similar to images, but demand more bandwidth, time, and
memory; they also have complications like [digital
rights management (DRM)](https://en.wikipedia.org/wiki/Digital_rights_management). The `<video>` tag
addresses some of that, with built-in support for advanced video [*codecs*](https://en.wikipedia.org/wiki/Video_codec),In video, it’s called a
“codec”, but in images it’s called a “format”–go figure.
DRM, and hardware acceleration. It also provides media controls like a
play/pause button and volume controls.

Modifying Image Sizes
=====================

So far, an image’s size on the screen is its size in pixels, possibly
zoomed.Note that zoom
already may cause an image to render at a size different than its
regular size, even before introducing the features in this
section. But in fact it’s generally valuable for authors
to control the size of embedded content. There are a number of ways to
do this,For example, the
`width` and `height` CSS properties (not to be
confused with the `width` and `height`
attributes!), which we met in Exercise 6-2. but one way is
the special `width` and `height` attributes.Images have these mostly for
historical reasons: they were invented before CSS
existed.

If *both* those attributes are present, things are pretty
easy: we just read from them when laying out the element, both in
`image`:

```
class BlockLayout:
    def image(self, node):
        if "width" in node.attributes:
            w = dpx(int(node.attributes["width"]), self.zoom)
        else:
            w = dpx(node.image.width(), self.zoom)
        # ...
```

And in `ImageLayout`:

```
class ImageLayout(EmbedLayout):
    def layout(self):
        # ...
        width_attr = self.node.attributes.get("width")
        height_attr = self.node.attributes.get("height")
        image_width = self.node.image.width()
        image_height = self.node.image.height()

        if width_attr and height_attr:
            self.width = dpx(int(width_attr), self.zoom)
            self.img_height = dpx(int(height_attr), self.zoom)
        else:
            self.width = dpx(image_width, self.zoom)
            self.img_height = dpx(image_height, self.zoom)
        # ...
```

This works great, but it has a major flaw: if the ratio of
`width` to `height` isn’t the same as the
underlying image size, the image ends up stretched in weird ways.
Sometimes that’s on purpose but usually it’s a mistake. So browsers let
authors specify *just one* of `width` and
`height`, and compute the other using the image’s *aspect
ratio*.Despite it
being easy to implement, this feature of real web browsers only reached
all of them in 2021. Before that, developers resorted to things like the
[`padding-top`
hack](https://web.dev/aspect-ratio/#the-old-hack-maintaining-aspect-ratio-with-padding-top). Sometimes design oversights take a long time to
fix.

Implementing this aspect ratio tweak is easy:

```
class ImageLayout(EmbedLayout):
    # ...
    def layout(self):
        # ...
        aspect_ratio = image_width / image_height

        if width_attr and height_attr:
            # ...
        elif width_attr:
            self.width = dpx(int(width_attr), self.zoom)
            self.img_height = self.width / aspect_ratio
        elif height_attr:
            self.img_height = dpx(int(height_attr), self.zoom)
            self.width = self.img_height * aspect_ratio
        else:
            # ...
        # ...
```

Your browser should now be able to render the following [example
page](https://browser.engineering/examples/example15-img.html) correctly, as shown in Figure 2. When it’s scrolled down a bit
you should see what’s shown in Figure 3 (notice the different aspect
ratios). And scrolling to the end will show what appears in Figure 4,
including the “broken image” icon.

```
Original size:  <img src="/im/hes.jpg" alt="A computer operator ...">
<br>
Smaller: <img width=50 height=50 src="/im/hes.jpg">
<br>
Different aspect ratio:
<img width=50 height=100 src="/im/hes.jpg">
<br>
Larger:
<img width=1000 height=1000 src="/im/hes.jpg">
<br>
Larger with only width:
<img width=1000 src="/im/hes.jpg">
<br>
Smaller with only height:
<img height=50 src="/im/hes.jpg">
Broken image:
<img src="non-existent-image">
<script src="example15-img.js"></script>
<link rel="stylesheet" href="example15-img.css">
```

![Figure 2: Rendering of an example with images.](https://browser.engineering/examples/example15-img.png)

Figure 2: Rendering of an example with
images.

![Figure 3: Rendering of an example with images after scrolling to aspect-ratio differences.](https://browser.engineering/examples/example15-img-scroll1.png)

Figure 3: Rendering of an example with
images after scrolling to aspect-ratio differences.

![Figure 4: Rendering of an example with images after scrolling to a broken image icon.](https://browser.engineering/examples/example15-img-scroll2.png)

Figure 4: Rendering of an example with
images after scrolling to a broken image icon.

Our browser computes an aspect ratio from the loaded image
dimensions, but that’s not available before an image loads, which is a
problem in real browsers where images are loaded asynchronously and
where the image size can [respond
to](https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout/Responsive_Design) layout parameters. Not knowing the aspect ratio can cause the [layout to shift](https://web.dev/cls/) when the image loads,
which can be frustrating for users. The [`aspect-ratio`
property](https://developer.mozilla.org/en-US/docs/Web/CSS/aspect-ratio) is one way web pages can address this issue.

Interactive Widgets
===================

So far, our browser has two kinds of embedded content: images and
input elements. While both are important and widely used,As are variations like the [`<canvas>`](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/canvas)
element. Instead of loading an image from the network, JavaScript can
draw on a `<canvas>` element via an API. Unlike images,
`<canvas>` elements don’t have intrinsic sizes, but
besides that they are pretty similar in terms of layout.
they don’t offer quite the customizabilityThere’s actually [ongoing work](https://open-ui.org/) aimed at allowing web pages
to customize what input elements look like, and it builds on earlier
work supporting [custom
elements](https://developer.mozilla.org/en-US/docs/Web/Web_Components) and [forms](https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/attachInternals).
This problem is quite challenging, interacting with platform
independence, accessibility, scripting, and styling. and
flexibility that complex embedded content use cases like maps, PDFs,
ads, and social media controls require. So in modern browsers, these are
handled by *embedding one web page within another* using the
`<iframe>` element.Or via the `embed` and `object` tags,
for cases like PDFs. I won’t discuss those here.

Semantically, an `<iframe>` is similar to a
`Tab` inside a `Tab`—it has its own HTML document,
CSS, and scripts. And layout-wise, an `<iframe>` is a
lot like the `<img>` tag, with `width` and
`height` attributes. So implementing basic iframes just
requires handling these three significant differences:

* Iframes have *no browser chrome*. So any page navigation
  has to happen from within the page (either through an
  `<a>` element or a script), or as a side effect of
  navigation on the web page that *contains* the
  `<iframe>` element. Clicking on a link in an iframe
  also navigates the iframe, not the top-level page.
* Iframes can *share a rendering event loop*.For example, if an iframe has
  the same origin as the web page that embeds it, then scripts in the
  iframe can synchronously access the parent DOM. That means that it’d be
  basically impossible to put that iframe in a different thread or CPU
  process, and in practice it ends up in the same rendering event
  loop. In real browsers, [cross-origin](https://developer.mozilla.org/en-US/docs/Web/Security/Same-origin_policy)
  iframes are often “site isolated”, meaning that the iframe has its own
  CPU process for [security
  reasons](https://www.chromium.org/Home/chromium-security/site-isolation/). In our browser we’ll just make all iframes (even nested
  ones—yes, iframes can include iframes!) use the same rendering event
  loop.
* Cross-origin iframes are *script-isolated* from the
  containing page. That means that a script in the iframe [can’t
  access](https://developer.mozilla.org/en-US/docs/Web/Security/Same-origin_policy#cross-origin_script_api_access) the containing page’s variables or DOM, nor can scripts in
  the containing page access the iframe’s variables or DOM. Same-origin
  iframes, however, can.

We’ll get to these differences, but for now, let’s start working on
the idea of a `Tab` within a `Tab`. What we’re
going to do is split the `Tab` class into two pieces:
`Tab` will own the event loop and script environments,
`Frame`s will do the rest.

It’s good to plan out complicated refactors like this in some detail.
A `Tab` will:

* interface between the `Browser` and the
  `Frame`s to handle events;
* proxy communication between frames;
* kick off animation frames and rendering;
* paint and own the display list for all frames in the tab;
* construct and own the accessibility tree;
* commit to the browser thread.

And the new `Frame` class will:

* own the DOM, layout trees, and scroll offset for its HTML
  document;
* run style and layout on the its DOM and layout tree;
* implement loading and event handling (focus, hit testing, etc) for
  its HTML document.

Create these two classes and split the methods between them
accordingly.

Naturally, every `Frame` will need a reference to its
`Tab`; it’s also convenient to have access to the parent
frame and the corresponding `<iframe>` element:

```
class Frame:
    def __init__(self, tab, parent_frame, frame_element):
        self.tab = tab
        self.parent_frame = parent_frame
        self.frame_element = frame_element
        # ...
```

Now let’s look at how `Frame`s are created. The first
place is in `Tab`’s `load` method, which needs to
create the *root frame*:

```
class Tab:
    def __init__(self, browser, tab_height):
        # ...
        self.root_frame = None

    def load(self, url, payload=None):
        self.history.append(url)
        # ...
        self.root_frame = Frame(self, None, None)
        self.root_frame.load(url, payload)
```

Note that the guts of `load` now live in the
`Frame`, because the `Frame` owns the HTML tree.
The `Frame` can *also* construct child
`Frame`s, for `<iframe>` elements:

```
class Frame:
    def load(self, url, payload=None):
        # ...
        iframes = [node
                   for node in tree_to_list(self.nodes, [])
                   if isinstance(node, Element)
                   and node.tag == "iframe"
                   and "src" in node.attributes]
        for iframe in iframes:
            document_url = url.resolve(iframe.attributes["src"])
            if not self.allowed_request(document_url):
                print("Blocked iframe", document_url, "due to CSP")
                iframe.frame = None
                continue
            iframe.frame = Frame(self.tab, self, iframe)
        # ...
```

Since iframes can have subresources (and subframes!) and therefore be
slow to load, we should load them asynchronously, just like scripts:

```
class Frame:
    def load(self, url, payload=None):
        for iframe in iframes:
            # ...
            task = Task(iframe.frame.load, document_url)
            self.tab.task_runner.schedule_task(task)
```

And since they are asynchronous, we need to record whether they have
loaded yet, to avoid trying to render an unloaded iframe:

```
class Frame:
    def __init__(self, tab, parent_frame, frame_element):
        # ...
        self.loaded = False

    def load(self, url, payload=None):
        self.loaded = False
        ...
        self.loaded = True
```

So we’ve now got a tree of frames inside a single tab. But because we
will sometimes need direct access to an arbitrary frame, let’s also give
each frame an identifier, which I’m calling a *window ID*:

```
class Tab:
    def __init__(self, browser, tab_height):
        # ...
        self.window_id_to_frame = {}
```

```
class Frame:
    def __init__(self, tab, parent_frame, frame_element):
        # ...
        self.window_id = len(self.tab.window_id_to_frame)
        self.tab.window_id_to_frame[self.window_id] = self
```

Now that we have frames being created, let’s work on rendering those
frames to the screen.

For quite a while, browsers also supported embedded content in the
form of *plugins* like [Java applets](https://en.wikipedia.org/wiki/Java_applet) or [Flash](https://en.wikipedia.org/wiki/Adobe_Flash). But there
were [performance,
security, and accessibility problems](https://developer.mozilla.org/en-US/docs/Learn/HTML/Multimedia_and_embedding/Other_embedding_technologies#the_embed_and_object_elements) because plugins typically
implemented their own rendering, sandboxing, and UI primitives. Over
time, new APIs have closed the gap between web-native content and
“non-web” plugins,For
example, in the last decade the `<canvas>` element has
gained support for hardware-accelerated 3D content, while [WebAssembly](https://en.wikipedia.org/wiki/WebAssembly) can run
at near-native speed. and plugins have therefore become
less common. Personally, I think that’s a good thing: the web is about
making information accessible to everyone, and that requires open
standards, including for embedded content.

Iframe Rendering
================

Rendering is split between the `Tab` and its
`Frame`s: the `Frame` does style and layout, while
the `Tab` does accessibility and paint.Why split the rendering
pipeline this way? Because the accessibility tree and display list are
ultimately transferred from the main thread to the browser thread, so
they get combined anyway. DOM, style, and layout trees, meanwhile, don’t
get passed between threads so don’t intermingle. We’ll
need to implement that split, and also add code to trigger each
`Frame`’s rendering from the `Tab`.

Let’s start with splitting the rendering pipeline. The main methods
here are still the `Tab`’s `run_animation_frame`
and `render`, which iterate over all loaded iframes:

```
class Tab:
    def run_animation_frame(self, scroll):
        # ...
        for (window_id, frame) in self.window_id_to_frame.items():
            if not frame.loaded:
                continue
            frame.js.dispatch_RAF(frame.window_id)
            # ...

    def render(self):
        self.browser.measure.time('render')

        for id, frame in self.window_id_to_frame.items():
            if frame.loaded:
                frame.render()

        if self.needs_accessibility:
            # ...

        if self.needs_paint:
            # ...

        # ...
```

In this code I used a new `dispatch_RAF` method:

```
class JSContext:
    def dispatch_RAF(self):
        self.interp.evaljs("window.__runRAFHandlers()")
```

Note that the `needs_accessibility`,
`pending_hover`, and other flags are all still on the
`Tab`, because they relate to the `Tab`’s part of
rendering. Meanwhile, style and layout happen in the `Frame`
now:

```
class Frame:
    def __init__(self, tab, parent_frame, frame_element):
        # ...
        self.needs_style = False
        self.needs_layout = False

    def set_needs_render(self):
        self.needs_style = True
        self.tab.needs_accessibility = True
        self.tab.set_needs_paint()

    def set_needs_layout(self):
        self.needs_layout = True
        self.tab.needs_accessibility = True
        self.tab.set_needs_paint()

    def render(self):
        if self.needs_style:
            # ...

        if self.needs_layout:
            # ...
```

Again, these dirty flags move to the `Frame` because they
relate to the frame’s part of rendering.

Unlike images, iframes have *no [intrinsic
size](https://developer.mozilla.org/en-US/docs/Glossary/Intrinsic_Size)*: the layout size of an `<iframe>` element
does not depend on its content.There was an attempt to provide iframes with intrinsic
sizing in the past, but it was [removed](https://github.com/whatwg/html/issues/331) from the
HTML specification when no browser implemented it. This may change [in the
future](https://github.com/w3c/csswg-drafts/issues/1771), as there are good use cases for a “seamless” iframe whose
layout is coordinated with its parent frame. That means
there’s a crucial extra bit of communication that needs to happen
between the parent and child frames: how wide and tall should a frame be
laid out? This is defined by the attributes and CSS of the
`iframe` element:

```
class BlockLayout:
    def layout_mode(self):
        # ...
        elif self.node.tag in ["input", "img", "iframe"]:
            return "inline"

    def recurse(self, node):
        else:
            # ...
            elif node.tag == "iframe" and \
                 "src" in node.attributes:
                self.iframe(node)
            # ...

    def iframe(self, node):
        if "width" in self.node.attributes:
            w = dpx(int(self.node.attributes["width"]),
                    self.zoom)
        else:
            w = IFRAME_WIDTH_PX + dpx(2, self.zoom)
        self.add_inline_child(node, w, IframeLayout, self.frame)

    def s
... (truncated)