"""
Point layers for the 3D view with their own softness ("blur") and opacity ("fade").

The scene is drawn as two layers:

- lit: the particles that carry the colouring property (or every particle when
  colouring by type), drawn after the other layer, so they are always on top;
- unlit: the rest, drawn first and dim, or not at all.

Both are GLScatterPlotItems whose fragment shader multiplies each point's alpha
by ``fade`` and blends a hard disc into a Gaussian profile by ``blur``. These
are shader uniforms, so moving a slider redraws without re-uploading points.

The lit layer can blend two ways, because "always on top" alone fails when
many lit particles overlap:

- Glow: additive (src*alpha + dst), order independent; dense regions brighten
  and saturate, so fade sets how much overlap it takes to saturate.
- Solid: depth-tested among lit particles only (the unlit layer writes no
  depth), so the nearest lit particle wins and dense regions stay readable.

Weighted blended order-independent transparency (McGuire & Bavoil 2013) would
combine both, but needs extra render targets that pyqtgraph does not expose.
docs/FEATURES.md explains the trade-off for users.

If the installed pyqtgraph has a different point shader, the layers fall back
to plain GLScatterPlotItems and fade is applied to the colours instead.
"""

from typing import Optional

import numpy as np

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl.items import GLScatterPlotItem as _gsp
    from OpenGL import GL
    from OpenGL.GL import shaders
    from pyqtgraph.Qt import QtGui
    HAS_GL = True
except Exception:  # no OpenGL: the window shows a placeholder instead of the 3D view
    gl = None
    HAS_GL = False

GLOW, SOLID = 'Glow', 'Solid'
BLEND_MODES = (GLOW, SOLID)

_FRAG_BODY = """
    vec2 xy = (gl_PointCoord - 0.5) * 2.0;
    float r2 = dot(xy, xy);
    if (r2 > 1.0) discard;
    float a = v_color.a * mix(1.0, exp(-3.0 * r2), u_blur) * u_fade;
    if (u_solid > 0.5 && a < 0.04) discard;
"""
_FRAG_LEGACY = """
    #ifdef GL_ES
    precision mediump float;
    #endif
    uniform float u_blur;
    uniform float u_fade;
    uniform float u_solid;
    varying vec4 v_color;
    void main() {""" + _FRAG_BODY + """
        gl_FragColor = vec4(v_color.rgb, a);
    }
"""
_FRAG_CORE = """
    #ifdef GL_ES
    precision mediump float;
    #endif
    uniform float u_blur;
    uniform float u_fade;
    uniform float u_solid;
    in vec4 v_color;
    out vec4 fragColor;
    void main() {""" + _FRAG_BODY + """
        fragColor = vec4(v_color.rgb, a);
    }
"""

SOFT_SHADER = HAS_GL and all(hasattr(_gsp, n) for n in ('SHADER_LEGACY', 'SHADER_CORE')) and \
    hasattr(_gsp.GLScatterPlotItem, 'getShaderProgram')


def gl_options(mode: str) -> dict:
    """OpenGL state for a layer: unlit and Glow are additive without depth; Solid is depth-tested."""
    if mode == SOLID:
        return {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: True, GL.GL_CULL_FACE: False,
                'glBlendFunc': (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)}
    # With the depth test off nothing is written to the depth buffer, so these never hide anything.
    return {GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True, GL.GL_CULL_FACE: False,
            'glBlendFunc': (GL.GL_SRC_ALPHA, GL.GL_ONE)}


if HAS_GL:
    class SoftPoints(_gsp.GLScatterPlotItem):
        """GLScatterPlotItem with blur and fade uniforms (see the module docstring)."""

        _program = None

        def __init__(self, **kw):
            self.blur, self.fade, self.mode = 0.0, 1.0, GLOW
            self._raw_color = None
            super().__init__(**kw)
            self.setGLOptions(gl_options(GLOW))

        def set_style(self, size: Optional[float] = None, blur: Optional[float] = None,
                      fade: Optional[float] = None, mode: Optional[str] = None):
            if size is not None and float(size) != self.size:
                super().setData(size=float(size))
            if blur is not None:
                self.blur = float(np.clip(blur, 0.0, 1.0))
            if fade is not None:
                self.fade = float(np.clip(fade, 0.0, 1.0))
                if not SOFT_SHADER and self._raw_color is not None:
                    self._apply_fallback_fade()
            if mode is not None and mode != self.mode:
                self.mode = mode
                self.setGLOptions(gl_options(mode))
            self.update()

        def setData(self, **kw):
            if not SOFT_SHADER and 'color' in kw and isinstance(kw['color'], np.ndarray):
                self._raw_color = kw['color']
                kw['color'] = self._faded(kw['color'])
            super().setData(**kw)

        def _faded(self, col: np.ndarray) -> np.ndarray:
            col = np.array(col, dtype=np.float32, copy=True)
            if col.ndim == 2 and col.shape[1] == 4:
                col[:, 3] *= self.fade
            return col

        def _apply_fallback_fade(self):
            super().setData(color=self._faded(self._raw_color))

        if SOFT_SHADER:
            @staticmethod
            def getShaderProgram():
                klass = SoftPoints
                if klass._program is not None:
                    return klass._program
                ctx = QtGui.QOpenGLContext.currentContext()
                fmt = ctx.format()
                if ctx.isOpenGLES():
                    core = fmt.version() >= (3, 0)
                    glsl = "#version 300 es\n" if core else "#version 100\n"
                else:
                    core = fmt.version() >= (3, 1)
                    glsl = "#version 140\n" if core else "#version 120\n"
                base = _gsp.SHADER_CORE if core else _gsp.SHADER_LEGACY
                sources = {GL.GL_VERTEX_SHADER: base[GL.GL_VERTEX_SHADER],
                           GL.GL_FRAGMENT_SHADER: _FRAG_CORE if core else _FRAG_LEGACY}
                compiled = [shaders.compileShader([glsl, v], k) for k, v in sources.items()]
                program = shaders.compileProgram(*compiled)
                GL.glBindAttribLocation(program, 0, "a_position")
                GL.glBindAttribLocation(program, 1, "a_color")
                GL.glBindAttribLocation(program, 2, "a_size")
                GL.glLinkProgram(program)
                klass._program = program
                return program

            def paint(self):
                if self.pos is None:
                    return
                # Uniforms are program state: set them, then let the base class draw.
                program = self.getShaderProgram()
                GL.glUseProgram(program)
                GL.glUniform1f(GL.glGetUniformLocation(program, "u_blur"), self.blur)
                GL.glUniform1f(GL.glGetUniformLocation(program, "u_fade"), self.fade)
                GL.glUniform1f(GL.glGetUniformLocation(program, "u_solid"), 1.0 if self.mode == SOLID else 0.0)
                GL.glUseProgram(0)
                super().paint()
