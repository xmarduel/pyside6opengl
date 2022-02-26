
import ctypes
import math
import numpy as np
import sys

from PySide6.QtCore import (QSize, QPoint)
from PySide6.QtGui import (QOpenGLFunctions, QVector2D, QVector3D, QVector4D, QMatrix4x4, QImage)
from PySide6.QtWidgets import (QApplication, QMainWindow)

from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6 import QtGui

from PySide6.QtCore import Signal

from PySide6.QtOpenGL import (QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLTexture)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL import GL

M_PI = math.acos(-1)
ZOOMSTEP = 1.1


class Window(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.gl_widget = GLWidget()

        self.setCentralWidget(self.gl_widget)
        self.setWindowTitle(self.tr("Hello GL"))

class Vertex:
    nb_float = 9
    bytes_size = nb_float * 4 #  4 bytes each
    # the size/offset do not strictly belong to the Vertex class, but are properties
    # of the generated numpy array. However it is pratical to have them here.
    size = {'position' : 3 , 'color': 4, 'texcoord': 2} # size in float of position/texcoord
    offset = {'position' : 0, 'color': 12, 'texcoord': 28 } # offsets in np array in bytes

    def __init__(self, position: QVector3D, color: QVector4D, texcoord: QVector2D):
        self.position = position
        self.color = color
        self.texcoord = texcoord

class Scene():
    '''
    A cube with 24 vertices "not shared"
    '''
    def __init__(self):
        self.nb_float = 0
        self.nb_int = 0

        vtype = [('position', np.float32, 3),
                 ('color', np.float32, 4),
                 ('texcoord', np.float32, 2)]
        itype = np.uint32
        
        # 8 points constituting a cube
        p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                  [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]],
                  dtype=float)

        # the colors on the points
        c = np.array([[0,1,1,1], [0,0,1,1], [0,0,0,1], [0,1,0,1],
                  [1,1,0,1], [1,1,1,1], [1,0,1,1], [1,0,0,1]],
                  dtype=float)

        # Texture coords
        t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

        # the 24 nodes for the faces, 4 for each face - their coords is given with the p array (of length 8)
        faces_p = [0, 1, 2, 3,   0, 3, 4, 5,   0, 5, 6, 1,   1, 6, 7, 2,   7, 4, 3, 2,    4, 7, 6, 5]
        # the 24 nodes for the texture, 4 for each face - their coords is given with the t array (of length 4)- the 4 corner of the image
        faces_t = [0, 1, 2, 3,   0, 1, 2, 3,   0, 1, 2, 3,   0, 1, 2, 3,   0, 1, 2, 3,    0, 1, 2, 3]

        # list of vertices : 6 faces with for each 4 vertices : use faces_p indexes 
        self.vertices = np.zeros(24, vtype)
        self.vertices['position'] = p[faces_p]
        self.vertices['color'] = c[faces_p]
        self.vertices['texcoord'] = t[faces_t]

        print("--- vertex buffer -----------------")
        print(self.vertices)

        # index buffer - of size 36 = 12 triangles * 3 nodes
        self.filled = np.resize(np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
        self.filled += np.repeat(4 * np.arange(6, dtype=itype), 6)
        
        print("--- index buffer -----------------")
        print(self.filled)

        self.nb_float = 24 * Vertex.nb_float
        self.nb_int = 36

        self.textureData = QImage("./pics/crate.png")

    def const_vertex_data(self):
        return self.vertices.tobytes()

    def const_index_data(self):
        return self.filled.tobytes()

    def getMinimumExtremes(self) -> QVector3D : 
        return QVector3D(-1, -1, -1)

    def getMaximumExtremes(self) -> QVector3D :
        return QVector3D(1, 1, 1)
    


class GLWidget(QOpenGLWidget, QOpenGLFunctions):
    '''
    '''
    vertex_code_only_color = """
    uniform mat4   model;
    uniform mat4   view;
    uniform mat4   projection;
    attribute vec3 position;
    attribute vec4 color;
    varying vec4   v_color;

    void main()
    {
        gl_Position = projection * view * model * vec4(position, 1.0);

        v_color = color;
    }  """

    fragment_code_only_color = """
    varying vec4 v_color;
    void main() { gl_FragColor = v_color; } """

    vertex_code_tex = """
    uniform mat4   model;
    uniform mat4   view;
    uniform mat4   projection;
    attribute vec3 position;
    attribute vec2 texcoord;   // Vertex texture coordinates
    varying vec2   v_texcoord;   // Interpolated fragment texture coordinates (out)

    void main()
    {
        // Assign varying variables
        v_texcoord  = texcoord;

        // Final position
        gl_Position = projection * view * model * vec4(position,1.0);
    } """

    fragment_code_tex = """
    uniform sampler2D texture; // Texture
    varying vec2 v_texcoord;   // Interpolated fragment texture coordinates (in)
    void main()
    {
        // Get texture color
        gl_FragColor = texture2D(texture, v_texcoord);
    } """

    float_size = ctypes.sizeof(ctypes.c_float) # 4 bytes
    int_size = ctypes.sizeof(ctypes.c_uint32) # 4 bytes

    rotationChanged = Signal()
    resized = Signal()

    def __init__(self, parent=None):
        QOpenGLWidget.__init__(self, parent)
        QOpenGLFunctions.__init__(self)

        self.m_xRot = 90.0 
        self.m_yRot = 0.0 
        self.m_xLastRot = 0.0 
        self.m_yLastRot = 0.0 
        self.m_xPan = 0.0 
        self.m_yPan = 0.0 
        self.m_xLastPan = 0.0 
        self.m_yLastPan = 0.0 
        self.m_xLookAt = 0.0
        self.m_yLookAt = 0.0
        self.m_zLookAt = 0.0
        self.m_lastPos = QPoint(0, 0)
        self.m_zoom = 1
        self.m_distance = 10.0 
        self.m_xMin = 0.0 
        self.m_xMax = 0.0 
        self.m_yMin = 0.0
        self.m_yMax = 0.0 
        self.m_zMin = 0.0
        self.m_zMax = 0.0 
        self.m_xSize = 0.0 
        self.m_ySize = 0.0 
        self.m_zSize = 0.0 

        self.m_xRotTarget = 90.0 
        self.m_yRotTarget = 0.0 
        self.m_xRotStored = 0.0  
        self.m_yRotStored = 0.0 

        self.proj = QMatrix4x4()
        self.proj.setToIdentity()

        self.view = QMatrix4x4()
        self.view.setToIdentity()
        self.view.translate(QVector3D(0,0,-5))

        self.model = QMatrix4x4()
        self.model.setToIdentity()

        self.updateProjection()
        self.updateView()

        self.cmdFit = QtWidgets.QToolButton(self)
        self.cmdIsometric = QtWidgets.QToolButton(self)
        self.cmdTop = QtWidgets.QToolButton(self)
        self.cmdFront = QtWidgets.QToolButton(self)
        self.cmdLeft = QtWidgets.QToolButton(self)

        self.cmdFit.setMinimumSize(QtCore.QSize(24,24))
        self.cmdIsometric.setMinimumSize(QtCore.QSize(24,24))
        self.cmdTop.setMinimumSize(QtCore.QSize(24,24))
        self.cmdFront.setMinimumSize(QtCore.QSize(24,24))
        self.cmdLeft.setMinimumSize(QtCore.QSize(24,24))

        self.cmdFit.setMaximumSize(QtCore.QSize(24,24))
        self.cmdIsometric.setMaximumSize(QtCore.QSize(24,24))
        self.cmdTop.setMaximumSize(QtCore.QSize(24,24))
        self.cmdFront.setMaximumSize(QtCore.QSize(24,24))
        self.cmdLeft.setMaximumSize(QtCore.QSize(24,24))

        self.cmdFit.setToolTip("Fit")
        self.cmdIsometric.setToolTip("Isometric view")
        self.cmdTop.setToolTip("Top view")
        self.cmdFront.setToolTip("Front view")
        self.cmdLeft.setToolTip("Left view")

        self.cmdFit.setIcon(QtGui.QIcon("./pics/fit_1.png"))
        self.cmdIsometric.setIcon(QtGui.QIcon("./pics/cube.png"))
        self.cmdTop.setIcon(QtGui.QIcon("./pics/cubeTop.png"))
        self.cmdFront.setIcon(QtGui.QIcon("./pics/cubeFront.png"))
        self.cmdLeft.setIcon(QtGui.QIcon("./pics/cubeLeft.png"))

        self.cmdFit.clicked.connect(self.on_cmdFit_clicked)
        self.cmdIsometric.clicked.connect(self.on_cmdIsometric_clicked)
        self.cmdTop.clicked.connect(self.on_cmdTop_clicked)
        self.cmdFront.clicked.connect(self.on_cmdFront_clicked)
        self.cmdLeft.clicked.connect(self.on_cmdLeft_clicked)

        self.rotationChanged.connect(self.onVisualizatorRotationChanged)
        self.resized.connect(self.placeVisualizerButtons)

        self.scene = Scene()
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer()
        self.ibo = QOpenGLBuffer()
        self.texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        self.program = QOpenGLShaderProgram()


    def placeVisualizerButtons(self):
        w = self.width()
        cmdIsometric_w =  self.cmdIsometric.width() 

        self.cmdIsometric.move(self.width() - self.cmdIsometric.width() - 8, 8)
        self.cmdTop.move(self.cmdIsometric.geometry().left() - self.cmdTop.width() - 8, 8)
        self.cmdLeft.move(self.width() - self.cmdLeft.width() - 8, self.cmdIsometric.geometry().bottom() + 8)
        self.cmdFront.move(self.cmdLeft.geometry().left() - self.cmdFront.width() - 8, self.cmdIsometric.geometry().bottom() + 8)
        self.cmdFit.move(self.width() - self.cmdFit.width() - 8, self.cmdLeft.geometry().bottom() + 8)

    def on_cmdTop_clicked(self):
        self.setTopView()
        self.updateView()

        self.onVisualizatorRotationChanged()

    def on_cmdFront_clicked(self):
        self.setFrontView()
        self.updateView()

        self.onVisualizatorRotationChanged()

    def on_cmdLeft_clicked(self):
        self.setLeftView()
        self.updateView()

        self.onVisualizatorRotationChanged()

    def on_cmdIsometric_clicked(self):
        self.setIsometricView()
        self.updateView()

        self.onVisualizatorRotationChanged()

    def on_cmdFit_clicked(self):
        self.fitDrawable()

    def calculateVolume(self, size: QtGui.QVector3D) -> float:
        return size.x() * size.y() * size.z()

    def fitDrawable(self):
        self.updateExtremes()

        a = self.m_ySize / 2 / 0.25 * 1.3 + \
            (self.m_zMax - self.m_zMin) / 2
        b = self.m_xSize / 2 / 0.25 * 1.3 / (self.width() / self.height()) + \
            (self.m_zMax - self.m_zMin) / 2
        
        self.m_distance = max(a, b)

        if self.m_distance == 0:
            self.m_distance = 10

        self.m_xLookAt = (self.m_xMax - self.m_xMin) / 2 + self.m_xMin
        self.m_zLookAt = -((self.m_yMax - self.m_yMin) / 2 + self.m_yMin)
        self.m_yLookAt = (self.m_zMax - self.m_zMin) / 2 + self.m_zMin
       

        self.m_xPan = 0
        self.m_yPan = 0
        self.m_zoom = 1

        self.updateProjection()
        self.updateView()

    def normalizeAngle(self, angle: float) -> float:
        while angle < 0: 
            angle += 360
        while angle > 360: 
            angle -= 360

        return angle

    def updateExtremes(self):
        self.m_xMin = self.scene.getMinimumExtremes().x() 
        self.m_xMax = self.scene.getMaximumExtremes().x() 
        
        self.m_yMin = self.scene.getMinimumExtremes().y() 
        self.m_yMax = self.scene.getMaximumExtremes().y() 
        
        self.m_zMin = self.scene.getMinimumExtremes().z()
        self.m_zMax = self.scene.getMaximumExtremes().z() 

        self.m_xSize = self.m_xMax - self.m_xMin
        self.m_ySize = self.m_yMax - self.m_yMin
        self.m_zSize = self.m_zMax - self.m_zMin
    
    def setIsometricView(self):
        ''' no animation yet '''
        self.m_xRotTarget = 45
        self.m_yRotTarget = 405 if self.m_yRot > 180 else 45

        self.m_xRot = 45
        self.m_yRot = 405 if self.m_yRot > 180 else 45

    def setTopView(self):
        ''' no animation yet '''
        self.m_xRotTarget = 90
        self.m_yRotTarget = 360 if self.m_yRot > 180 else 0

        self.m_xRot = 90
        self.m_yRot = 360 if self.m_yRot > 180 else 0

    def setFrontView(self):
        ''' no animation yet '''
        self.m_xRotTarget = 0
        self.m_yRotTarget = 360 if self.m_yRot > 180 else 0

        self.m_xRot = 0
        self.m_yRot = 360 if self.m_yRot > 180 else 0

    def setLeftView(self):
        ''' no animation yet '''
        self.m_xRotTarget = 0
        self.m_yRotTarget = 450 if self.m_yRot > 270 else 90

        self.m_xRot= 0
        self.m_yRot = 450 if self.m_yRot > 270 else 90

    def updateProjection(self):
        # Reset projection
        self.proj.setToIdentity()

        asp = self.width() / self.height()
        self.proj.frustum( \
                (-0.5 + self.m_xPan) * asp, \
                (0.5 + self.m_xPan) * asp,
                -0.5 + self.m_yPan, \
                0.5 + self.m_yPan, 2, self.m_distance * 2)

    def updateView(self):
        # Set view matrix
        self.view.setToIdentity()

        r = self.m_distance
        angY = M_PI / 180 * self.m_yRot
        angX = M_PI / 180 * self.m_xRot

        eye = QtGui.QVector3D( \
            r * math.cos(angX) * math.sin(angY) + self.m_xLookAt, \
            r * math.sin(angX) + self.m_yLookAt, \
            r * math.cos(angX) * math.cos(angY) + self.m_zLookAt)
        
        center = QtGui.QVector3D(self.m_xLookAt, self.m_yLookAt, self.m_zLookAt)

        xRot = M_PI if self.m_xRot < 0 else 0

        up = QtGui.QVector3D( \
            -math.sin(angY + xRot) if math.fabs(self.m_xRot) == 90 else 0, 
            math.cos(angX), 
            -math.cos(angY + xRot) if math.fabs(self.m_xRot) == 90 else 0)

        self.view.lookAt(eye, center, up.normalized())

        self.view.translate(self.m_xLookAt, self.m_yLookAt, self.m_zLookAt)
        self.view.scale(self.m_zoom, self.m_zoom, self.m_zoom)
        self.view.translate(-self.m_xLookAt, -self.m_yLookAt, -self.m_zLookAt)

        self.view.rotate(-90, 1.0, 0.0, 0.0)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.m_lastPos = event.position()
        self.m_xLastRot = self.m_xRot
        self.m_yLastRot = self.m_yRot
        self.m_xLastPan = self.m_xPan
        self.m_yLastPan = self.m_yPan

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if (event.buttons() & QtGui.Qt.MiddleButton and (not(event.modifiers() & QtGui.Qt.ShiftModifier))) or event.buttons() & QtCore.Qt.LeftButton:

            self.m_yRot = self.normalizeAngle(self.m_yLastRot - (event.position().x() - self.m_lastPos.x()) * 0.5)
            self.m_xRot = self.m_xLastRot + (event.position().y() - self.m_lastPos.y()) * 0.5

            if self.m_xRot < -90:
                self.m_xRot = -90
            if self.m_xRot > 90:
                self.m_xRot = 90

            self.updateView()
            self.rotationChanged.emit()
    

        if (event.buttons() & QtCore.Qt.MiddleButton and event.modifiers() & QtGui.Qt.ShiftModifier) or event.buttons() & QtCore.Qt.RightButton:
            self.m_xPan = self.m_xLastPan - (event.position().x() - self.m_lastPos.x()) * 1 / (float)(self.width())
            self.m_yPan = self.m_yLastPan + (event.position().y() - self.m_lastPos.y()) * 1 / (float)(self.height())

            self.updateProjection()

        self.update()

    def wheelEvent(self, we: QtGui.QWheelEvent):
        if self.m_zoom > 0.1 and we.angleDelta().y() < 0:
            self.m_xPan -= ((float)(we.position().x() / self.width() - 0.5 + self.m_xPan)) * (1 - 1 / ZOOMSTEP)
            self.m_yPan += ((float)(we.position().y() / self.height() - 0.5 - self.m_yPan)) * (1 - 1 / ZOOMSTEP)

            self.m_zoom /= ZOOMSTEP
        elif self.m_zoom < 10 and we.angleDelta().y() > 0:
            self.m_xPan -= ((float)(we.position().x() / self.width() - 0.5 + self.m_xPan)) * (1 - ZOOMSTEP)
            self.m_yPan += ((float)(we.position().y() / self.height() - 0.5 - self.m_yPan)) * (1 - ZOOMSTEP)

            self.m_zoom *= ZOOMSTEP

        self.updateProjection()
        self.updateView()

        self.update()

    def onVisualizatorRotationChanged(self):
        self.update()

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
     
    def sizeHint(self):
        return QSize(400, 400)

    def cleanup(self):
        self.makeCurrent()
        self.vbo.destroy()
        #self.ibo.destroy()
        del self.program
        self.program = None
        self.doneCurrent()

    def initializeGL(self):
        self.context().aboutToBeDestroyed.connect(self.cleanup)
        self.initializeOpenGLFunctions()
        self.glClearColor(0, 0, 0, 0)

        self.program = QOpenGLShaderProgram()

        #self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertex_code_only_color)
        #self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragment_code_only_color)
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertex_code_tex)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragment_code_tex)
        self.program.link()

        self.program.bind()

        self.vao.create()
        vao_binder = QOpenGLVertexArrayObject.Binder(self.vao)

        self.vbo.create() # QOpenGLBuffer.VertexBuffer
        self.vbo.bind()
        self.vbo.allocate(self.scene.const_vertex_data(), self.scene.nb_float * self.float_size)

        self.ibo.create() # QOpenGLBuffer.IndexBuffer
        self.ibo.bind()
        self.ibo.allocate(self.scene.const_index_data(), self.scene.nb_int * self.int_size)

        # -------------------------------------------------------------------------------------
        # # ------------------------- the texture ---------------------------------------------
        self.texture.create()
        # Wrap style
        self.texture.setWrapMode(QOpenGLTexture.ClampToBorder)

        # Texture Filtering
        self.texture.setMinificationFilter(QOpenGLTexture.NearestMipMapLinear)
        self.texture.setMagnificationFilter(QOpenGLTexture.Linear)

        # Kopiere Daten in Texture und Erstelle Mipmap
        self.texture.setData(self.scene.textureData)
        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------

        self.setup_vertex_attribs()

        self.program.release()
        vao_binder = None

    def setup_vertex_attribs(self):
        self.vbo.bind()

        modelLocation = self.program.uniformLocation("model")
        self.program.setUniformValue(modelLocation, self.model)

        projLocation = self.program.uniformLocation("projection")
        self.program.setUniformValue(projLocation, self.proj)

        viewLocation = self.program.uniformLocation("view")
        self.program.setUniformValue(viewLocation, self.view)

        # Offset for position
        offset = Vertex.offset['position']
        size = Vertex.size['position'] # nb float in a position "packet" 
        stride = Vertex.bytes_size # nb bytes in a vertex 

        vertexLocation = self.program.attributeLocation("position")
        self.program.enableAttributeArray(vertexLocation)
        self.program.setAttributeBuffer(vertexLocation, GL.GL_FLOAT, offset, size, stride)

        # Offset for color
        offset = Vertex.offset['color'] # size in bytes of preceding data (position = QVector3D)
        size = Vertex.size['color'] # nb float in a color "packet" 
        stride = Vertex.bytes_size # nb bytes in a vertex

        colorLocation =  self.program.attributeLocation("color")
        self.program.enableAttributeArray(colorLocation)
        self.program.setAttributeBuffer(colorLocation, GL.GL_FLOAT, offset, size, stride)

        # Offset for texcoord
        offset = Vertex.offset['texcoord'] # size in bytes of preceding data (position = QVector3D + color = QVector4D)
        size = Vertex.size['texcoord'] # nb float in a texcoord "packet" 
        stride = Vertex.bytes_size # nb bytes in a vertex 

        texcoordLocation =  self.program.attributeLocation("texcoord")
        self.program.enableAttributeArray(texcoordLocation)
        self.program.setAttributeBuffer(texcoordLocation, GL.GL_FLOAT, offset, size, stride)

        # --------------------------------- the texture ------------------------------------------
        # uniform for fragment texture
        textureLocationID = self.program.uniformLocation("texture")
        self.program.setUniformValue(textureLocationID, 0) # the index of the texture
        # --------------------------------- the texture ------------------------------------------

        self.vbo.release()

    def paintGL(self):
        self.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.glEnable(GL.GL_DEPTH_TEST)
       
        vao_binder = QOpenGLVertexArrayObject.Binder(self.vao)
        self.program.bind()
        
        modelLocation = self.program.uniformLocation("model")
        self.program.setUniformValue(modelLocation, self.model)

        projLocation = self.program.uniformLocation("projection")
        self.program.setUniformValue(projLocation, self.proj)

        viewLocation = self.program.uniformLocation("view")
        self.program.setUniformValue(viewLocation, self.view)

        # --------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------
        self.texture.bind(0) # bind texture to texture index 0 -> accessible in fragment shader through "texture"
        # --------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------

        # Filled cube
        self.glDrawElements(GL.GL_TRIANGLES, 12*3, GL.GL_UNSIGNED_INT, self.scene.const_index_data())

        self.program.release()
        vao_binder = None

        self.updateProjection()
        self.updateView()
        
        self.update()

    def resizeGL(self, width, height):
        ratio = width / float(height)
        self.proj.perspective(45.0, ratio, 2.0, 100.0)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
     

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = Window()
    main_window.show()

    res = app.exec()
    sys.exit(res)