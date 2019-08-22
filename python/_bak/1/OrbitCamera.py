from direct.showbase.DirectObject import DirectObject
from panda3d.core import Point3, Vec3
import math

class CameraController(DirectObject):

    def __init__(self, base, r, theta, phi):
        DirectObject.__init__(self)

        # Camera properties
        self.base = base
        self.r = r
        self.theta = theta
        self.phi = phi
        self.position = Point3(0.0, 0.0, 0.0)
        self.offset = Vec3(0.0, 0.0, 0.0)
        self.target = Point3(0.0, 0.0, 0.0)

        # Controls
        self.mouseDown1 = False
        self.mouseDown2 = False
        self.mouseDown3 = False
        self.mousePrevX = 0.0
        self.mousePrevY = 0.0
        self.accept("mouse1", self.onMouse1, [True])
        self.accept("mouse1-up", self.onMouse1, [False])
        self.accept("mouse2", self.onMouse2, [True])
        self.accept("mouse2-up", self.onMouse2, [False])
        self.accept("mouse3", self.onMouse3, [True])
        self.accept("mouse3-up", self.onMouse3, [False])
        base.taskMgr.add(self.updateCamera, "UpdateCameraTask")

    def setTarget(self, target):
        self.target = target

    def zoom(self, dR):
        self.r += dR
        if self.r < 0.0:
            self.r = 0.0

    def rotateTheta(self, dTheta):
        self.theta += dTheta
        if self.theta < 0.0:
            self.theta += 2 * math.pi
        if self.theta > 2 * math.pi:
            self.theta -= 2 * math.pi

    def rotatePhi(self, dPhi):
        self.phi += dPhi
        if self.phi < 0.1:
            self.phi = 0.1
        if self.phi > math.pi-0.1:
            self.phi = math.pi-0.1

    def onMouse1(self, down):
        if not self.mouseDown2 and not self.mouseDown3:
            if down and self.base.mouseWatcherNode.hasMouse():
                self.mouseDown1 = True
                self.mousePrevX = self.base.mouseWatcherNode.getMouseX()
                self.mousePrevY = self.base.mouseWatcherNode.getMouseY()
            else:
                self.mouseDown1 = False

    def onMouse2(self, down):
        if not self.mouseDown1 and not self.mouseDown3:
            if down and self.base.mouseWatcherNode.hasMouse():
                self.mouseDown2 = True
                self.mousePrevX = self.base.mouseWatcherNode.getMouseX()
                self.mousePrevY = self.base.mouseWatcherNode.getMouseY()
            else:
                self.mouseDown2 = False

    def onMouse3(self, down):
        if not self.mouseDown1 and not self.mouseDown2:
            if down and self.base.mouseWatcherNode.hasMouse():
                self.mouseDown3 = True
                self.mousePrevX = self.base.mouseWatcherNode.getMouseX()
                self.mousePrevY = self.base.mouseWatcherNode.getMouseY()
            else:
                self.mouseDown3 = False

    def updateCamera(self, task):
        if self.base.mouseWatcherNode.hasMouse():
            # Register camera controls
            mouseX = self.base.mouseWatcherNode.getMouseX()
            mouseY = self.base.mouseWatcherNode.getMouseY()
            dX = self.mousePrevX - mouseX
            dY = self.mousePrevY - mouseY

            if self.mouseDown1:
                self.rotateTheta(dX * (math.pi / 2.0))
                self.rotatePhi(-dY * (math.pi / 2.0))

            if self.mouseDown2:
                self.offset.setX(self.offset.getX() + dX * 10)
                self.offset.setY(self.offset.getY() + dY * 10)

            if self.mouseDown3:
                self.zoom(dY * 50)

            self.mousePrevX = mouseX
            self.mousePrevY = mouseY

            # Update camera position
            self.position.setX(self.r * math.cos(self.theta) * math.sin(self.phi))
            self.position.setY(self.r * math.sin(self.theta) * math.sin(self.phi))
            self.position.setZ(self.r * math.cos(self.phi))

            vecX = self.base.render.getRelativeVector(self.base.camera, Vec3.right())
            vecY = self.base.render.getRelativeVector(self.base.camera, Vec3.forward())
            vecY.setZ(0.0)
            vecY.normalize()
            offsetWorld = (vecX * self.offset.getX()) + (vecY * self.offset.getY())

            self.base.camera.setPos(self.position + offsetWorld)
            self.base.camera.lookAt(self.target + offsetWorld)

        return task.cont
