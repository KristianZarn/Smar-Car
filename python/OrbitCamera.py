from direct.showbase.DirectObject import DirectObject
from panda3d.core import NodePath, Point3, Vec3
import math


class CameraController(DirectObject):

    def __init__(self, base, r, theta, phi):
        DirectObject.__init__(self)
        self.base = base

        # Parameters
        self.rotateMag = 0.5
        self.moveMag = 50
        self.zoomMag = 100

        # Camera properties
        self.r = r
        self.theta = theta
        self.phi = phi
        self.target = NodePath("target")
        self.target.reparentTo(self.base.render)
        self.base.camera.reparentTo(self.target)
        self.followingObject = None

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

        # Run task that updates camera
        self.base.taskMgr.add(self.updateCamera, "UpdateCameraTask", priority=1)

    def setTarget(self, parent):
        self.target.reparentTo(parent)

    def follow(self, obj):
        self.followingObject = obj

    def stopFollowing(self):
        self.followingObject = None

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
                self.rotateTheta(dX * math.pi * self.rotateMag)
                self.rotatePhi(-dY * math.pi * self.rotateMag)

            if self.mouseDown2 and self.followingObject is None:
                vecX = self.target.getRelativeVector(self.base.camera, Vec3.right())
                vecY = self.target.getRelativeVector(self.base.camera, Vec3.forward())
                vecY.setZ(0.0)
                vecY.normalize()
                offset = (vecX * dX * self.moveMag) + (vecY * dY * self.moveMag)
                self.target.setPos(self.target, offset)

            if self.followingObject is not None:
                self.target.setPos(self.followingObject.getPos())

            if self.mouseDown3:
                self.zoom(dY * self.zoomMag)

            self.mousePrevX = mouseX
            self.mousePrevY = mouseY

            # Update camera position
            position = Point3(0.0, 0.0, 0.0)
            position.setX(self.r * math.cos(self.theta) * math.sin(self.phi))
            position.setY(self.r * math.sin(self.theta) * math.sin(self.phi))
            position.setZ(self.r * math.cos(self.phi))
            self.base.camera.setPos(position)
            self.base.camera.lookAt(self.target)

        return task.cont
