from direct.showbase.DirectObject import DirectObject
from panda3d.core import NodePath, LineSegs, Point3, Vec3
import math

class KeyboardController(DirectObject):

    def __init__(self, base):
        DirectObject.__init__(self)
        self.base = base
        self.lastUpdate = 0.0

        # Load car model
        self.car = self.base.loader.loadModel("models/car")
        self.car.reparentTo(self.base.render)

        # Parameters
        self.wheelFront = Vec3(1.0, 0.0, 0.0) * 3.37
        self.wheelBack = Vec3(-1.0, 0.0, 0.0) * 3.62

        # Car properties
        self.position = Point3(0.0, 0.0, 0.0)
        self.steerAngle = 10.0
        self.speed = 5.0

        # Controls
        self.upArrowDown = False
        self.leftArrowDown = False
        self.rightArrowDown = False
        self.accept("arrow_up", self.onUpArrow, [True])
        self.accept("arrow_up-up", self.onUpArrow, [False])
        self.accept("arrow_left", self.onLeftArrow, [True])
        self.accept("arrow_left-up", self.onLeftArrow, [False])
        self.accept("arrow_right", self.onRightArrow, [True])
        self.accept("arrow_right-up", self.onRightArrow, [False])
        self.base.taskMgr.add(self.updateCar, "UpdateCarTask")

        # DEBUG
        self.drawAxis(10)
        self.drawWheelBase()

    def onUpArrow(self, down):
        self.upArrowDown = not self.upArrowDown

    def onLeftArrow(self, down):
        self.leftArrowDown = not self.leftArrowDown

    def onRightArrow(self, down):
        self.rightArrowDown = not self.rightArrowDown

    def updateCar(self, task):
        dt = task.time - self.lastUpdate
        self.lastUpdate = task.time

        #self.car.setH(0.0)
        #backWheelUpdate =

        return task.cont

    def drawAxis(self, scale):
        axisSegs = LineSegs("axis")
        axisSegs.setThickness(5)
        axisSegs.setColor(1.0, 0.0, 0.0)
        axisSegs.moveTo(Point3(0.0, 0.0, 0.0))
        axisSegs.drawTo(Point3(1.0, 0.0, 0.0) * scale)

        axisSegs.setColor(0.0, 1.0, 0.0)
        axisSegs.moveTo(Point3(0.0, 0.0, 0.0))
        axisSegs.drawTo(Point3(0.0, 1.0, 0.0) * scale)

        axisSegs.setColor(0.0, 0.0, 1.0)
        axisSegs.moveTo(Point3(0.0, 0.0, 0.0))
        axisSegs.drawTo(Point3(0.0, 0.0, 1.0) * scale)

        axisNode = axisSegs.create()
        axisNodePath = self.car.attachNewNode(axisNode)
        return axisNodePath

    def drawWheelBase(self):
        wheelSegs = LineSegs("wheelBase")
        wheelSegs.setThickness(5)
        wheelSegs.moveTo(self.wheelFront + Vec3(0.0, 5.0, 1.44))
        wheelSegs.drawTo(self.wheelFront + Vec3(0.0, -5.0, 1.44))

        wheelSegs.moveTo(self.wheelBack + Vec3(0.0, 5.0, 1.44))
        wheelSegs.drawTo(self.wheelBack + Vec3(0.0, -5.0, 1.44))

        wheelNode = wheelSegs.create()
        wheelNodePath = self.car.attachNewNode(wheelNode)
        return wheelNodePath
