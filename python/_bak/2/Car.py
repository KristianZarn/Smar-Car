from direct.showbase.DirectObject import DirectObject
from panda3d.core import CollisionHandlerQueue, CollisionNode, CollisionRay, LineSegs, Point3, Vec3
import math

class KeyboardController(DirectObject):

    def __init__(self, base):
        DirectObject.__init__(self)
        self.base = base
        self.lastUpdate = 0.0

        # Parameters
        self.wheelBaseOffset = 3.4
        self.wheelSideOffset = 3.1
        self.speedMax = 50.0
        self.steerAngleMax = 20.0
        self.numSensors = 5
        self.sensorHeight = 2.0

        # Collisions
        self.collisionHandler = CollisionHandlerQueue()

        # Load car model
        self.car = self.base.loader.loadModel("models/carBody")
        self.car.reparentTo(self.base.render)

        self.wheelFrontLeft = self.base.loader.loadModel("models/carWheel")
        self.wheelFrontLeft.reparentTo(self.car)
        self.wheelFrontLeft.setX(self.wheelBaseOffset)
        self.wheelFrontLeft.setY(self.wheelSideOffset)

        self.wheelFrontRight = self.base.loader.loadModel("models/carWheel")
        self.wheelFrontRight.reparentTo(self.car)
        self.wheelFrontRight.setX(self.wheelBaseOffset)
        self.wheelFrontRight.setY(-self.wheelSideOffset)

        self.wheelBackLeft = self.base.loader.loadModel("models/carWheel")
        self.wheelBackLeft.reparentTo(self.car)
        self.wheelBackLeft.setX(-self.wheelBaseOffset)
        self.wheelBackLeft.setY(self.wheelSideOffset)

        self.wheelBackRight = self.base.loader.loadModel("models/carWheel")
        self.wheelBackRight.reparentTo(self.car)
        self.wheelBackRight.setX(-self.wheelBaseOffset)
        self.wheelBackRight.setY(-self.wheelSideOffset)

        # Car properties
        self.steerAngle = 0.0
        self.speed = 0.0
        self.wheelFront = Vec3(1.0, 0.0, 0.0) * self.wheelBaseOffset
        self.wheelBack = Vec3(-1.0, 0.0, 0.0) * self.wheelBaseOffset
        self.initSensors(self.numSensors, self.sensorHeight)

        # Controls
        self.arrowUpDown = False
        self.arrowLeftDown = False
        self.arrowRightDown = False
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

    def initSensors(self, n, h):
        origin = Point3(0.0, 0.0, h)
        angles = [math.pi * t / (n-1) for t in range(n)]
        sensors = [Vec3(math.sin(ang), math.cos(ang), h) for ang in angles]

        # Create collision nodes for sensors
        sensorNode = CollisionNode('sensor')
        sensorRay = CollisionRay(origin, Vec3(1.0, 0.0, 0.0))
        sensorNode.addSolid(sensorRay)

        # Add collision node to car and handler
        sensorNodePath = self.car.attachNewNode(sensorNode)
        self.base.cTrav.addCollider(sensorNodePath, self.collisionHandler)
        self.base.taskMgr.add(self.checkCollisions, "CheckCollisionsTask")

    def checkCollisions(self, task):
        for entry in self.collisionHandler.getEntries():
            pass

        return task.cont

    def getNodePath(self):
        return self.car

    def onUpArrow(self, down):
        self.arrowUpDown = down

    def onLeftArrow(self, down):
        self.arrowLeftDown = down

    def onRightArrow(self, down):
        self.arrowRightDown = down

    def degToRad(self, deg):
        return deg * (math.pi / 180.0)

    def radToDeg(self, rad):
        return rad * (180.0 / math.pi)

    def updateCar(self, task):
        # Register controls
        if self.arrowUpDown:
            self.speed = self.speedMax
        else:
            self.speed = 0.0
        if self.arrowLeftDown and not self.arrowRightDown:
            self.steerAngle = self.steerAngleMax
        if self.arrowRightDown and not self.arrowLeftDown:
            self.steerAngle = -self.steerAngleMax
        if not self.arrowLeftDown and not self.arrowRightDown:
            self.steerAngle = 0.0

        # Update car pose
        dt = task.time - self.lastUpdate
        self.lastUpdate = task.time

        wheelFrontUpdate = self.wheelFront + Vec3(math.cos(self.degToRad(self.steerAngle)),
                                                  math.sin(self.degToRad(self.steerAngle)), 0.0) * self.speed * dt
        wheelBackUpdate = self.wheelBack + Vec3(1.0, 0.0, 0.0) * self.speed * dt

        positionLocalUpdate = (wheelFrontUpdate + wheelBackUpdate) / 2.0
        headingLocalUpdate = math.atan2(wheelFrontUpdate.getY() - wheelBackUpdate.getY(),
                                        wheelFrontUpdate.getX() - wheelBackUpdate.getX())

        self.car.setPos(self.car, positionLocalUpdate)
        self.car.setH(self.car, self.radToDeg(headingLocalUpdate))
        self.wheelFrontLeft.setH(self.steerAngle)
        self.wheelFrontRight.setH(self.steerAngle)

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
        axisNodePath.setZ(axisNodePath, 1.0)
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
