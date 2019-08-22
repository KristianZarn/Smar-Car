from direct.showbase.DirectObject import DirectObject
from panda3d.core import (CollisionHandlerQueue, CollisionHandlerEvent, CollisionNode, CollisionRay, CollisionBox,
                          CollisionSphere, LineSegs, Point3, Vec3, BitMask32)
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
        self.numSensors = 15
        self.sensorHeight = 2.0

        # Collision handler
        self.carCollisionHandler = CollisionHandlerEvent()
        self.carCollisionHandler.addInPattern("%fn-into-%in")

        # Load car model
        self.car = self.base.loader.loadModel("models/carBody")
        carCollider = self.car.attachNewNode(CollisionNode("carCollider"))
        carCollider.node().addSolid(CollisionSphere(1.5, 0.0, 4.0, 4.0))
        carCollider.setCollideMask(BitMask32(0))
        self.base.cTrav.addCollider(carCollider, self.carCollisionHandler)
        self.accept("carCollider-into-trackCollision", self.onCrash)
        self.accept("carCollider-into-checkpoints", self.onCheckpoint)
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
        self.base.taskMgr.add(self.updateCar, "UpdateCarTask", priority=3)

        # DEBUG
        self.debugMode = False
        self.accept("d", self.debugHandler)

    def initSensors(self, n, h):
        origin = Point3(0.0, 0.0, h)
        angles = [(math.pi/6.0) + ((2.0/3.0)*math.pi/(n-1)) * t for t in range(n)]
        sensors = [Vec3(math.sin(ang), math.cos(ang), 0.0) for ang in angles]

        # Create collision handlers for sensors
        self.sensorCollisionHandlers = []
        for i, sensor in enumerate(sensors):
            sensorNode = CollisionNode("sensor{}".format(i))
            sensorRay = CollisionRay(origin, sensor)
            sensorNode.addSolid(sensorRay)
            sensorNodePath = self.car.attachNewNode(sensorNode)
            sensorNodePath.setCollideMask(BitMask32(0))

            handlerQueue = CollisionHandlerQueue()
            self.sensorCollisionHandlers.append(handlerQueue)
            self.base.cTrav.addCollider(sensorNodePath, handlerQueue)

        # Start task that checks collisions
        self.base.taskMgr.add(self.checkCollisions, "CheckCollisionsTask", priority=2)

    def checkCollisions(self, task):
        sensorPoints = []
        self.sensorDistances = []
        for queue in self.sensorCollisionHandlers:
            if queue.getNumEntries() > 0:
                queue.sortEntries()
                entry = queue.getEntry(0)
                sensorPoint = entry.getSurfacePoint(entry.getFromNodePath())
                sensorPoints.append(sensorPoint)
                tmp = sensorPoint - Vec3(0.0, 0.0, self.sensorHeight)
                self.sensorDistances.append(tmp.length())

        # DEBUG
        tmp = self.car.find("sensorsSegs")
        if not tmp.isEmpty():
            tmp.removeNode()
        if self.debugMode:
            sensorSegs = LineSegs("sensorsSegs")
            sensorSegs.setColor(1.0, 0.45, 0.0)
            sensorSegs.setThickness(4)
            for point in sensorPoints:
                sensorSegs.moveTo(Point3(0.0, 0.0, self.sensorHeight))
                sensorSegs.drawTo(point)
            sensorNode = sensorSegs.create()
            self.car.attachNewNode(sensorNode)

        return task.cont

    def getNodePath(self):
        return self.car

    def onCrash(self, entry):
        self.resetCar()

    def onCheckpoint(self, entry):
        print "checkpoint"

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

    def resetCar(self):
        self.car.setPos(0.0, 0.0, 0.0)
        self.car.setH(0.0)

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

    def debugHandler(self):
        self.debugMode = not self.debugMode
