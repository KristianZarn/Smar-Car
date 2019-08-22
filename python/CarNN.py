from panda3d.core import (CollisionHandlerQueue, CollisionNode, CollisionRay,
                          CollisionSphere, LineSegs, Point3, Vec3, BitMask32)
import math
import numpy as np


class NeuralNetworkCar:

    def __init__(self, base):
        self.base = base

        # Parameters
        self.wheelBaseOffset = 3.4
        self.wheelSideOffset = 3.1
        self.speedMax = 50.0 / 30.0
        self.steerAngleMax = 15.0
        self.numSensors = 15
        self.sensorHeight = 2.0

        # Load car model
        self.car = self.base.loader.loadModel("models/carBody")
        self.carCollider = self.car.attachNewNode(CollisionNode("carCollider"))
        self.carCollider.node().addSolid(CollisionSphere(1.5, 0.0, 4.0, 4.0))
        self.carCollider.node().setFromCollideMask(BitMask32(0xF0))
        self.carCollider.setCollideMask(BitMask32(0xF00))
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
        self.checkpointCount = 0
        self.wheelFront = Vec3(1.0, 0.0, 0.0) * self.wheelBaseOffset
        self.wheelBack = Vec3(-1.0, 0.0, 0.0) * self.wheelBaseOffset
        self.sensorCollisionHandlers = []
        self.sensorDistances = []
        self.initSensors()

        # Controls
        self.arrowUpDown = False
        self.arrowLeftDown = False
        self.arrowRightDown = False

        # Read neural network parameters
        self.theta1 = np.genfromtxt('data/theta1.csv', delimiter=',')
        self.theta2 = np.genfromtxt('data/theta2.csv', delimiter=',')
        self.theta3 = np.genfromtxt('data/theta3.csv', delimiter=',')

        # Run task that checks for sensor collisions
        self.base.taskMgr.add(self.checkCollisions, "CheckCollisionsTask", priority=2)

        # Run task that computes inputs
        self.base.taskMgr.add(self.getInput, "getInputTask", priority=3)

        # Run task that updates car
        self.base.taskMgr.add(self.updateCar, "UpdateCarTask", priority=4)

        # DEBUG
        self.debugMode = False

    def getNodePath(self):
        return self.car

    def onCrash(self, entry):
        self.resetCar()

    def onCheckpoint(self, entry):
        self.checkpointCount += 1

    def initSensors(self):
        origin = Point3(0.0, 0.0, self.sensorHeight)
        angles = [(math.pi / 12.0) + ((10.0 / 12.0) * math.pi / (self.numSensors - 1)) * t for t in range(self.numSensors)]
        sensors = [Vec3(math.sin(ang), math.cos(ang), 0.0) for ang in angles]

        # Create collision handlers for sensors
        for i, sensor in enumerate(sensors):
            sensorNode = CollisionNode("sensor{}".format(i))
            sensorNode.setFromCollideMask(BitMask32(0xF))
            sensorRay = CollisionRay(origin, sensor)
            sensorNode.addSolid(sensorRay)
            sensorNodePath = self.car.attachNewNode(sensorNode)
            sensorNodePath.setCollideMask(BitMask32(0))

            handlerQueue = CollisionHandlerQueue()
            self.sensorCollisionHandlers.append(handlerQueue)
            self.base.cTrav.addCollider(sensorNodePath, handlerQueue)

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

    def getInput(self, task):
        if len(self.sensorDistances) == self.numSensors:
            # Feedforward
            x = np.array([self.sensorDistances])

            a1 = np.concatenate((np.ones((1, 1)), x.T))
            z2 = np.dot(self.theta1, a1)
            a2 = np.concatenate((np.ones((1, 1)), 1.0 / (1.0 + np.exp(-z2))))
            z3 = np.dot(self.theta2, a2)
            a3 = np.concatenate((np.ones((1, 1)), 1.0 / (1.0 + np.exp(-z3))))
            z4 = np.dot(self.theta3, a3)

            a4 = 1.0 / (1.0 + np.exp(-z4))

            pred = np.argmax(a4)

            # Set input
            self.arrowUpDown = True
            if pred == 0:
                self.arrowLeftDown = False
                self.arrowRightDown = True
            elif pred == 1:
                self.arrowLeftDown = True
                self.arrowRightDown = False

        return task.cont

    def resetCar(self):
        self.steerAngle = 0.0
        self.speed = 0.0
        self.checkpointCount = 0
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
        dt = 1.0
        wheelFrontUpdate = self.wheelFront + Vec3(math.cos(math.radians(self.steerAngle)),
                                                  math.sin(math.radians(self.steerAngle)), 0.0) * self.speed * dt
        wheelBackUpdate = self.wheelBack + Vec3(1.0, 0.0, 0.0) * self.speed * dt

        positionLocalUpdate = (wheelFrontUpdate + wheelBackUpdate) / 2.0
        headingLocalUpdate = math.atan2(wheelFrontUpdate.getY() - wheelBackUpdate.getY(),
                                        wheelFrontUpdate.getX() - wheelBackUpdate.getX())

        self.car.setPos(self.car, positionLocalUpdate)
        self.car.setH(self.car, math.degrees(headingLocalUpdate))
        self.wheelFrontLeft.setH(self.steerAngle)
        self.wheelFrontRight.setH(self.steerAngle)

        return task.cont
