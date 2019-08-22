from panda3d.core import (CollisionHandlerQueue, CollisionNode, CollisionRay,
                          CollisionSphere, Point3, Vec3, BitMask32)
import math
import numpy as np


class NeuroevolutionCar:

    def __init__(self, base, label, numSensors):
        self.base = base

        # Parameters
        self.wheelBaseOffset = 3.4
        self.wheelSideOffset = 3.1
        self.speedMax = 50.0 / 30.0
        self.steerAngleMax = 15.0
        self.numSensors = numSensors
        self.sensorHeight = 2.0

        # Load car model
        self.car = self.base.loader.loadModel("models/carBody")
        self.carCollider = self.car.attachNewNode(CollisionNode("carCollider{}".format(label)))
        self.carCollider.node().addSolid(CollisionSphere(1.5, 0.0, 4.0, 4.0))
        self.carCollider.node().setFromCollideMask(BitMask32(0xF0))
        self.carCollider.setCollideMask(BitMask32(0))
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
        self.simulationDone = True
        self.timeAlive = 0
        self.checkpointCount = 0

        self.steerAngle = 0.0
        self.speed = 0.0
        self.sensorDistances = []

        self.wheelFront = Vec3(1.0, 0.0, 0.0) * self.wheelBaseOffset
        self.wheelBack = Vec3(-1.0, 0.0, 0.0) * self.wheelBaseOffset
        self.sensorCollisionNodes = []
        self.sensorCollisionHandlers = []
        self.initSensors()

        # Neural network parameters
        self.theta1 = None
        self.theta2 = None
        self.theta3 = None

        # Controls
        self.arrowUpDown = False
        self.arrowLeftDown = False
        self.arrowRightDown = False

        # Run task that checks for sensor collisions
        self.base.taskMgr.add(self.checkCollisions, "CheckCollisionsTask", priority=2)

        # Run task that computes inputs
        self.base.taskMgr.add(self.getInput, "GetInputTask", priority=3)

        # Run task that updates car
        self.base.taskMgr.add(self.updateCar, "UpdateCarTask", priority=4)

    def getNodePath(self):
        return self.car

    def startSimulation(self, theta1, theta2, theta3):
        self.simulationDone = False
        self.steerAngle = 0.0
        self.speed = 0.0
        self.timeAlive = 0
        self.checkpointCount = 0
        self.car.setPos(0.0, 0.0, 0.0)
        self.car.setH(0.0)
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        # Enable sensors
        for sensorNode in self.sensorCollisionNodes:
            sensorNode.setFromCollideMask(BitMask32(0xF))

    def onCrash(self, entry):
        self.simulationDone = True
        # Disable sensors
        for sensorNode in self.sensorCollisionNodes:
            sensorNode.setFromCollideMask(BitMask32(0x0))

    def onCheckpoint(self, entry):
        self.checkpointCount += 1

    def initSensors(self):
        origin = Point3(0.0, 0.0, self.sensorHeight)
        angleStart = (math.pi / 12.0)
        angleStep = ((10.0 / 12.0) * math.pi / (self.numSensors - 1))
        angles = [angleStart + angleStep * t for t in range(self.numSensors)]
        sensors = [Vec3(math.sin(ang), math.cos(ang), 0.0) for ang in angles]

        # Create collision handlers for sensors
        for i, sensor in enumerate(sensors):
            sensorNode = CollisionNode("sensor{}".format(i))
            self.sensorCollisionNodes.append(sensorNode)
            sensorNode.setFromCollideMask(BitMask32(0xF))
            sensorRay = CollisionRay(origin, sensor)
            sensorNode.addSolid(sensorRay)
            sensorNodePath = self.car.attachNewNode(sensorNode)
            sensorNodePath.setCollideMask(BitMask32(0))

            handlerQueue = CollisionHandlerQueue()
            self.sensorCollisionHandlers.append(handlerQueue)
            self.base.cTrav.addCollider(sensorNodePath, handlerQueue)

    def checkCollisions(self, task):
        if not self.simulationDone:
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

        return task.cont

    def getInput(self, task):
        if not self.simulationDone:
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

    def updateCar(self, task):
        if not self.simulationDone:
            self.timeAlive += 1

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
            wheelFrontUpdate = self.wheelFront + Vec3(math.cos(math.radians(self.steerAngle)),
                                                      math.sin(math.radians(self.steerAngle)), 0.0) * self.speed
            wheelBackUpdate = self.wheelBack + Vec3(1.0, 0.0, 0.0) * self.speed

            positionLocalUpdate = (wheelFrontUpdate + wheelBackUpdate) / 2.0
            headingLocalUpdate = math.atan2(wheelFrontUpdate.getY() - wheelBackUpdate.getY(),
                                            wheelFrontUpdate.getX() - wheelBackUpdate.getX())

            self.car.setPos(self.car, positionLocalUpdate)
            self.car.setH(self.car, math.degrees(headingLocalUpdate))
            self.wheelFrontLeft.setH(self.steerAngle)
            self.wheelFrontRight.setH(self.steerAngle)

        return task.cont
