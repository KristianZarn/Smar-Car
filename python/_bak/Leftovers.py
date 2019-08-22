# Main class:
        # DEBUG
        # Draw axis
        #self.drawAxis(10)

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
        axisNodePath = self.render.attachNewNode(axisNode)
        return axisNodePath

# Car class
#self.drawAxis(10)
#self.drawWheelBase()

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

# OTHER
#carCollider.show()
#angles = [math.pi * t / (n-1) for t in range(n)]
#print " ".join('%0.2f' % d for d in self.sensorDistances)