import itertools
from collections import defaultdict

from ether.blocks.nodes import create_server_node, create_cloud_server_node
from ether.cell import LANCell, UpDownLink
from ether.qos import latency

counters = defaultdict(lambda: itertools.count(0, 1))


class MobileConnection(UpDownLink):

    def __init__(self, backhaul='internet') -> None:
        super().__init__(125, 25, backhaul, latency.mobile_isp)


class BusinessIsp(UpDownLink):

    def __init__(self, backhaul='internet') -> None:
        super().__init__(500, 50, backhaul, latency.business_isp)


class FiberToExchange(UpDownLink):

    def __init__(self, backhaul='internet') -> None:
        super().__init__(1000, 1000, backhaul, latency.lan)


class IoTComputeBox(LANCell):
    pass


class Cloudlet(LANCell):
    def __init__(self, server_per_rack=5, racks=1, backhaul=None, location_id=None, workload_quota=0) -> None:
        self.racks = racks
        self.server_per_rack = server_per_rack
        self.location_id = location_id
        self.workload_quota = workload_quota

        self._create_identity()
        super().__init__(nodes=[self._create_rack() for _ in range(racks)], 
                         backhaul=backhaul, 
                         location_id=location_id,
                         workload_quota=workload_quota)

    def _create_identity(self):
        self.nr = next(counters['cloudlet'])
        self.name = 'cloudlet_%d' % self.nr
        self.switch = 'switch_%s' % self.name

    def _create_rack(self):
        return LANCell([create_server_node(location_id=self.location_id, workload_quota=self.workload_quota) for _ in range(self.server_per_rack)], backhaul=self.switch)


class Cloud(LANCell):
    def __init__(self, server_per_rack=4, racks=4, backhaul=None, workload_quota=float('inf')) -> None:
        self.racks = racks
        self.server_per_rack = server_per_rack
        self.workload_quota = workload_quota

        self._create_identity()
        super().__init__(nodes=[self._create_rack() for _ in range(racks)], 
                         backhaul=backhaul,
                         workload_quota=workload_quota)

    def _create_identity(self):
        self.nr = next(counters['cloud'])
        self.name = 'cloud_%d' % self.nr
        self.switch = 'switch_%s' % self.name

    def _create_rack(self):
        return LANCell([create_cloud_server_node(workload_quota=self.workload_quota) for _ in range(self.server_per_rack)], backhaul=self.switch)
    