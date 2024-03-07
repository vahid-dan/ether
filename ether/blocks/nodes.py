import re
import itertools
from collections import defaultdict
from typing import Dict, Optional

from ether.core import Node, Capacity


__size_conversions = {
    'K': 10 ** 3,
    'M': 10 ** 6,
    'G': 10 ** 9,
    'T': 10 ** 12,
    'P': 10 ** 15,
    'E': 10 ** 18,
    'Ki': 2 ** 10,
    'Mi': 2 ** 20,
    'Gi': 2 ** 30,
    'Ti': 2 ** 40,
    'Pi': 2 ** 50,
    'Ei': 2 ** 60
}

__size_pattern = re.compile(r"([0-9]+)([a-zA-Z]*)")


def parse_size_string(size_string: str) -> int:
    m = __size_pattern.match(size_string)
    if len(m.groups()) > 1:
        number = m.group(1)
        unit = m.group(2)
        return int(number) * __size_conversions.get(unit, 1)
    else:
        return int(m.group(1))


def to_size_string(num_bytes, unit='M', precision=1) -> str:
    factor = __size_conversions[unit]
    value = num_bytes / factor

    fmt = f'%0.{precision}f{unit}'

    return fmt % value


counters = defaultdict(lambda: itertools.count(0, 1))


def create_vm_node(name=None) -> Node:
    name = name if name is not None else 'cloudvm_%d' % next(counters['cloudvm'])

    return create_node(name=name,
                       cpus=4, arch='x86', mem='8167784Ki',
                       labels={
                           'ether.edgerun.io/type': 'vm',
                           'ether.edgerun.io/model': 'vm'
                       })


def create_server_node(name=None, location_id=None, workload_quota=0) -> Node:
    name = name if name is not None else 'server_%d' % next(counters['server'])

    return create_node(name=name,
                       cpus=88, arch='x86', mem='188G',
                       labels={
                           'ether.edgerun.io/type': 'server',
                           'ether.edgerun.io/model': 'server'
                       },
                       location_id=location_id,
                       workload_quota=workload_quota)


def create_cloud_server_node(name=None) -> Node:
    name = name if name is not None else 'cloud_server_%d' % next(counters['cloud_server'])

    return create_node(name=name,
                       cpus=96, arch='x86', mem='256G',
                       labels={
                           'ether.edgerun.io/type': 'cloud_server',
                           'ether.edgerun.io/model': 'cloud_server'
                       })


def create_rpi3_node(name=None) -> Node:
    name = name if name is not None else 'rpi3_%d' % next(counters['rpi3'])

    return create_node(name=name,
                       cpus=4, arch='arm32', mem='999036Ki',
                       labels={
                           'ether.edgerun.io/type': 'sbc',
                           'ether.edgerun.io/model': 'rpi3b+'
                       })


def create_nuc_node(name=None) -> Node:
    name = name if name is not None else 'nuc_%d' % next(counters['nuc'])

    return create_node(name=name,
                       cpus=4, arch='x86', mem='16Gi',
                       labels={
                           'ether.edgerun.io/type': 'sffc',
                           'ether.edgerun.io/model': 'nuci5'
                       })


def create_tx2_node(name=None) -> Node:
    name = name if name is not None else 'tx2_%d' % next(counters['tx2'])

    return create_node(name=name,
                       cpus=4, arch='aarch64', mem='8047252Ki',
                       labels={
                           'ether.edgerun.io/type': 'embai',
                           'ether.edgerun.io/model': 'nvidia_jetson_tx2',
                           'ether.edgerun.io/capabilities/cuda': '10',
                           'ether.edgerun.io/capabilities/gpu': 'pascal',
                       })


def create_rockpi(name=None) -> Node:
    name = name if name is not None else 'rockpi_%d' % next(counters['rockpi'])

    return create_node(name=name,
                       cpus=6, arch='aarch64', mem='4G',
                       labels={
                           'ether.edgerun.io/type': 'sbc',
                           'ether.edgerun.io/model': 'rockpi4'
                       })


def create_rpi4_node(name=None) -> Node:
    name = name if name is not None else 'rpi4_%d' % next(counters['rpi4'])

    return create_node(name=name,
                       arch='arm32v7',
                       cpus=4,
                       mem='1G',
                       labels={
                           'ether.edgerun.io/type': 'sbc',
                           'ether.edgerun.io/model': 'rpi4',
                       })


def create_coral(name=None) -> Node:
    name = name if name is not None else 'coral_%d' % next(counters['coral'])

    return create_node(name=name,
                       cpus=4, arch='aarch64', mem='1G',
                       labels={
                           'ether.edgerun.io/type': 'sbc',
                           'ether.edgerun.io/model': 'coral',
                           'ether.edgerun.io/capabilities/tpu': 'edgetpu',
    })


def create_nano(name=None) -> Node:
    name = name if name is not None else 'nano_%d' % next(counters['nano'])

    return create_node(name=name,
                       cpus=4, arch='aarch64', mem='4G',
                       labels={
                           'ether.edgerun.io/type': 'embai',
                           'ether.edgerun.io/model': 'nvidia_jetson_nano',
                           'ether.edgerun.io/capabilities/cuda': '10',
                           'ether.edgerun.io/capabilities/gpu': 'maxwell',
                       })


def create_nx(name=None) -> Node:
    name = name if name is not None else 'nx_%d' % next(counters['nx'])

    return create_node(name=name,
                       cpus=6, arch='aarch64', mem='8G',
                       labels={
                           'ether.edgerun.io/type': 'embai',
                           'ether.edgerun.io/model': 'nvidia_jetson_nx',
                           'ether.edgerun.io/capabilities/cuda': '10',
                           'ether.edgerun.io/capabilities/gpu': 'volta',
                       })


def create_node(name: str, cpus: int, mem: str, arch: str, labels: Dict[str, str],
                location_id: Optional[str] = None, workload_quota: Optional[int] = 0) -> Node:
    capacity = Capacity(cpu_millis=cpus * 1000, memory=parse_size_string(mem))
    node = Node(name=name, capacity=capacity, arch=arch, labels=labels)
    if location_id:
        node.location_id = location_id
    if workload_quota:
        node.workload_quota = workload_quota
    return node


rpi3 = create_rpi3_node
nuc = create_nuc_node
tx2 = create_tx2_node
server = create_server_node
nx = create_nx
nano = create_nano
coral = create_coral
rpi4 = create_rpi4_node
rockpi = create_rockpi
