# Synthetic road network and risk-aware pathfinding used by the routing engine

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Dict, List

from models import Coord, RiskState, OrganProfile, Hospital
from transport_simulator import haversine

# Node within the synthetic road graph.
@dataclass
class RoadNode:
    id: int
    coord: Coord

# Edge within the synthetic road graph.
# Reliability penalty represents how delay-prone the route is.
@dataclass
class RoadEdge:
    to_id: int
    distance_km: float
    reliability_penalty: float  # higher = more delay-prone


class RoadGraph:
   

    def __init__(self):
        self.nodes: Dict[int, RoadNode] = {}
        self.edges: Dict[int, List[RoadEdge]] = {}

    def add_node(self, node_id: int, coord: Coord) -> None:
        self.nodes[node_id] = RoadNode(node_id, coord)
        self.edges[node_id] = []

    def add_edge(self, a: int, b: int) -> None:
        ca = self.nodes[a].coord
        cb = self.nodes[b].coord
        d = haversine(ca, cb)
        reliability = random.uniform(0.0, 5.0)
        self.edges[a].append(RoadEdge(b, d, reliability))
        self.edges[b].append(RoadEdge(a, d, reliability))

# Build a synthetic road graph spanning the hospital network.
def build_synthetic_graph(hospitals, num_nodes: int = 160) -> RoadGraph:
    g = RoadGraph()

    lats = [h.location[0] for h in hospitals]
    lons = [h.location[1] for h in hospitals]

    min_lat, max_lat = min(lats) - 0.2, max(lats) + 0.2
    min_lon, max_lon = min(lons) - 0.2, max(lons) + 0.2

    # hospital bounding box
    for i in range(num_nodes):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        g.add_node(i, (lat, lon))

    # Connect each node to nearest neighbours
    node_ids = list(g.nodes.keys())
    for a in node_ids:
        candidates = sorted(
            node_ids,
            key=lambda b: haversine(g.nodes[a].coord, g.nodes[b].coord)
        )[1:5]
        for b in candidates:
            g.add_edge(a, b)

    return g

# Compute the risk-aware cost of traversing an edge.
# Route cost increases with delay-proneness, urgency, and escalation risk.
def _edge_cost(edge: RoadEdge, risk: RiskState, organ: OrganProfile) -> float:
    
    risk_multiplier = 1.0
    if risk.risk_level == "MODERATE":
        risk_multiplier = 1.3
    elif risk.risk_level == "HIGH":
        risk_multiplier = 1.8
    elif risk.risk_level == "CRITICAL":
        risk_multiplier = 2.5

    reliability_cost = edge.reliability_penalty * risk_multiplier

    urgency = 1.0 - min(1.0, risk.remaining_safe_minutes / max(1, organ.max_transport_minutes))
    urgency_penalty = urgency * 5.0
    ml_penalty = float(risk.escalation_prob) * 3.0

    return edge.distance_km + reliability_cost + urgency_penalty + ml_penalty

# Return the total path cost from the current position to a hospital
# using risk-aware A* search.
def risk_aware_shortest_path(
    graph: RoadGraph,
    start_coord: Coord,
    hospital: Hospital,
    risk: RiskState,
    organ: OrganProfile,
) -> float:
    
    start_id = min(graph.nodes, key=lambda i: haversine(start_coord, graph.nodes[i].coord))
    goal_id = min(graph.nodes, key=lambda i: haversine(hospital.location, graph.nodes[i].coord))

    open_set: List[tuple[float, int]] = [(0.0, start_id)]
    g_cost: Dict[int, float] = {start_id: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_id:
            return g_cost[current]

        for edge in graph.edges[current]:
            new_cost = g_cost[current] + _edge_cost(edge, risk, organ)

            if edge.to_id not in g_cost or new_cost < g_cost[edge.to_id]:
                g_cost[edge.to_id] = new_cost

                heuristic = haversine(graph.nodes[edge.to_id].coord, hospital.location)
                heapq.heappush(open_set, (new_cost + heuristic, edge.to_id))

    return float("inf")