Algorithm A2: Risk-aware A* pathfinding

Input: road graph, current location, candidate hospital, risk state, organ profile
Output: path cost to candidate hospital

identify start node nearest to current location
identify goal node nearest to candidate hospital
initialise open set with start node
initialise g_cost(start) = 0
initialise f_cost(start) = heuristic(start, goal)

while open set is not empty do
    current = node with lowest f_cost

    if current = goal then
        return g_cost(current)
    end if

    for each neighbour of current do
        risk_multiplier = function of risk level
        urgency_penalty = function of remaining safe time
        ml_penalty = escalation probability factor

        edge_cost =
            distance
            + reliability_penalty × risk_multiplier
            + urgency_penalty
            + ml_penalty

        new_cost = g_cost(current) + edge_cost

        if neighbour not visited before or new_cost < g_cost(neighbour) then
            update g_cost(neighbour)
            update f_cost(neighbour) = new_cost + heuristic(neighbour, goal)
            add neighbour to open set
        end if
    end for
end while

return infinity
