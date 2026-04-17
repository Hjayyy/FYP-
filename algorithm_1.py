Algorithm A1: Simulation timestep update

Input: current simulation state
Output: updated simulation state

transport_state = TransportSimulator.step()
temperature = TemperatureSimulator.step(transport_state)

risk_state = RiskEngine.evaluate(temperature, transport_state)

if ML predictor is enabled then
    escalation_probability = MLRiskPredictor.predict(organ, temperature, transport_state, risk_state)
    attach escalation_probability to risk_state

    if escalation_probability >= 0.90 then
        set risk_state.level = CRITICAL
        set risk_state.requires_intervention = True
    else if escalation_probability >= 0.75 and risk_state.level in {LOW, MODERATE} then
        set risk_state.level = HIGH
        set risk_state.requires_intervention = True
    end if
end if

if risk_state.requires_intervention then
    apply cooling intervention
end if

if remaining_safe_minutes <= 0 after initial transport period then
    terminate simulation
end if

if elapsed transport time >= organ maximum transport time then
    terminate simulation
end if

if reroute cooldown is satisfied then
    if risk_state.level in {HIGH, CRITICAL}
       or remaining_safe_minutes <= organ denature margin then

        candidate_hospitals = nearby hospitals excluding start and current destination
        best_hospital, candidate_cost = select_optimal_hospital(...)

        if current_cost is undefined or candidate_cost provides meaningful improvement then
            update transport destination
            update reroute state
        end if
    end if
end if

log simulation state

if transport_state.status == ARRIVED then
    terminate simulation
end if

return latest state for dashboard update
                   
