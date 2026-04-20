# FastAPI backend for real-time simulation control and dashboard updates.
# Provides the main dashboard page and a WebSocket endpoint for simulation commands
# and live state streaming.

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio

from simulation_engine import TransportSimulation

# Maps each organ type to its trained escalation prediction model.
MODEL_MAP = {
    "heart": "heart_gboost.pkl",
    "kidney": "kidney_gboost.pkl",
    "lungs": "lungs_gboost.pkl"
}

app = FastAPI()

# Serve static dashboard files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Return the main monitoring dashboard page
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# WebSocket endpoint for real-time control commands and dashboard updates.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    simulation = None
    simulation_running = False
    simulation_task = None


    # Background task that advances the simulation and streams updated state
    # to the dashboard at fixed time intervals.
    
    async def run_simulation():

        nonlocal simulation_running, simulation

        try:
            while simulation_running and simulation:

                simulation.step()

                state = simulation.get_state()

                # Send updated state to the frontend
                await websocket.send_json(state)

                # Stop when simulation completes
                if state["completed"]:
                    simulation_running = False
                    break

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass


    try:
        while True:

            # Receive dashboard commands (START, PAUSE, RESET).
            data = await websocket.receive_text()


            # START simulation
            if data.startswith("START"):
                parts = data.split(":")
                organ = parts[1]
                start = parts[2]
                hospital = parts[3]

                model_path = MODEL_MAP.get(organ)

                # Stop any existing simulation before starting a new one.
                simulation_running = False
                if simulation_task:
                    simulation_task.cancel()
                    try:
                        await simulation_task
                    except asyncio.CancelledError:
                        pass
                    simulation_task = None

                # Create new simulation
                simulation = TransportSimulation(
                    organ_key=organ,
                    start_name=start,
                    destination_name=hospital,
                    ml_model_path=model_path
                )

                initial_state = simulation.get_state()
                await websocket.send_json(initial_state)
                # Start new simulation loop
                simulation_running = True
                simulation_task = asyncio.create_task(run_simulation())

            # PAUSE simulation
            elif data == "PAUSE":
                simulation_running = False

            # RESET simulation
            elif data == "RESET":

                simulation_running = False

                if simulation_task:
                    simulation_task.cancel()

                    try:
                        await simulation_task
                    except asyncio.CancelledError:
                        pass

                    simulation_task = None

                simulation = None

    # Handle client disconnects and stops any simulation task
    except WebSocketDisconnect:

        simulation_running = False

        if simulation_task:
            simulation_task.cancel()
