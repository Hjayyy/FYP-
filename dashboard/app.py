from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
from simulation_engine import TransportSimulation

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

simulation = None
simulation_running = False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global simulation, simulation_running

    await websocket.accept()

    simulation = TransportSimulation(
        organ_key="heart",
        ml_model_path="heart_model.pkl"
    )

    simulation_task = None

    async def run_simulation():
        global simulation_running
        while simulation_running:
            simulation.step()
            state = simulation.get_state()
            await websocket.send_json(state)

            if state["completed"]:
                simulation_running = False
                break

            await asyncio.sleep(1)

    try:
        while True:
            data = await websocket.receive_text()

            if data == "START":

                if not simulation_running:
                    simulation_running = True
                    simulation_task = asyncio.create_task(run_simulation())

            elif data == "PAUSE":
                simulation_running = False

            elif data == "RESET":

                simulation_running = False

                if simulation_task:
                    simulation_task.cancel()
                    simulation_task = None

                simulation = TransportSimulation(
                    organ_key="heart",
                    ml_model_path="heart_model.pkl"
                )

                await websocket.send_json(simulation.get_state())

    except WebSocketDisconnect:
        simulation_running = False
