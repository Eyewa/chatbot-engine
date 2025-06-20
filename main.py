from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langserve import add_routes

from agent import build_agent


def create_app() -> FastAPI:
    load_dotenv()
    app = FastAPI(title="Eyewear Chatbot API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent = build_agent()
    add_routes(app, agent, path="/chatbot")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
