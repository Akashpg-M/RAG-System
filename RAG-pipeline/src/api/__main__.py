import uvicorn


if __name__ == "__main__":
    uvicorn.run("src.api.app:create_api", factory=True, host="127.0.0.1", port=8000)

