from app.agent import run_agent
import uuid
#SESSION_ID = "local_chat_01"
SESSION_ID = str(uuid.uuid4())

if __name__ == "__main__":
    print(f"Session: {SESSION_ID}")
    while True:
        q = input("\n👤: ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        ans = run_agent(SESSION_ID, q)
        print("\n🤖:", ans)