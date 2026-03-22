from agent import agent


def main():
    print("Research Agent with RAG (type 'exit' to quit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        answer = agent.run(user_input)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()