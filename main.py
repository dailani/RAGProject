from chat_model.translator import translate_text
from retreival.retreival import retreive


def response_chain(user_input):
    print(f"INFO:: Calling translate_text...")
    try:
        translated_user_input = translate_text(user_input)
        response = retreive(translated_user_input)
        return response["answer"]
    except Exception as e:
        print(f"INFO:: Exception in response_chain:", e)
        return "Sorry, there was an error processing your input."


def main():
    print("Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        print(f"INFO:: Processing input:", user_input)
        response = response_chain(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
