

import json
import openai
# Assuming the import paths are correct based on your instruction
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage

class ReplayClient:
    def __init__(self, file_path, original_client, raise_on_empty=False):
        self.file_path = file_path
        self.original_client = original_client
        self.raise_on_empty = raise_on_empty
        self.conversation_history = self._load_conversation_history()
        self.message_index = 0

    def _load_conversation_history(self):
        with open(self.file_path, 'r') as file:
            return json.load(file)

    def mk_chat(self, message):
        # Constructing ChatCompletionMessage from the message dictionary
        chat_completion_message = ChatCompletionMessage(**message)
        # Creating a ChatCompletion object using the message
        chat_completion = ChatCompletion(
            id='A',
            created=0,
            model='A',
            choices=[{'finish_reason': 'stop', 'index': 0, 'message': chat_completion_message}],
            object='chat.completion'
        )
        return chat_completion

    def chat_completions_create(self, **args):
        while True:
            if self.message_index < len(self.conversation_history):
                message = self.conversation_history[self.message_index]
                self.message_index += 1
                yield self.mk_chat(message)
            else:
                if self.raise_on_empty:
                    raise StopIteration("No more messages in the file and raise_on_empty is set.")
                else:
                    yield from self._delegate_to_original_client(**args)
                    break

    def _delegate_to_original_client(self, **args):
        response = self.original_client.ChatCompletion.create(**args)
        yield response


if __name__ == "__main__":
    # Example usage
    original_client = openai.OpenAI(api_key="your_api_key")
    replay_client = ReplayClient('conversation_history.json', original_client, raise_on_empty=True)

    # Example of iterating over the generator
    try:
        for chat_completion in replay_client.chat_completions_create(model="text-davinci-003", prompt="Hello, world!"):
            print(chat_completion)
    except StopIteration as e:
        print(f"Generator stopped: {e}")