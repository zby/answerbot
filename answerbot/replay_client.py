import json
import openai
# Assuming the import paths are correct based on your instruction
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage

class MessagesExhausted(Exception):
    """Exception raised when there are no more messages to replay."""
    pass

def user_message_predicate(message):
    return message['role'] == 'assistant'


class LLMReplayClient:
    def __init__(self, file_path, original_client=None, filter_predicate=None):
        self.file_path = file_path
        self.original_client = original_client
        self.filter_predicate = user_message_predicate if filter_predicate is None else filter_predicate
        self.conversation_history = self._load_conversation_history()
        self.message_index = 0
        # Directly assign chat_completions_create to the desired access path
        self.chat = lambda: None  # Placeholder for the chat attribute
        self.chat.completions = lambda: None  # Placeholder for the completions attribute
        self.chat.completions.create = self.chat_completions_create

    def chat_completions_create(self, **args):
        if self.message_index < len(self.conversation_history):
            message = self.conversation_history[self.message_index]
            self.message_index += 1
            return self.mk_chat(message)
        else:
            if self.original_client is None:
                raise MessagesExhausted("No more messages in the file and no backup client to delegate to.")
            else:
                return self._delegate_to_original_client()

    def _load_conversation_history(self):
        with open(self.file_path, 'r') as file:
            all_messages = json.load(file)
        messages = []
        for message in all_messages:
            if self.filter_predicate(message):
                messages.append(message)
        return messages

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


    def _delegate_to_original_client(self, **args):
        response = self.original_client.chat.completions.create(**args)
        return response

 
if __name__ == "__main__":
    # Example usage
    original_client = openai.OpenAI(api_key="your_api_key")
    replay_client = LLMReplayClient('data/conversation.json')

    # Example of iterating over the generator
    try:
        while True:
            chat_completion = replay_client.chat.completions.create(model="text-davinci-003", prompt="Hello, world!")
            print(chat_completion)
    except MessagesExhausted as e:
        print(f"Generator stopped: {e}")