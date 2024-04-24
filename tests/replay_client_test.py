import pytest
from answerbot.replay_client import ReplayClient, MessagesExhausted
from unittest.mock import MagicMock
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage

def test_replay_client_conversation_history():
    # Initialize ReplayClient with a test JSON file
    test_file_path = 'tests/data/conversation.json'
    replay_client = ReplayClient(test_file_path)
    
    # Load the conversation history from the file
    conversation_history = replay_client.conversation_history
    
    # Check if the conversation history is loaded correctly
    assert isinstance(conversation_history, list), "Conversation history should be a list"
    assert len(conversation_history) > 0, "Conversation history should not be empty"
    
    # Check the structure of the conversation messages
    for message in conversation_history:
        assert message['role'] == 'assistant'

def test_replay_client_response_generation():
    # Initialize ReplayClient with a test JSON file
    test_file_path = 'tests/data/conversation.json'
    replay_client = ReplayClient(test_file_path)
    
    # Simulate conversation by generating responses
    with pytest.raises(MessagesExhausted):
        while True:
            response = replay_client.chat.completions.create()
            assert isinstance(response, ChatCompletion), "Response should be ChatCompletion"
            message = response.choices[0].message
            assert isinstance(message, ChatCompletionMessage), "Message should be a ChatCompletionMessage"


def test_replay_client_delegation():
    # Initialize ReplayClient with a test JSON file and a mock original client
    test_file_path = 'tests/data/conversation.json'
    original_client = MagicMock()
    original_client.chat = MagicMock()
    original_client.chat.completions = MagicMock()
    original_client.chat.completions.create = MagicMock(return_value={'message': 'Delegated response'})
    
    replay_client = ReplayClient(test_file_path, original_client=original_client)
    
    # Exhaust the replay client's internal messages
    for _ in replay_client.conversation_history:
        replay_client.chat.completions.create()
        
    # Now, the next call should delegate to the original client
    response = replay_client.chat.completions.create()
    original_client.chat.completions.create.assert_called_once()
    assert response == {'message': 'Delegated response'}, "Response should be delegated from the original client"
