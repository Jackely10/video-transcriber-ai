import os
from dotenv import load_dotenv, find_dotenv
from anthropic import Anthropic

DOTENV_PATH = find_dotenv()
print(f"Dotenv path: {DOTENV_PATH!r}")
load_dotenv(DOTENV_PATH, override=True)

print('=' * 60)
print('TESTING ANTHROPIC API CONNECTION')
print('=' * 60)

api_key = os.getenv('ANTHROPIC_API_KEY', '')
model = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')

print('Configuration:')
print(f'  - API Key set: {bool(api_key)}')
print(f'  - API Key length: {len(api_key)} chars')
print(f'  - API Key prefix: {api_key[:20] + "..." if api_key else "N/A"}')
print(f'  - Model: {model}')
print(f'  - Raw env value: {os.environ.get("ANTHROPIC_API_KEY")!r}')

try:
    print('\nCreating Anthropic client...')
    client = Anthropic(api_key=api_key)
    print('Client created successfully!')

    print(f'Testing API call with model: {model}')
    print('Sending simple test message...')

    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[
            {
                'role': 'user',
                'content': 'Bitte bestaetige in einem kurzen Satz, dass diese Testanfrage funktioniert.'
            }
        ],
    )

    print('\nAPI call succeeded!')
    if response and response.content:
        text_parts = []
        for part in response.content:
            if hasattr(part, 'text'):
                text_parts.append(part.text)
            elif isinstance(part, dict) and part.get('type') == 'text':
                text_parts.append(part.get('text', ''))
        message = '\n'.join(p.strip() for p in text_parts if p)
        print('Response:')
        print(message or '[No text returned]')
    else:
        print('Response contained no content.')
except Exception as exc:
    print('\nAPI call failed!')
    print(f'Error type: {type(exc).__name__}')
    print(f'Error message: {exc}')
    if hasattr(exc, "status_code"):
        print(f'Status code: {exc.status_code}')
    if hasattr(exc, "response"):
        print(f'Response object: {exc.response}')
