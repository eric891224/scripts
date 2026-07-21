export API_URL=http://140.112.90.45:8002/v1
export API_KEY=
export SERVED_MODEL_ID=openai/gpt-oss-120b

curl $API_URL/chat/completions \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$SERVED_MODEL_ID"'",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in Taipei?"}
        ],
        "temperature": 0.7,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]
    }'