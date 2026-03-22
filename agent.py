from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from config import Settings, SYSTEM_PROMPT
from tools import web_search, read_url, write_report, knowledge_search


settings = Settings()
client = OpenAI(api_key=settings.api_key.get_secret_value())


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
    "knowledge_search": knowledge_search,
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for recent or external information. "
                "Use this first when the question requires internet research."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Read and extract the main content from a webpage URL. "
                "Use after web_search when a result looks relevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to read.",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": (
                "Save a Markdown report to the output directory. "
                "Use when the user asks for a report, summary file, or saved notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The output markdown filename.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The markdown content to save.",
                    },
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": (
                "Search the local knowledge base using hybrid retrieval "
                "(semantic + BM25) with reranking. "
                "Use for questions about ingested local documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The knowledge-base search query.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


class ResearchAgent:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": _clean_text(SYSTEM_PROMPT)}
        ]
        self.max_iterations = settings.max_iterations
        self.model_name = settings.model_name

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": _clean_text(user_input)})

        for _ in range(self.max_iterations):
            response = client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )

            assistant_message = response.choices[0].message

            message_dict: dict[str, Any] = {
                "role": "assistant",
                "content": _clean_text(assistant_message.content),
            }

            if assistant_message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": _clean_text(tc.function.arguments),
                        },
                    }
                    for tc in assistant_message.tool_calls
                ]

            self.messages.append(message_dict)

            if not assistant_message.tool_calls:
                return _clean_text(assistant_message.content) or "No response generated."

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                raw_args = _clean_text(tool_call.function.arguments)

                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    error_text = f"Invalid JSON arguments for tool '{tool_name}': {raw_args}"
                    print(f"\n🔧 Tool call: {tool_name}(INVALID JSON)")
                    print(f"📎 Result: {error_text}")

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_text,
                        }
                    )
                    continue

                print(f"\n🔧 Tool call: {tool_name}({args})")

                tool_fn = TOOL_FUNCTIONS.get(tool_name)
                if tool_fn is None:
                    result = f"Unknown tool: {tool_name}"
                else:
                    try:
                        result = tool_fn(**args)
                    except Exception as e:
                        result = f"Tool execution error in '{tool_name}': {e}"

                result_text = _clean_text(self._stringify_tool_result(result))
                print(f"📎 Result: {result_text[:800]}")

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )

        return "Error: Maximum iterations reached before producing a final answer."

    @staticmethod
    def _stringify_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, indent=2)


agent = ResearchAgent()
