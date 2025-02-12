from typing import Callable, Dict, Optional, List
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator,
)
from litellm import completion


@register_validator(name="toxic-words", data_type="string")
def toxic_words(value, metadata: Dict) -> ValidationResult:
    print(f"toxic_words: Input value: {value}")  # Debug print
    mentioned_words = []
    for word in ["butt", "poop", "booger"]:
        if word in value:
            mentioned_words.append(word)

    print(f"toxic_words: Mentioned words: {mentioned_words}")  # Debug print
    if len(mentioned_words) > 0:
        return FailResult(
            error_message=f"Mention toxic words: {', '.join(mentioned_words)}",
        )
    else:
        return PassResult()
    

@register_validator(name="toxic-words", data_type="string")
class ToxicWords(Validator):
    def __init__(self, search_words: List[str]=["booger", "butt"], on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, search_words=search_words)
        self.search_words = search_words

    def _validate(self, value: List[str], metadata: Dict) -> ValidationResult:
        print(f"ToxicWords: Input value: {value}")  # Debug print
        mentioned_words = []
        for word in self.search_words:
            if word in value:
                mentioned_words.append(word)

        print(f"ToxicWords: Mentioned words: {mentioned_words}")  # Debug print
        if len(mentioned_words) > 0:
            return FailResult(
                error_message=f"Mentioned toxic words: {', '.join(mentioned_words)}",
            )
        else:
            return PassResult()
        



PROMPT = """
You are a helpful assistant.
Analyze the other messages provided and respond with a number between 0 and 100 and nothing else.
Do not output prose.
0 means the messages contain no toxic language.
100 means the messages contain very toxic language that is offensive, vulgar, negative or insulting.
"""

@register_validator(name="toxic-language", data_type="string")
class ToxicLanguage(Validator):
    def __init__(self, threshold: int=70, model='ollama/llama3', on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold
        self._model = model

    def _llm_callable(self, messages):
        print(f"ToxicLanguage: Calling LLM with messages: {messages}")  # Debug print
        return completion(
            model=self._model,
            api_base="http://localhost:11434",
            messages=messages,
            temperature=0.0,
        )
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        print(f"ToxicLanguage: Input value: {value}")  # Debug print
        messages = [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": value,
            }
        ]

        score = int(self._llm_callable(messages).choices[0].message.content)
        print(f"ToxicLanguage: LLM score: {score}")  # Debug print
        if score > self._threshold:
            print(f"ToxicLanguage: Validation failed. Score {score} is above threshold {self._threshold}.")  # Debug print
            return FailResult(
                error_message=f"{value} failed validation. Score {score} is below threshold of {self._threshold}."
            )
        else:
            print(f"ToxicLanguage: Validation passed. Score {score} is below threshold {self._threshold}.")  # Debug print
            return PassResult()