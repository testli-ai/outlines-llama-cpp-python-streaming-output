from outlines import models, generate
from llama_cpp import Llama
import os
import json

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal


class Question(BaseModel):
    question_text: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_option: int = Field(..., ge=0, le=3)

    @field_validator("options")
    def check_options_length(cls, v):
        if len(v) != 4:
            raise ValueError("Each question must have exactly 4 options")
        return v

    @field_validator("correct_option")
    def check_correct_option(cls, v, values):
        options = values.data.get("options", [])
        if v not in range(len(options)):
            raise ValueError("correct_option must be an integer between 0 and 3")
        return v


class Section(BaseModel):
    section: Literal[1, 2, 3, 4, 5, 6]
    section_name: Literal[
        "Cloze Grammar Vocabulary",
        "Cloze Contextual Vocabulary",
        "Best Arrangement of Utterances",
        "Cloze Informational Comprehension",
        "Reading Comprehension",
        "Reading Comprehension Advanced",
    ]
    passage_text: str
    questions: List[Question]


class ExamSchema(BaseModel):
    sections: List[Section] = Field(..., min_length=6, max_length=6)


exam_schema_json = json.dumps(ExamSchema.model_json_schema(), indent=2)

llm = Llama(
    model_path=os.path.join(os.getcwd(), "src", "models", "unsloth.Q4_K_M.gguf"),
    n_threads=4,
)
"""
Uncomment the code portion below to test the code with another model
"""
# llm = Llama(
#     model_path=os.path.join(
#         os.getcwd(), "src", "models", "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
#     ),
#     n_threads=4,
# )
model = models.LlamaCpp(llm)


generator = generate.json(model, exam_schema_json)
exam_stream = generator.stream(
    "Instruct: You are a teacher creating an English exam for the Vietnamese National High School Graduation Examination for the English subject.\nThe exam level is medium.\nGenerate a new English exam.\nOutput:",
    max_tokens=None,
    stop_at=["Q:", "\n"],
)

for stream in exam_stream:
    print(stream)
