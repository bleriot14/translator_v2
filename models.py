from sqlmodel import SQLModel, Field
from typing import Optional

class Translation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    total_time: float
    input_prep_time: float
    translation_time: float
    decoding_time: float
    model_load_time: float = Field(default=0.0)

class TranslationRequest(SQLModel):
    text: str
    source_lang: str
    target_lang: str