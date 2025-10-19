
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import MarianMTModel, MarianTokenizer

MARIAN_MODELS: Dict[str, str] = {
    "german": "Helsinki-NLP/opus-mt-en-de",
    "arabic": "Helsinki-NLP/opus-mt-en-ar",
    "darija": "Helsinki-NLP/opus-mt-en-ary",
    "spanish": "Helsinki-NLP/opus-mt-en-es",
    "french": "Helsinki-NLP/opus-mt-en-fr",
    "polish": "Helsinki-NLP/opus-mt-en-pl",
    "czech": "Helsinki-NLP/opus-mt-en-cs",
    "russian": "Helsinki-NLP/opus-mt-en-ru",
}

SUPPORTED_TRANSLATION_LANGUAGES = set(MARIAN_MODELS.keys()) | {"english"}


@dataclass
class SegmentPayload:
    start: float
    end: float
    text: str


@dataclass
class TranslatedTranscript:
    language: str
    segments: List[SegmentPayload]


def _batch(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


class MarianAdapter:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)
        self._device = torch.device("cpu")
        self._model.to(self._device)

    def translate_batch(self, sentences: Sequence[str]) -> List[str]:
        inputs = self._tokenizer(
            list(sentences), return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs)
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [text.strip() for text in decoded]


class MarianTranslationService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._adapters: Dict[str, MarianAdapter] = {}

    def supports(self, language: str) -> bool:
        return language.lower() in SUPPORTED_TRANSLATION_LANGUAGES

    def translate_text(self, text: str, target_language: str) -> str:
        payload = SegmentPayload(start=0.0, end=0.0, text=text)
        translated = self.translate_segments([payload], target_language)
        return " ".join(segment.text for segment in translated).strip()

    def translate_segments(
        self, segments: Sequence[SegmentPayload], target_language: str, batch_size: int = 8
    ) -> List[SegmentPayload]:
        target_language = target_language.lower()
        if target_language == "english":
            return [
                SegmentPayload(start=segment.start, end=segment.end, text=segment.text)
                for segment in segments
            ]
        if target_language not in MARIAN_MODELS:
            raise ValueError(f"No Marian model configured for '{target_language}'")

        adapter = self._get_adapter(target_language)
        texts = [segment.text.strip() for segment in segments]
        translated_texts: List[str] = []

        for batch in _batch(texts, batch_size):
            if not batch:
                continue
            translated_texts.extend(adapter.translate_batch(batch))

        output_segments: List[SegmentPayload] = []
        for original, translated in zip(segments, translated_texts):
            output_segments.append(
                SegmentPayload(start=original.start, end=original.end, text=translated)
            )

        return output_segments

    def _get_adapter(self, language: str) -> MarianAdapter:
        with self._lock:
            adapter = self._adapters.get(language)
            if adapter is None:
                adapter = MarianAdapter(MARIAN_MODELS[language])
                self._adapters[language] = adapter
        return adapter


def _refine_segments(segments: List[SegmentPayload]) -> List[SegmentPayload]:
    refined: List[SegmentPayload] = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            text = text[0].upper() + text[1:] if text[0].islower() else text
        refined.append(SegmentPayload(start=segment.start, end=segment.end, text=text))
    return refined


def translate_fanout(
    segments: Sequence[SegmentPayload],
    source_language: str,
    targets: Iterable[str],
    profile: str = "fast",
    service: Optional[MarianTranslationService] = None,
) -> List[TranslatedTranscript]:
    service = service or MarianTranslationService()
    base_segments = [
        SegmentPayload(start=segment.start, end=segment.end, text=segment.text)
        for segment in segments
    ]
    source_language = (source_language or "english").lower()
    results: List[TranslatedTranscript] = []

    for target in targets:
        target_language = target.lower()
        if target_language == source_language:
            results.append(
                TranslatedTranscript(
                    language=target_language,
                    segments=[
                        SegmentPayload(start=segment.start, end=segment.end, text=segment.text)
                        for segment in base_segments
                    ],
                )
            )
            continue

        if source_language != "english":
            raise ValueError(
                "translate_fanout currently supports fanout only from English source transcripts."
            )

        translated = service.translate_segments(base_segments, target_language)
        if profile == "pro":
            translated = _refine_segments(translated)
        results.append(TranslatedTranscript(language=target_language, segments=translated))

    return results


def supported_translation_languages() -> List[str]:
    return sorted(SUPPORTED_TRANSLATION_LANGUAGES)
