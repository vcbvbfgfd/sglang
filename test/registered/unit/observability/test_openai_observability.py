"""Unit tests for OpenAI observability helpers."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    _OTEL_SDK_AVAILABLE = False

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _load_openai_observability():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "python" / "sglang" / "srt" / "openai_observability.py"

    stub_modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
    }
    stub_modules["sglang"].__path__ = [str(repo_root / "python" / "sglang")]
    stub_modules["sglang.srt"].__path__ = [str(repo_root / "python" / "sglang" / "srt")]

    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)

    module_name = "_test_openai_observability"
    previous_test_module = sys.modules.get(module_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
        if previous_test_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_test_module

    return module


mod = _load_openai_observability()


class _Recorder:
    def __init__(self):
        self.values = []

    def record(self, value, attributes=None):
        self.values.append((value, attributes))


class _Counter:
    def __init__(self):
        self.values = []

    def add(self, value, attributes=None):
        self.values.append((value, attributes))


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def is_recording(self):
        return True

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True


class _FakeTracer:
    def __init__(self, span):
        self.span = span

    def start_span(self, *args, **kwargs):
        return self.span


class _Dumpable:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def model_dump(self):
        return self._kwargs


class TestOpenAIObservability(CustomTestCase):
    def test_trace_content_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(mod.should_send_prompts())
        with patch.dict(os.environ, {"SGLANG_OTEL_TRACE_CONTENT": "true"}):
            self.assertTrue(mod.should_send_prompts())

    def test_model_as_dict_tolerates_plain_object(self):
        plain = object()
        self.assertIs(mod.model_as_dict(plain), plain)

    def test_accumulate_stream_items_noops_when_complete_response_is_none(self):
        with (
            patch.object(mod, "is_otel_available", return_value=True),
            patch.object(
                mod,
                "model_as_dict",
                side_effect=AssertionError("should not convert discarded chunks"),
            ),
        ):
            mod.accumulate_stream_items({"model": "test-model"}, None)

    def test_accumulate_stream_items_skips_tool_calls_without_valid_index(self):
        complete_response = {"choices": [], "model": "", "usage": None, "error": None}
        chunk = {
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "missing_index"}},
                            {"index": None, "function": {"name": "none_index"}},
                            {
                                "index": "0",
                                "id": "call-0",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"city"',
                                },
                            },
                            {
                                "index": "0",
                                "function": {"arguments": ': "Paris"}'},
                            },
                        ],
                    },
                }
            ],
        }

        with (
            patch.object(mod, "is_otel_available", return_value=True),
            patch.object(mod, "should_send_prompts", return_value=True),
        ):
            mod.accumulate_stream_items(chunk, complete_response)

        tool_calls = complete_response["choices"][0]["message"]["tool_calls"]
        self.assertEqual(tool_calls[0]["id"], "call-0")
        self.assertEqual(tool_calls[0]["function"]["name"], "lookup")
        self.assertEqual(tool_calls[0]["function"]["arguments"], '{"city": "Paris"}')

    def test_accumulate_stream_items_converts_nested_choice_objects(self):
        complete_response = {"choices": [], "model": "", "usage": None, "error": None}
        chunk = {
            "model": "test-model",
            "choices": [
                _Dumpable(
                    index=0,
                    delta=_Dumpable(role="assistant", content="hello"),
                    finish_reason=None,
                )
            ],
        }

        with (
            patch.object(mod, "is_otel_available", return_value=True),
            patch.object(mod, "should_send_prompts", return_value=True),
        ):
            mod.accumulate_stream_items(chunk, complete_response)

        self.assertEqual(
            complete_response["choices"][0]["message"]["role"], "assistant"
        )
        self.assertEqual(complete_response["choices"][0]["message"]["content"], "hello")

    def test_accumulate_stream_items_does_not_retain_content_by_default(self):
        complete_response = {"choices": [], "model": "", "usage": None, "error": None}
        chunk = {
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta": {"role": "assistant", "content": "sensitive"},
                }
            ],
        }

        with (
            patch.object(mod, "is_otel_available", return_value=True),
            patch.object(mod, "should_send_prompts", return_value=False),
        ):
            mod.accumulate_stream_items(chunk, complete_response)

        self.assertEqual(complete_response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(complete_response["choices"][0]["message"]["content"], "")

    @unittest.skipUnless(mod._is_msgspec_available, "msgspec not installed")
    def test_accumulate_stream_items_converts_msgspec_nested_choice(self):
        class Delta(mod.msgspec.Struct, omit_defaults=True):
            reasoning_content: str | None = None
            role: str | None = None
            content: str | None = None

        class Choice(mod.msgspec.Struct):
            index: int
            delta: Delta
            logprobs: dict | None = None
            finish_reason: str | None = None

        class Chunk(mod.msgspec.Struct, omit_defaults=True):
            model: str
            choices: list[Choice]
            usage: dict | None = None

        complete_response = {"choices": [], "model": "", "usage": None, "error": None}
        chunk = Chunk(
            "test-model", [Choice(0, Delta(role="assistant", content="hello"))]
        )

        with (
            patch.object(mod, "is_otel_available", return_value=True),
            patch.object(mod, "should_send_prompts", return_value=True),
        ):
            mod.accumulate_stream_items(chunk, complete_response)

        self.assertEqual(
            complete_response["choices"][0]["message"]["role"], "assistant"
        )
        self.assertEqual(complete_response["choices"][0]["message"]["content"], "hello")

    def test_provider_lazily_disables_request_spans_when_initialization_fails(self):
        original = mod._is_otel_imported
        try:
            mod._is_otel_imported = True
            provider = mod.OpenTelemetryProvider()
            self.assertIsNone(provider.tracer)
            with (
                patch.object(mod, "init_tracer", side_effect=RuntimeError("boom")),
                patch.object(mod, "init_metrics") as init_metrics,
                self.assertLogs(mod.logger, level="WARNING"),
            ):
                initialized = provider._ensure_initialized()
            self.assertFalse(initialized)
            self.assertIsNone(provider.tracer)
            self.assertIsNone(provider.meter)
            self.assertTrue(mod._is_otel_imported)
            init_metrics.assert_not_called()
        finally:
            mod._is_otel_imported = original

    def test_collect_request_metrics_avoids_duplicate_prompt_and_cache_tokens(self):
        request_metrics = mod.collect_request_metrics(
            [
                {
                    "id": "req-1",
                    "weight_version": "v2",
                    "dp_rank": 3,
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                    "reasoning_tokens": 2,
                    "cached_tokens": 60,
                    "cached_tokens_details": {
                        "device": 30,
                        "host": 20,
                        "storage": 10,
                        "storage_backend": "mooncake",
                    },
                    "queue_time": 0.25,
                    "forward_entry_time": 10.0,
                    "prefill_finished_time": 10.5,
                    "request_received_ts": 9.0,
                    "request_finished_ts": 12.0,
                    "num_retractions": 1,
                    "spec_accepted_drafts": 3,
                    "spec_proposed_drafts": 4,
                    "spec_verify_ct": 2,
                    "finish_reason": {"type": "stop"},
                },
                {
                    "id": "req-1",
                    "prompt_tokens": 100,
                    "completion_tokens": 6,
                    "reasoning_tokens": 1,
                    "cached_tokens": 60,
                    "finish_reason": {"type": "length"},
                },
            ]
        )

        self.assertEqual(request_metrics.prompt_tokens, 100)
        self.assertEqual(request_metrics.cached_tokens, 60)
        self.assertEqual(request_metrics.completion_tokens, 10)
        self.assertEqual(request_metrics.completion_token_intervals, 8)
        self.assertEqual(request_metrics.num_sequences, 2)
        self.assertEqual(request_metrics.reasoning_tokens, 3)
        self.assertEqual(request_metrics.cache_miss_tokens, 40)
        self.assertAlmostEqual(request_metrics.cache_hit_ratio, 0.6)
        self.assertEqual(request_metrics.cache_device_tokens, 30)
        self.assertEqual(request_metrics.cache_host_tokens, 20)
        self.assertEqual(request_metrics.cache_storage_tokens, 10)
        self.assertEqual(request_metrics.cache_storage_backend, "mooncake")
        self.assertAlmostEqual(request_metrics.scheduler_prefill_time, 0.5)
        self.assertAlmostEqual(request_metrics.backend_e2e_time, 3.0)
        self.assertAlmostEqual(request_metrics.spec_accept_ratio, 0.75)
        self.assertEqual(request_metrics.finish_reasons, ["stop", "length"])

    def test_record_adds_cache_attributes_and_uses_inter_token_denominator(self):
        span = _FakeSpan()
        provider = mod.OpenTelemetryProvider.__new__(mod.OpenTelemetryProvider)
        provider.tracer = _FakeTracer(span)
        provider.meter = None

        request = SimpleNamespace(
            model="test-model",
            stream=True,
            messages=[],
            max_tokens=None,
            max_completion_tokens=None,
            temperature=None,
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None,
            user=None,
        )
        request_metrics = mod.RequestMetrics(
            prompt_tokens=20,
            completion_tokens=4,
            reasoning_tokens=1,
            cached_tokens=10,
            cache_device_tokens=6,
            cache_host_tokens=4,
        )

        original_metrics_state = mod.Meters.is_metrics_inited
        try:
            mod.Meters.is_metrics_inited = False
            with (
                patch.object(mod, "is_otel_available", return_value=True),
                patch.object(mod, "extract_trace_context", return_value=None),
                patch.object(mod, "should_send_prompts", return_value=False),
                patch.object(mod, "SpanKind", SimpleNamespace(SERVER="server")),
                patch.object(mod, "StatusCode", SimpleNamespace(OK="ok")),
                patch.object(
                    mod, "Status", lambda status_code: ("status", status_code)
                ),
                patch.object(mod.time, "time", return_value=10.0),
            ):
                provider.record(
                    "sglang_chat_completion",
                    {},
                    request,
                    {"choices": [], "model": "test-model"},
                    {},
                    start_time=1.0,
                    time_of_first_token=4.0,
                    stream=True,
                    request_metrics=request_metrics,
                )

            self.assertEqual(
                span.attributes[mod.SpanAttributes.SGLANG_KV_CACHE_HIT_TOKENS], 10
            )
            self.assertEqual(
                span.attributes[mod.SpanAttributes.SGLANG_KV_CACHE_MISS_TOKENS], 10
            )
            self.assertAlmostEqual(
                span.attributes[mod.SpanAttributes.SGLANG_KV_CACHE_HIT_RATIO], 0.5
            )
            self.assertAlmostEqual(
                span.attributes[
                    mod.SpanAttributes.GEN_AI_STREAMING_TIME_PER_OUTPUT_TOKEN
                ],
                2.0,
            )
            self.assertTrue(span.ended)
        finally:
            mod.Meters.is_metrics_inited = original_metrics_state

    def test_record_request_metrics_exports_weighted_cache_token_counters(self):
        meter_names = (
            "tokens_histogram",
            "kv_cache_lookup_tokens",
            "kv_cache_hit_tokens",
            "kv_cache_miss_tokens",
            "kv_cache_request_hit_ratio",
            "request_queue_duration",
            "request_prefill_duration",
            "request_backend_duration",
            "request_retractions",
            "spec_accept_ratio",
        )
        original_metrics_state = mod.Meters.is_metrics_inited
        original_meters = {name: getattr(mod.Meters, name) for name in meter_names}
        try:
            mod.Meters.is_metrics_inited = True
            for name in meter_names:
                setattr(
                    mod.Meters,
                    name,
                    (
                        _Counter()
                        if name.endswith("tokens") or name == "request_retractions"
                        else _Recorder()
                    ),
                )

            request_metrics = mod.RequestMetrics(
                prompt_tokens=100,
                cached_tokens=60,
                cache_device_tokens=30,
                cache_host_tokens=20,
                cache_storage_tokens=5,
                cache_storage_backend="mooncake",
            )
            mod.record_request_metrics(request_metrics, {"model": "test-model"})

            self.assertEqual(mod.Meters.kv_cache_lookup_tokens.values[0][0], 100)
            self.assertEqual(mod.Meters.kv_cache_miss_tokens.values[0][0], 40)
            hit_values = mod.Meters.kv_cache_hit_tokens.values
            self.assertEqual(sum(value for value, _ in hit_values), 60)
            self.assertEqual(
                [attributes["cache_source"] for _, attributes in hit_values],
                ["device", "host", "storage", "unknown"],
            )
        finally:
            mod.Meters.is_metrics_inited = original_metrics_state
            for name, value in original_meters.items():
                setattr(mod.Meters, name, value)

    @unittest.skipUnless(_OTEL_SDK_AVAILABLE, "OpenTelemetry SDK not installed")
    def test_real_otel_sdk_metric_instruments(self):
        instrument_names = (
            "chat_counter",
            "tokens_histogram",
            "chat_choice_counter",
            "chat_duration_histogram",
            "chat_exception_counter",
            "streaming_time_to_first_token",
            "streaming_time_to_generate",
            "streaming_time_per_output_token",
            "kv_cache_lookup_tokens",
            "kv_cache_hit_tokens",
            "kv_cache_miss_tokens",
            "kv_cache_request_hit_ratio",
            "request_queue_duration",
            "request_prefill_duration",
            "request_backend_duration",
            "request_retractions",
            "spec_accept_ratio",
            "spec_accepted_draft_tokens",
            "spec_proposed_draft_tokens",
            "spec_verify_calls",
        )
        original_metrics_state = mod.Meters.is_metrics_inited
        original_meters = {name: getattr(mod.Meters, name) for name in instrument_names}
        reader = InMemoryMetricReader()
        meter_provider = SDKMeterProvider(metric_readers=[reader])
        try:
            mod.Meters.is_metrics_inited = False
            mod.init_genai_metrics(meter_provider.get_meter("sglang.test"))
            mod.record_request_metrics(
                mod.RequestMetrics(
                    prompt_tokens=100,
                    cached_tokens=60,
                    cache_device_tokens=30,
                    cache_host_tokens=20,
                    cache_storage_tokens=10,
                    queue_time=0.1,
                    scheduler_prefill_time=0.2,
                    backend_e2e_time=0.5,
                    num_retractions=1,
                    spec_accepted_drafts=3,
                    spec_proposed_drafts=4,
                    spec_verify_count=2,
                ),
                {"model": "test-model"},
            )

            metrics_data = reader.get_metrics_data()
            exported_names = {
                metric.name
                for resource_metrics in metrics_data.resource_metrics
                for scope_metrics in resource_metrics.scope_metrics
                for metric in scope_metrics.metrics
            }
            self.assertIn(mod.Meters.KV_CACHE_LOOKUP_TOKENS, exported_names)
            self.assertIn(mod.Meters.KV_CACHE_HIT_TOKENS, exported_names)
            self.assertIn(mod.Meters.KV_CACHE_REQUEST_HIT_RATIO, exported_names)
            self.assertIn(mod.Meters.SPEC_ACCEPTED_DRAFT_TOKENS, exported_names)
            self.assertIn(mod.Meters.SPEC_PROPOSED_DRAFT_TOKENS, exported_names)
        finally:
            mod.Meters.is_metrics_inited = original_metrics_state
            for name, value in original_meters.items():
                setattr(mod.Meters, name, value)
            meter_provider.shutdown()

    @unittest.skipUnless(_OTEL_SDK_AVAILABLE, "OpenTelemetry SDK not installed")
    def test_real_otel_sdk_span_lifetime_and_context_injection(self):
        exporter = InMemorySpanExporter()
        tracer_provider = SDKTracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

        provider = mod.OpenTelemetryProvider()
        request = SimpleNamespace(
            model="test-model",
            stream=True,
            messages=[],
            max_tokens=16,
            max_completion_tokens=None,
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            user=None,
        )
        request_metrics = mod.RequestMetrics(
            request_id="req-real-sdk",
            prompt_tokens=20,
            completion_tokens=3,
            cached_tokens=5,
        )

        with (
            patch.object(mod, "get_tracer_provider", return_value=tracer_provider),
            patch.object(mod, "LoggingInstrumentor", None),
            patch.object(mod, "init_metrics", return_value=None),
            patch.object(mod, "should_send_prompts", return_value=False),
        ):
            observation = provider.start_request(
                "sglang_chat_completion", {}, request, start_time=1.0
            )
            self.assertIsNotNone(observation)
            self.assertIn("traceparent", observation.trace_headers)
            with patch.object(mod.time, "time", return_value=3.0):
                provider.record(
                    "sglang_chat_completion",
                    {},
                    request,
                    {"choices": [], "model": "test-model"},
                    {},
                    start_time=1.0,
                    time_of_first_token=2.0,
                    stream=True,
                    observation=observation,
                    request_metrics=request_metrics,
                )

        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].start_time, 1_000_000_000)
        self.assertEqual(
            spans[0].attributes[mod.SpanAttributes.GEN_AI_REQUEST_ID],
            "req-real-sdk",
        )
        self.assertEqual(
            spans[0].attributes[mod.SpanAttributes.SGLANG_KV_CACHE_HIT_TOKENS], 5
        )

    def test_record_stream_with_usage_and_no_first_token_time_does_not_crash(self):
        span = _FakeSpan()
        provider = mod.OpenTelemetryProvider.__new__(mod.OpenTelemetryProvider)
        provider.tracer = _FakeTracer(span)
        provider.meter = None

        original_metrics_state = mod.Meters.is_metrics_inited
        original_meters = {
            name: getattr(mod.Meters, name)
            for name in (
                "chat_counter",
                "tokens_histogram",
                "chat_choice_counter",
                "chat_duration_histogram",
                "streaming_time_to_first_token",
                "streaming_time_to_generate",
                "streaming_time_per_output_token",
            )
        }
        per_token_recorder = _Recorder()
        try:
            mod.Meters.is_metrics_inited = True
            mod.Meters.chat_counter = _Counter()
            mod.Meters.tokens_histogram = _Recorder()
            mod.Meters.chat_choice_counter = _Counter()
            mod.Meters.chat_duration_histogram = _Recorder()
            mod.Meters.streaming_time_to_first_token = _Recorder()
            mod.Meters.streaming_time_to_generate = _Recorder()
            mod.Meters.streaming_time_per_output_token = per_token_recorder

            request = SimpleNamespace(
                model="test-model",
                stream=True,
                messages=[],
                max_tokens=None,
                temperature=None,
                top_p=None,
                frequency_penalty=None,
                presence_penalty=None,
                user=None,
            )

            with (
                patch.object(mod, "is_otel_available", return_value=True),
                patch.object(mod, "extract_trace_context", return_value=None),
                patch.object(mod, "should_send_prompts", return_value=False),
                patch.object(mod, "SpanKind", SimpleNamespace(SERVER="server")),
                patch.object(mod, "StatusCode", SimpleNamespace(OK="ok")),
                patch.object(
                    mod, "Status", lambda status_code: ("status", status_code)
                ),
                patch.object(mod.time, "time", return_value=10.0),
            ):
                provider.record(
                    "sglang_chat_completion",
                    {},
                    request,
                    {"choices": [], "model": "test-model"},
                    {"completion_tokens": 2},
                    start_time=1.0,
                    time_of_first_token=None,
                    stream=True,
                )

            self.assertEqual(per_token_recorder.values, [])
            self.assertTrue(span.ended)
        finally:
            mod.Meters.is_metrics_inited = original_metrics_state
            for name, value in original_meters.items():
                setattr(mod.Meters, name, value)


if __name__ == "__main__":
    unittest.main()
