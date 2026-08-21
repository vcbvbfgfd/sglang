import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional

from sglang.srt.openai_observability_buckets import (
    _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
    _GEN_AI_SERVER_KV_CACHE_HIT_RATIO_BUCKETS,
    _GEN_AI_SERVER_TIME_PER_OUTPUT_TOKEN_BUCKETS,
    _GEN_AI_SERVER_TIME_TO_FIRST_TOKEN_BUCKETS,
)

TRACE_HEADERS = ["traceparent", "tracestate"]
LLM_USAGE_TOKEN_TYPES = ["prompt_tokens", "completion_tokens", "total_tokens"]
PROMPT_FILTER_KEY = "prompt_filter_results"
CONTENT_FILTER_KEY = "content_filter_results"

logger = logging.getLogger(__name__)

_is_otel_imported = False
otel_import_error_traceback: Optional[str] = None
tracer = None
meter = None
LoggingInstrumentor = None
SystemMetricsInstrumentor = None

try:
    import msgspec  # type: ignore

    _is_msgspec_available = True
except ImportError:
    msgspec = None  # type: ignore
    _is_msgspec_available = False
try:
    from opentelemetry.context import get_current
    from opentelemetry.context.context import Context
    from opentelemetry.metrics import (
        Meter,
        get_meter_provider,
        set_meter_provider,
    )
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_METRICS_PROTOCOL,
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import (
        INVALID_SPAN,
        SpanKind,
        Tracer,
        get_current_span,
        get_tracer_provider,
        set_span_in_context,
        set_tracer_provider,
    )
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
    from opentelemetry.trace.status import Status, StatusCode

    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    import traceback

    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass

    class Meter:  # type: ignore
        pass

    class Status:  # type: ignore
        pass

    class StatusCode:  # type: ignore
        pass

    class TraceContextTextMapPropagator:  # type: ignore
        pass


if _is_otel_imported:
    # Logging and host metrics are useful additions, but they must not disable
    # request traces when the optional instrumentation packages are absent.
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
    except ImportError:
        LoggingInstrumentor = None
    try:
        from opentelemetry.instrumentation.system_metrics import (
            SystemMetricsInstrumentor,
        )
    except ImportError:
        SystemMetricsInstrumentor = None


def is_otel_available() -> bool:
    return _is_otel_imported


def _is_proxy_provider(provider: Any) -> bool:
    """Return whether an OTel API provider is still the unconfigured proxy."""
    return provider.__class__.__name__ in {"ProxyTracerProvider", "_ProxyMeterProvider"}


def init_tracer(instrumenting_module_name: str) -> Optional[Tracer]:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )
    trace_provider = get_tracer_provider()
    if _is_proxy_provider(trace_provider):
        trace_provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: "sglang"})
        )
        span_exporter = get_span_exporter()
        trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        set_tracer_provider(trace_provider)

    if LoggingInstrumentor is not None:
        LoggingInstrumentor().instrument()

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter():
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter()


def init_metrics(instrumenting_module_name: str) -> Optional[Meter]:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a meter. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )
    metrics_provider = get_meter_provider()
    if _is_proxy_provider(metrics_provider):
        metric_exporter = get_metrics_exporter()
        export_interval_millis = (
            _safe_int(
                os.getenv(
                    "SGLANG_OTEL_METRICS_EXPORT_INTERVAL_MILLIS",
                    os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "30000"),
                )
            )
            or 30_000
        )
        reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=export_interval_millis
        )
        metrics_provider = MeterProvider(
            metric_readers=[reader],
            resource=Resource.create({SERVICE_NAME: "sglang"}),
        )
        set_meter_provider(metrics_provider)

    meter = metrics_provider.get_meter(instrumenting_module_name)

    init_genai_metrics(meter)
    if SystemMetricsInstrumentor is not None:
        SystemMetricsInstrumentor().instrument()
    return meter


def get_metrics_exporter():
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_METRICS_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPMetricExporter()


def extract_trace_context(headers: Optional[Mapping[str, str]]) -> Optional[Context]:
    if is_otel_available():
        current_span = get_current_span()
        get_span_context = getattr(current_span, "get_span_context", None)
        has_valid_current_span = (
            callable(get_span_context) and get_span_context().is_valid
        )
        if current_span is INVALID_SPAN or not has_valid_current_span:
            headers = headers or {}
            return TraceContextTextMapPropagator().extract(headers)
        return get_current()
    return None


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None


@dataclass
class RequestMetrics:
    """Low-cardinality per-request data exported to OTel spans and metrics."""

    request_id: Optional[str] = None
    weight_version: Optional[str] = None
    dp_rank: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    completion_token_intervals: Optional[int] = None
    num_sequences: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_device_tokens: int = 0
    cache_host_tokens: int = 0
    cache_storage_tokens: int = 0
    cache_storage_backend: Optional[str] = None
    queue_time: Optional[float] = None
    scheduler_prefill_time: Optional[float] = None
    backend_e2e_time: Optional[float] = None
    decode_throughput: Optional[float] = None
    num_retractions: int = 0
    spec_accepted_drafts: int = 0
    spec_proposed_drafts: int = 0
    spec_verify_count: int = 0
    finish_reasons: list[str] = field(default_factory=list)

    @property
    def cache_hit_tokens(self) -> int:
        return min(self.cached_tokens, self.prompt_tokens)

    @property
    def inter_token_intervals(self) -> int:
        if self.completion_token_intervals is not None:
            return self.completion_token_intervals
        return max(self.completion_tokens - 1, 0)

    @property
    def cache_miss_tokens(self) -> int:
        return max(self.prompt_tokens - self.cache_hit_tokens, 0)

    @property
    def cache_hit_ratio(self) -> Optional[float]:
        if self.prompt_tokens <= 0:
            return None
        return self.cache_hit_tokens / self.prompt_tokens

    @property
    def spec_accept_ratio(self) -> Optional[float]:
        if self.spec_proposed_drafts <= 0:
            return None
        return min(self.spec_accepted_drafts / self.spec_proposed_drafts, 1.0)


def collect_request_metrics(meta_infos: Iterable[Mapping[str, Any]]) -> RequestMetrics:
    """Aggregate the latest per-choice meta_info without double-counting prompts."""
    infos = []
    for meta_info in meta_infos:
        converted = model_as_dict(meta_info)
        if isinstance(converted, Mapping):
            infos.append(converted)
    if not infos:
        return RequestMetrics()

    first = infos[0]

    def first_value(key: str):
        return next(
            (info.get(key) for info in infos if info.get(key) is not None), None
        )

    prompt_tokens = max(_safe_int(info.get("prompt_tokens")) for info in infos)
    cached_tokens = max(_safe_int(info.get("cached_tokens")) for info in infos)
    completion_tokens = sum(_safe_int(info.get("completion_tokens")) for info in infos)
    reasoning_tokens = sum(_safe_int(info.get("reasoning_tokens")) for info in infos)

    cache_details = first_value("cached_tokens_details")
    cache_details = model_as_dict(cache_details) if cache_details is not None else {}
    if not isinstance(cache_details, Mapping):
        cache_details = {}

    finish_reasons = []
    for info in infos:
        finish_reason = info.get("finish_reason")
        if isinstance(finish_reason, Mapping):
            finish_reason = finish_reason.get("type")
        if finish_reason and str(finish_reason) not in finish_reasons:
            finish_reasons.append(str(finish_reason))

    forward_entry_time = _safe_float(first_value("forward_entry_time"))
    prefill_finished_time = _safe_float(first_value("prefill_finished_time"))
    request_received_ts = _safe_float(first_value("request_received_ts"))
    request_finished_ts = _safe_float(first_value("request_finished_ts"))

    return RequestMetrics(
        request_id=str(first.get("id")) if first.get("id") else None,
        weight_version=(
            str(first_value("weight_version"))
            if first_value("weight_version") is not None
            else None
        ),
        dp_rank=(
            _safe_int(first_value("dp_rank"))
            if first_value("dp_rank") is not None
            else None
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_token_intervals=sum(
            max(_safe_int(info.get("completion_tokens")) - 1, 0) for info in infos
        ),
        num_sequences=len(infos),
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        cache_device_tokens=_safe_int(cache_details.get("device")),
        cache_host_tokens=_safe_int(cache_details.get("host")),
        cache_storage_tokens=_safe_int(cache_details.get("storage")),
        cache_storage_backend=(
            str(cache_details.get("storage_backend"))
            if cache_details.get("storage_backend")
            else None
        ),
        queue_time=_safe_float(first_value("queue_time")),
        scheduler_prefill_time=(
            prefill_finished_time - forward_entry_time
            if forward_entry_time is not None
            and prefill_finished_time is not None
            and prefill_finished_time >= forward_entry_time
            else None
        ),
        backend_e2e_time=(
            request_finished_ts - request_received_ts
            if request_received_ts is not None
            and request_finished_ts is not None
            and request_finished_ts >= request_received_ts
            else None
        ),
        decode_throughput=_safe_float(first_value("decode_throughput")),
        num_retractions=max(_safe_int(info.get("num_retractions")) for info in infos),
        spec_accepted_drafts=sum(
            _safe_int(info.get("spec_accepted_drafts")) for info in infos
        ),
        spec_proposed_drafts=sum(
            _safe_int(info.get("spec_proposed_drafts")) for info in infos
        ),
        spec_verify_count=sum(_safe_int(info.get("spec_verify_ct")) for info in infos),
        finish_reasons=finish_reasons,
    )


class Meters:
    LLM_GENERATION_CHOICES = "gen_ai.client.generation.choices"
    LLM_TOKEN_USAGE = "gen_ai.client.token.usage"
    LLM_OPERATION_DURATION = "gen_ai.client.operation.duration"
    LLM_COMPLETIONS_EXCEPTIONS = "gen_ai.chat_completions.exceptions"
    LLM_STREAMING_TIME_TO_FIRST_TOKEN = (
        "gen_ai.chat_completions.streaming_time_to_first_token"
    )
    LLM_STREAMING_TIME_TO_GENERATE = (
        "gen_ai.chat_completions.streaming_time_to_generate"
    )
    LLM_STREAMING_TIME_PER_OUTPUT_TOKEN = (
        "gen_ai.chat_completions.streaming_time_per_output_token"
    )
    LLM_CHAT_COUNT = "gen_ai.chat.count"
    KV_CACHE_LOOKUP_TOKENS = "sglang.kv_cache.lookup_tokens"
    KV_CACHE_HIT_TOKENS = "sglang.kv_cache.hit_tokens"
    KV_CACHE_MISS_TOKENS = "sglang.kv_cache.miss_tokens"
    KV_CACHE_REQUEST_HIT_RATIO = "sglang.kv_cache.request_hit_ratio"
    REQUEST_QUEUE_DURATION = "sglang.request.queue_duration"
    REQUEST_PREFILL_DURATION = "sglang.request.prefill_duration"
    REQUEST_BACKEND_DURATION = "sglang.request.backend_duration"
    REQUEST_RETRACTIONS = "sglang.request.retractions"
    SPEC_ACCEPT_RATIO = "sglang.speculative.accept_ratio"
    SPEC_ACCEPTED_DRAFT_TOKENS = "sglang.speculative.accepted_draft_tokens"
    SPEC_PROPOSED_DRAFT_TOKENS = "sglang.speculative.proposed_draft_tokens"
    SPEC_VERIFY_CALLS = "sglang.speculative.verify_calls"

    LLM_EMBEDDINGS_EXCEPTIONS = "gen_ai.embeddings.exceptions"
    LLM_EMBEDDINGS_VECTOR_SIZE = "gen_ai.embeddings.vector_size"
    LLM_IMAGE_GENERATIONS_EXCEPTIONS = "gen_ai.image_generations.exceptions"
    LLM_ANTHROPIC_COMPLETION_EXCEPTIONS = "gen_ai.anthropic.completion.exceptions"

    PINECONE_DB_QUERY_DURATION = "db.pinecone.query.duration"
    PINECONE_DB_QUERY_SCORES = "db.pinecone.query.scores"
    PINECONE_DB_USAGE_READ_UNITS = "db.pinecone.usage.read_units"
    PINECONE_DB_USAGE_WRITE_UNITS = "db.pinecone.usage_write_units"

    LLM_WATSONX_COMPLETIONS_DURATION = "llm.watsonx.completions.duration"
    LLM_WATSONX_COMPLETIONS_EXCEPTIONS = "llm.watsonx.completions.exceptions"
    LLM_WATSONX_COMPLETIONS_RESPONSES = "llm.watsonx.completions.responses"
    LLM_WATSONX_COMPLETIONS_TOKENS = "llm.watsonx.completions.tokens"

    is_metrics_inited = False

    chat_counter = None
    tokens_histogram = None
    chat_choice_counter = None
    chat_duration_histogram = None
    chat_exception_counter = None
    streaming_time_to_first_token = None
    streaming_time_to_generate = None
    streaming_time_per_output_token = None
    kv_cache_lookup_tokens = None
    kv_cache_hit_tokens = None
    kv_cache_miss_tokens = None
    kv_cache_request_hit_ratio = None
    request_queue_duration = None
    request_prefill_duration = None
    request_backend_duration = None
    request_retractions = None
    spec_accept_ratio = None
    spec_accepted_draft_tokens = None
    spec_proposed_draft_tokens = None
    spec_verify_calls = None


def init_genai_metrics(meter: Meter) -> None:
    if Meters.is_metrics_inited:
        return
    try:
        Meters.chat_counter = meter.create_counter(
            name=Meters.LLM_CHAT_COUNT,
            unit="{request}",
            description="Number of chat completion requests",
        )
        Meters.tokens_histogram = meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            unit="token",
            description="Measures number of input and output tokens used",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
        )
        # Meters.chat_token_recoder = meter.create_observable_counter()
        Meters.chat_choice_counter = meter.create_counter(
            name=Meters.LLM_GENERATION_CHOICES,
            unit="{choice}",
            description="Number of choices returned by chat completion requests",
        )

        Meters.chat_duration_histogram = meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )

        Meters.chat_exception_counter = meter.create_counter(
            name=Meters.LLM_COMPLETIONS_EXCEPTIONS,
            unit="{error}",
            description="Number of exceptions during chat completion requests",
        )

        Meters.streaming_time_to_first_token = meter.create_histogram(
            name=Meters.LLM_STREAMING_TIME_TO_FIRST_TOKEN,
            unit="s",
            description="Time to first token in streaming chat completions",
            explicit_bucket_boundaries_advisory=_GEN_AI_SERVER_TIME_TO_FIRST_TOKEN_BUCKETS,
        )
        Meters.streaming_time_to_generate = meter.create_histogram(
            name=Meters.LLM_STREAMING_TIME_TO_GENERATE,
            unit="s",
            description="Time between first token and completion in streaming chat completions",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        Meters.streaming_time_per_output_token = meter.create_histogram(
            name=Meters.LLM_STREAMING_TIME_PER_OUTPUT_TOKEN,
            unit="s",
            description="Time per output token in streaming chat completions",
            explicit_bucket_boundaries_advisory=_GEN_AI_SERVER_TIME_PER_OUTPUT_TOKEN_BUCKETS,
        )
        Meters.kv_cache_lookup_tokens = meter.create_counter(
            name=Meters.KV_CACHE_LOOKUP_TOKENS,
            unit="token",
            description="Number of prompt tokens looked up in the KV prefix cache",
        )
        Meters.kv_cache_hit_tokens = meter.create_counter(
            name=Meters.KV_CACHE_HIT_TOKENS,
            unit="token",
            description="Number of prompt tokens served from KV cache",
        )
        Meters.kv_cache_miss_tokens = meter.create_counter(
            name=Meters.KV_CACHE_MISS_TOKENS,
            unit="token",
            description="Number of prompt tokens that required prefill compute",
        )
        Meters.kv_cache_request_hit_ratio = meter.create_histogram(
            name=Meters.KV_CACHE_REQUEST_HIT_RATIO,
            unit="1",
            description="Distribution of request-level KV cache hit ratios",
            explicit_bucket_boundaries_advisory=(
                _GEN_AI_SERVER_KV_CACHE_HIT_RATIO_BUCKETS
            ),
        )
        Meters.request_queue_duration = meter.create_histogram(
            name=Meters.REQUEST_QUEUE_DURATION,
            unit="s",
            description="Time a request spent waiting in the scheduler queue",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        Meters.request_prefill_duration = meter.create_histogram(
            name=Meters.REQUEST_PREFILL_DURATION,
            unit="s",
            description="Scheduler prefill duration for a request",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        Meters.request_backend_duration = meter.create_histogram(
            name=Meters.REQUEST_BACKEND_DURATION,
            unit="s",
            description="Backend end-to-end request duration",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        Meters.request_retractions = meter.create_counter(
            name=Meters.REQUEST_RETRACTIONS,
            unit="{retraction}",
            description="Number of scheduler request retractions",
        )
        Meters.spec_accept_ratio = meter.create_histogram(
            name=Meters.SPEC_ACCEPT_RATIO,
            unit="1",
            description="Distribution of speculative decoding acceptance ratios",
            explicit_bucket_boundaries_advisory=(
                _GEN_AI_SERVER_KV_CACHE_HIT_RATIO_BUCKETS
            ),
        )
        Meters.spec_accepted_draft_tokens = meter.create_counter(
            name=Meters.SPEC_ACCEPTED_DRAFT_TOKENS,
            unit="token",
            description="Number of accepted speculative draft tokens",
        )
        Meters.spec_proposed_draft_tokens = meter.create_counter(
            name=Meters.SPEC_PROPOSED_DRAFT_TOKENS,
            unit="token",
            description="Number of proposed speculative draft tokens",
        )
        Meters.spec_verify_calls = meter.create_counter(
            name=Meters.SPEC_VERIFY_CALLS,
            unit="{call}",
            description="Number of speculative verification calls",
        )
        Meters.is_metrics_inited = True
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to init genai metrics, error: %s", str(ex))


def set_choice_counter_metrics(choices, shared_attributes):
    if Meters.is_metrics_inited:
        for choice in choices:
            choice = model_as_dict(choice)
            if not isinstance(choice, Mapping):
                continue
            if choice.get("finish_reason"):
                attributes_with_reason = {
                    **shared_attributes,
                    SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON: choice.get(
                        "finish_reason"
                    ),
                }
            else:
                attributes_with_reason = shared_attributes
            Meters.chat_choice_counter.add(1, attributes=attributes_with_reason)


def set_token_counter_metrics(usage, shared_attributes):
    if Meters.is_metrics_inited:
        usage = model_as_dict(usage)
        if not isinstance(usage, Mapping):
            return
        for name, val in usage.items():
            if name in LLM_USAGE_TOKEN_TYPES:
                attributes_with_token_type = {
                    **shared_attributes,
                    SpanAttributes.GEN_AI_TOKEN_TYPE: _token_type(name),
                }
                Meters.tokens_histogram.record(
                    val, attributes=attributes_with_token_type
                )


def record_request_metrics(
    request_metrics: RequestMetrics, shared_attributes: Dict[str, Any]
) -> None:
    if not Meters.is_metrics_inited:
        return

    if request_metrics.reasoning_tokens:
        Meters.tokens_histogram.record(
            request_metrics.reasoning_tokens,
            attributes={
                **shared_attributes,
                SpanAttributes.GEN_AI_TOKEN_TYPE: "reasoning",
            },
        )

    if request_metrics.prompt_tokens:
        Meters.kv_cache_lookup_tokens.add(
            request_metrics.prompt_tokens, attributes=shared_attributes
        )
        Meters.kv_cache_miss_tokens.add(
            request_metrics.cache_miss_tokens, attributes=shared_attributes
        )

    cache_sources = {
        "device": request_metrics.cache_device_tokens,
        "host": request_metrics.cache_host_tokens,
        "storage": request_metrics.cache_storage_tokens,
    }
    remaining_hit_tokens = request_metrics.cache_hit_tokens
    for source, raw_value in cache_sources.items():
        value = min(raw_value, remaining_hit_tokens)
        if not value:
            continue
        source_attributes = {**shared_attributes, "cache_source": source}
        if source == "storage" and request_metrics.cache_storage_backend:
            source_attributes["storage_backend"] = request_metrics.cache_storage_backend
        Meters.kv_cache_hit_tokens.add(value, attributes=source_attributes)
        remaining_hit_tokens -= value
    if remaining_hit_tokens:
        Meters.kv_cache_hit_tokens.add(
            remaining_hit_tokens,
            attributes={**shared_attributes, "cache_source": "unknown"},
        )

    if request_metrics.cache_hit_ratio is not None:
        Meters.kv_cache_request_hit_ratio.record(
            request_metrics.cache_hit_ratio, attributes=shared_attributes
        )
    if request_metrics.queue_time is not None:
        Meters.request_queue_duration.record(
            request_metrics.queue_time, attributes=shared_attributes
        )
    if request_metrics.scheduler_prefill_time is not None:
        Meters.request_prefill_duration.record(
            request_metrics.scheduler_prefill_time, attributes=shared_attributes
        )
    if request_metrics.backend_e2e_time is not None:
        Meters.request_backend_duration.record(
            request_metrics.backend_e2e_time, attributes=shared_attributes
        )
    if request_metrics.num_retractions:
        Meters.request_retractions.add(
            request_metrics.num_retractions, attributes=shared_attributes
        )
    if request_metrics.spec_accept_ratio is not None:
        Meters.spec_accept_ratio.record(
            request_metrics.spec_accept_ratio, attributes=shared_attributes
        )
    if request_metrics.spec_accepted_drafts:
        Meters.spec_accepted_draft_tokens.add(
            request_metrics.spec_accepted_drafts, attributes=shared_attributes
        )
    if request_metrics.spec_proposed_drafts:
        Meters.spec_proposed_draft_tokens.add(
            request_metrics.spec_proposed_drafts, attributes=shared_attributes
        )
    if request_metrics.spec_verify_count:
        Meters.spec_verify_calls.add(
            request_metrics.spec_verify_count, attributes=shared_attributes
        )


def metric_shared_attributes(
    response_model: str,
    operation: str,
    is_streaming: bool = False,
    request_metrics: Optional[RequestMetrics] = None,
):
    attributes = {
        SpanAttributes.GEN_AI_SYSTEM: "sglang",
        SpanAttributes.GEN_AI_RESPONSE_MODEL: response_model,
        "gen_ai_operation_name": operation,
        "stream": is_streaming,
    }
    if request_metrics is not None:
        if request_metrics.weight_version:
            attributes[SpanAttributes.SGLANG_WEIGHT_VERSION] = (
                request_metrics.weight_version
            )
        if request_metrics.dp_rank is not None:
            attributes[SpanAttributes.SGLANG_DP_RANK] = request_metrics.dp_rank
    return attributes


def _token_type(token_type: str):
    if token_type == "prompt_tokens":
        return "input"
    elif token_type == "completion_tokens":
        return "output"
    elif token_type == "total_tokens":
        return "total"

    return None


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"
    GEN_AI_USAGE_CACHED_TOKENS = "gen_ai.usage.cached_tokens"
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_PROMPTS = "gen_ai.prompt"
    GEN_AI_COMPLETIONS = "gen_ai.completion"
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
        "gen_ai.usage.cache_creation_input_tokens"
    )
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read_input_tokens"
    GEN_AI_TOKEN_TYPE = "gen_ai.token.type"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_USER = "gen_ai.user"
    GEN_AI_HEADERS = "gen_ai.headers"
    GEN_AI_TOP_K = "gen_ai.top_k"
    GEN_AI_IS_STREAMING = "gen_ai.is_streaming"
    GEN_AI_FREQUENCY_PENALTY = "gen_ai.frequency_penalty"
    GEN_AI_PRESENCE_PENALTY = "gen_ai.presence_penalty"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_REQUEST_PRIORITY = "gen_ai.request.priority"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_REQUEST_TYPE = "gen_ai.request.type"
    # TTFT TPOP span
    GEN_AI_STREAMING_TIME_TO_FIRST_TOKEN = (
        "gen_ai.chat_completions.streaming_time_to_first_token"
    )
    GEN_AI_STREAMING_TIME_PER_OUTPUT_TOKEN = (
        "gen_ai.chat_completions.streaming_time_per_output_token"
    )
    GEN_AI_STREAMING_TIME_TO_GENERATE = (
        "gen_ai.chat_completions.streaming_time_to_generate"
    )
    SGLANG_KV_CACHE_HIT_TOKENS = "sglang.kv_cache.hit_tokens"
    SGLANG_KV_CACHE_MISS_TOKENS = "sglang.kv_cache.miss_tokens"
    SGLANG_KV_CACHE_HIT_RATIO = "sglang.kv_cache.hit_ratio"
    SGLANG_KV_CACHE_DEVICE_HIT_TOKENS = "sglang.kv_cache.device_hit_tokens"
    SGLANG_KV_CACHE_HOST_HIT_TOKENS = "sglang.kv_cache.host_hit_tokens"
    SGLANG_KV_CACHE_STORAGE_HIT_TOKENS = "sglang.kv_cache.storage_hit_tokens"
    SGLANG_KV_CACHE_STORAGE_BACKEND = "sglang.kv_cache.storage_backend"
    SGLANG_SCHEDULER_PREFILL_DURATION = "sglang.latency.scheduler_prefill"
    SGLANG_BACKEND_E2E_DURATION = "sglang.latency.backend_e2e"
    SGLANG_DECODE_THROUGHPUT = "sglang.decode.throughput"
    SGLANG_REQUEST_RETRACTIONS = "sglang.request.retractions"
    SGLANG_SPEC_ACCEPTED_DRAFTS = "sglang.speculative.accepted_drafts"
    SGLANG_SPEC_PROPOSED_DRAFTS = "sglang.speculative.proposed_drafts"
    SGLANG_SPEC_ACCEPT_RATIO = "sglang.speculative.accept_ratio"
    SGLANG_SPEC_VERIFY_COUNT = "sglang.speculative.verify_count"
    SGLANG_WEIGHT_VERSION = "sglang.model.weight_version"
    SGLANG_DP_RANK = "sglang.dp_rank"


class LLMRequestTypeValues(Enum):
    COMPLETION = "completion"
    CHAT = "chat"
    RERANK = "rerank"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")


def set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return

    try:
        for i, msg in enumerate(messages):
            prefix = f"{SpanAttributes.GEN_AI_PROMPTS}.{i}"
            content = ""
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                content = json.dumps(msg.content)

            _set_span_attribute(span, f"{prefix}.role", msg.role)
            _set_span_attribute(span, f"{prefix}.content", content)
            # TODO: set tool call attributes

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set prompts for openai span, error: %s", str(ex))


def set_request_attributes(span, raw_request):
    if not span.is_recording():
        return

    try:
        # _set_api_attributes(span)
        _set_span_attribute(span, SpanAttributes.GEN_AI_SYSTEM, "sglang")
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_REQUEST_MODEL,
            getattr(raw_request, "model", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS,
            getattr(raw_request, "max_completion_tokens", None)
            or getattr(raw_request, "max_tokens", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_REQUEST_TEMPERATURE,
            getattr(raw_request, "temperature", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_REQUEST_TOP_P,
            getattr(raw_request, "top_p", None),
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_TOP_K, getattr(raw_request, "top_k", None)
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_REQUEST_N, getattr(raw_request, "n", None)
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_REQUEST_PRIORITY,
            getattr(raw_request, "priority", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_FREQUENCY_PENALTY,
            getattr(raw_request, "frequency_penalty", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_PRESENCE_PENALTY,
            getattr(raw_request, "presence_penalty", None),
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_USER, getattr(raw_request, "user", None)
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_IS_STREAMING,
            getattr(raw_request, "stream", False) or False,
        )
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set input attributes for request span, error: %s", str(ex)
        )


def set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        choice = model_as_dict(choice)
        if not isinstance(choice, Mapping):
            continue
        index = choice.get("index")
        prefix = f"{SpanAttributes.GEN_AI_COMPLETIONS}.{index}"
        _set_span_attribute(
            span, f"{prefix}.finish_reason", choice.get("finish_reason")
        )

        if choice.get("content_filter_results"):
            _set_span_attribute(
                span,
                f"{prefix}.{CONTENT_FILTER_KEY}",
                json.dumps(choice.get("content_filter_results")),
            )

        if choice.get("finish_reason") == "content_filter":
            _set_span_attribute(span, f"{prefix}.role", "assistant")
            _set_span_attribute(span, f"{prefix}.content", "FILTERED")
            return

        message = choice.get("message")
        if not message:
            return

        _set_span_attribute(span, f"{prefix}.role", message.get("role"))
        if message.get("refusal"):
            _set_span_attribute(span, f"{prefix}.refusal", message.get("refusal"))
        else:
            _set_span_attribute(span, f"{prefix}.content", message.get("content"))
            _set_span_attribute(
                span, f"{prefix}.reasoning_content", message.get("reasoning_content")
            )

        function_call = message.get("function_call")
        if function_call:
            _set_span_attribute(
                span, f"{prefix}.tool_calls.0.name", function_call.get("name")
            )
            _set_span_attribute(
                span,
                f"{prefix}.tool_calls.0.arguments",
                function_call.get("arguments"),
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                function = tool_call.get("function")
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.id",
                    tool_call.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.name",
                    function.get("name"),
                )
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.arguments",
                    function.get("arguments"),
                )


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def set_response_attributes(span, response, usage):
    if not span.is_recording():
        return

    try:
        response = model_as_dict(response)
        if not isinstance(response, Mapping):
            return

        _set_span_attribute(
            span, SpanAttributes.GEN_AI_RESPONSE_MODEL, response.get("model")
        )

        if not usage:
            return

        usage = model_as_dict(usage)
        if not isinstance(usage, Mapping):
            return

        _set_span_attribute(
            span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
            usage.get("completion_tokens"),
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens")
        )

        return
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set response attributes for response span, error: %s", str(ex)
        )


def set_request_metrics_attributes(span, request_metrics: RequestMetrics) -> None:
    if not span.is_recording():
        return

    attributes = {
        SpanAttributes.GEN_AI_REQUEST_ID: request_metrics.request_id,
        SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: request_metrics.prompt_tokens,
        SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (
            request_metrics.completion_tokens
        ),
        SpanAttributes.GEN_AI_USAGE_NUM_SEQUENCES: request_metrics.num_sequences,
        SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS: request_metrics.reasoning_tokens,
        SpanAttributes.GEN_AI_USAGE_CACHED_TOKENS: request_metrics.cache_hit_tokens,
        SpanAttributes.SGLANG_KV_CACHE_HIT_TOKENS: request_metrics.cache_hit_tokens,
        SpanAttributes.SGLANG_KV_CACHE_MISS_TOKENS: request_metrics.cache_miss_tokens,
        SpanAttributes.SGLANG_KV_CACHE_HIT_RATIO: request_metrics.cache_hit_ratio,
        SpanAttributes.SGLANG_KV_CACHE_DEVICE_HIT_TOKENS: (
            request_metrics.cache_device_tokens
        ),
        SpanAttributes.SGLANG_KV_CACHE_HOST_HIT_TOKENS: (
            request_metrics.cache_host_tokens
        ),
        SpanAttributes.SGLANG_KV_CACHE_STORAGE_HIT_TOKENS: (
            request_metrics.cache_storage_tokens
        ),
        SpanAttributes.SGLANG_KV_CACHE_STORAGE_BACKEND: (
            request_metrics.cache_storage_backend
        ),
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: request_metrics.queue_time,
        SpanAttributes.SGLANG_SCHEDULER_PREFILL_DURATION: (
            request_metrics.scheduler_prefill_time
        ),
        SpanAttributes.SGLANG_BACKEND_E2E_DURATION: request_metrics.backend_e2e_time,
        SpanAttributes.SGLANG_DECODE_THROUGHPUT: request_metrics.decode_throughput,
        SpanAttributes.SGLANG_REQUEST_RETRACTIONS: request_metrics.num_retractions,
        SpanAttributes.SGLANG_SPEC_ACCEPTED_DRAFTS: (
            request_metrics.spec_accepted_drafts
        ),
        SpanAttributes.SGLANG_SPEC_PROPOSED_DRAFTS: (
            request_metrics.spec_proposed_drafts
        ),
        SpanAttributes.SGLANG_SPEC_ACCEPT_RATIO: request_metrics.spec_accept_ratio,
        SpanAttributes.SGLANG_SPEC_VERIFY_COUNT: request_metrics.spec_verify_count,
        SpanAttributes.SGLANG_WEIGHT_VERSION: request_metrics.weight_version,
        SpanAttributes.SGLANG_DP_RANK: request_metrics.dp_rank,
    }
    for name, value in attributes.items():
        _set_span_attribute(span, name, value)
    if request_metrics.finish_reasons:
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON,
            request_metrics.finish_reasons,
        )


def should_send_prompts():
    value = os.getenv("SGLANG_OTEL_TRACE_CONTENT", os.getenv("TRACE_CONTENT", "false"))
    return value.lower() == "true"


def model_as_dict(model):
    # Keep this helper tolerant of different model containers:
    # - raw dicts
    # - msgspec Structs (fast streaming chunks)
    # - pydantic models (v1/v2)
    if isinstance(model, dict):
        return model
    if (
        _is_msgspec_available
        and hasattr(msgspec, "Struct")
        and isinstance(model, msgspec.Struct)
    ):
        # Fast-path chunks are msgspec Structs; convert to plain dict for accumulation.
        return msgspec.structs.asdict(model)
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump()

    model_dict = getattr(model, "dict", None)
    if callable(model_dict):
        return model_dict()

    parse = getattr(model, "parse", None)
    if callable(parse):  # Raw API response
        return model_as_dict(parse())

    return model


def accumulate_stream_items(item, complete_response):
    """Accumulate a streaming chunk into complete_response when provided."""
    if not is_otel_available() or complete_response is None:
        return

    item = model_as_dict(item)
    if not isinstance(item, dict):
        return
    capture_content = should_send_prompts()
    complete_response["model"] = item.get("model")

    if item.get("error"):
        complete_response["error"] = item.get("error")

    if item.get("usage"):
        complete_response["usage"] = item.get("usage")

    # prompt filter results
    if item.get("prompt_filter_results"):
        complete_response["prompt_filter_results"] = item.get("prompt_filter_results")

    if item.get("choices"):
        for choice in item.get("choices"):
            choice = model_as_dict(choice)
            if not isinstance(choice, dict):
                continue
            index = choice.get("index")
            while len(complete_response.get("choices")) <= index:
                complete_response["choices"].append(
                    {
                        "index": len(complete_response.get("choices")),
                        "message": {
                            "content": "",
                            "role": "",
                            "reasoning_content": "",
                        },
                    }
                )
            complete_choice = complete_response.get("choices")[index]
            if choice.get("finish_reason"):
                complete_choice["finish_reason"] = choice.get("finish_reason")
            if choice.get("content_filter_results"):
                complete_choice["content_filter_results"] = choice.get(
                    "content_filter_results"
                )

            delta = choice.get("delta")
            if not delta:
                continue
            delta = model_as_dict(delta)
            if not isinstance(delta, dict):
                continue

            if capture_content and delta.get("content"):
                complete_choice["message"]["content"] += delta.get("content")

            if capture_content and delta.get("reasoning_content"):
                complete_choice["message"]["reasoning_content"] += delta.get(
                    "reasoning_content"
                )

            if delta.get("role"):
                complete_choice["message"]["role"] = delta.get("role")

            if capture_content and delta and delta.get("tool_calls"):
                tool_calls = delta.get("tool_calls")
                if not tool_calls:
                    continue

                if not complete_choice["message"].get("tool_calls"):
                    complete_choice["message"]["tool_calls"] = []

                for tool_call in tool_calls:
                    tool_call = model_as_dict(tool_call)
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_index = tool_call.get("index")
                    if tool_call_index is None:
                        continue
                    i = int(tool_call_index)
                    while len(complete_choice["message"]["tool_calls"]) <= i:
                        complete_choice["message"]["tool_calls"].append(
                            {"id": "", "function": {"name": "", "arguments": ""}}
                        )

                    span_tool_call = complete_choice["message"]["tool_calls"][i]
                    span_function = span_tool_call["function"]
                    tool_call_function = tool_call.get("function")

                    if tool_call.get("id"):
                        span_tool_call["id"] = tool_call.get("id")
                    if tool_call_function and tool_call_function.get("name"):
                        span_function["name"] = tool_call_function.get("name")
                    if tool_call_function and tool_call_function.get("arguments"):
                        span_function["arguments"] += tool_call_function.get(
                            "arguments"
                        )


@dataclass
class RequestObservation:
    span: Any
    start_time: float
    trace_headers: Dict[str, str] = field(default_factory=dict)
    ended: bool = False


class OpenTelemetryProvider:
    def __init__(self):
        # Initialization is intentionally lazy. SGLang's built-in tracing is
        # configured during FastAPI lifespan; initializing here at module import
        # time would install a competing global TracerProvider first.
        self.tracer = None
        self.meter = None
        self._initialization_attempted = False

    def _ensure_initialized(self) -> bool:
        if self.tracer is not None:
            return True
        if getattr(self, "_initialization_attempted", False):
            return False
        self._initialization_attempted = True
        if not is_otel_available():
            return False

        try:
            self.tracer = init_tracer("sglang.openai")
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to initialize OpenTelemetry tracing; request spans are "
                "disabled. Error: %s",
                str(ex),
            )
            return False

        try:
            self.meter = init_metrics("sglang.openai")
        except Exception as ex:  # pylint: disable=broad-except
            # Metrics and traces have independent exporters. A metrics endpoint
            # error must not suppress otherwise healthy request traces.
            self.meter = None
            logger.warning(
                "Failed to initialize OpenTelemetry metrics; request metrics are "
                "disabled while tracing remains active. Error: %s",
                str(ex),
            )
        return True

    def start_request(
        self,
        name: str,
        headers: Optional[Mapping[str, str]],
        request: Any,
        start_time: Optional[float] = None,
    ) -> Optional[RequestObservation]:
        if not self._ensure_initialized():
            return None

        start_time = start_time if start_time is not None else time.time()
        span = self.tracer.start_span(
            name=name,
            kind=SpanKind.SERVER,
            context=extract_trace_context(headers),
            start_time=int(start_time * 1e9),
            attributes={
                SpanAttributes.GEN_AI_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value
            },
        )
        set_request_attributes(span, request)
        if should_send_prompts():
            set_prompts(span, getattr(request, "messages", None))
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_RESPONSE_MODEL, getattr(request, "model", None)
        )

        trace_headers: Dict[str, str] = {}
        try:
            TraceContextTextMapPropagator().inject(
                trace_headers, context=set_span_in_context(span)
            )
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug("Failed to inject OpenAI request trace context: %s", ex)
        return RequestObservation(
            span=span, start_time=start_time, trace_headers=trace_headers
        )

    @staticmethod
    def _end(observation: Optional[RequestObservation]) -> None:
        if observation is None or observation.ended:
            return
        observation.span.end()
        observation.ended = True

    def close_unfinished(self, observation: Optional[RequestObservation]) -> None:
        """Close a span if a streaming consumer disappears before completion."""
        if observation is None or observation.ended:
            return
        observation.span.set_status(
            Status(
                status_code=StatusCode.ERROR,
                description="response stream closed before completion",
            )
        )
        self._end(observation)

    def recordException(
        self,
        name,
        headers,
        request,
        exception: BaseException,
        observation: Optional[RequestObservation] = None,
        start_time: Optional[float] = None,
        request_metrics: Optional[RequestMetrics] = None,
    ):
        observation = observation or self.start_request(
            name, headers, request, start_time=start_time
        )
        if observation is None:
            return

        span = observation.span
        if request_metrics is not None:
            set_request_metrics_attributes(span, request_metrics)
        record_exception = getattr(span, "record_exception", None)
        if callable(record_exception) and isinstance(exception, Exception):
            record_exception(exception)
        _set_span_attribute(span, "error.type", type(exception).__name__)
        span.set_status(
            Status(status_code=StatusCode.ERROR, description=str(exception))
        )

        shared_attributes = metric_shared_attributes(
            response_model=getattr(request, "model", None),
            operation="chat",
            is_streaming=getattr(request, "stream", False),
            request_metrics=request_metrics,
        )
        shared_attributes["error.type"] = type(exception).__name__
        if Meters.is_metrics_inited:
            Meters.chat_counter.add(1, attributes=shared_attributes)
            Meters.chat_exception_counter.add(1, attributes=shared_attributes)
            Meters.chat_duration_histogram.record(
                max(time.time() - observation.start_time, 0.0),
                attributes=shared_attributes,
            )
            if request_metrics is not None:
                record_request_metrics(request_metrics, shared_attributes)
        self._end(observation)

    def record(
        self,
        name,
        headers,
        request,
        response,
        usage,
        start_time,
        time_of_first_token=None,
        stream=False,
        observation: Optional[RequestObservation] = None,
        request_metrics: Optional[RequestMetrics] = None,
    ):
        observation = observation or self.start_request(
            name, headers, request, start_time=start_time
        )
        if observation is None:
            return

        span = observation.span
        response_dict = model_as_dict(response)
        choices = (
            response_dict.get("choices", [])
            if isinstance(response_dict, Mapping)
            else []
        )
        usage_dict = model_as_dict(usage) if usage else {}
        if not isinstance(usage_dict, Mapping):
            usage_dict = {}
        if request_metrics is not None:
            # Streaming continuous-usage chunks are per choice. Normalize the
            # final OTel usage to one prompt plus all output sequences.
            usage_dict = {
                **usage_dict,
                "prompt_tokens": request_metrics.prompt_tokens,
                "completion_tokens": request_metrics.completion_tokens,
                "total_tokens": (
                    request_metrics.prompt_tokens + request_metrics.completion_tokens
                ),
            }

        shared_attributes = metric_shared_attributes(
            response_model=getattr(request, "model", None),
            operation="chat",
            is_streaming=getattr(request, "stream", stream),
            request_metrics=request_metrics,
        )
        if Meters.is_metrics_inited:
            Meters.chat_counter.add(1, attributes=shared_attributes)
        if choices:
            set_choice_counter_metrics(choices, shared_attributes)
        if usage_dict:
            set_token_counter_metrics(usage_dict, shared_attributes)
        if request_metrics is not None:
            set_request_metrics_attributes(span, request_metrics)
            record_request_metrics(request_metrics, shared_attributes)

        now = time.time()
        duration = max(now - observation.start_time, 0.0)
        _set_span_attribute(span, SpanAttributes.GEN_AI_LATENCY_E2E, duration)
        if Meters.is_metrics_inited:
            Meters.chat_duration_histogram.record(
                duration, attributes=shared_attributes
            )

        if stream and time_of_first_token is not None:
            if time_of_first_token > observation.start_time:
                time_to_first_token = time_of_first_token - observation.start_time
                time_to_generate = max(now - time_of_first_token, 0.0)
                _set_span_attribute(
                    span,
                    SpanAttributes.GEN_AI_STREAMING_TIME_TO_FIRST_TOKEN,
                    time_to_first_token,
                )
                _set_span_attribute(
                    span,
                    SpanAttributes.GEN_AI_STREAMING_TIME_TO_GENERATE,
                    time_to_generate,
                )
                if Meters.is_metrics_inited:
                    Meters.streaming_time_to_first_token.record(
                        time_to_first_token, attributes=shared_attributes
                    )
                    Meters.streaming_time_to_generate.record(
                        time_to_generate, attributes=shared_attributes
                    )

                inter_token_intervals = (
                    request_metrics.inter_token_intervals
                    if request_metrics is not None
                    else max(_safe_int(usage_dict.get("completion_tokens")) - 1, 0)
                )
                # TTFT accounts for the first token of each output sequence.
                # The remaining intervals are the conventional TPOT denominator.
                if inter_token_intervals > 0:
                    time_per_output_token = time_to_generate / inter_token_intervals
                    _set_span_attribute(
                        span,
                        SpanAttributes.GEN_AI_STREAMING_TIME_PER_OUTPUT_TOKEN,
                        time_per_output_token,
                    )
                    if Meters.is_metrics_inited:
                        Meters.streaming_time_per_output_token.record(
                            time_per_output_token,
                            attributes=shared_attributes,
                        )

        set_response_attributes(span, response, usage_dict)
        if should_send_prompts():
            set_completions(span, choices)

        span.set_status(Status(StatusCode.OK))
        self._end(observation)


otel_provider = OpenTelemetryProvider()
