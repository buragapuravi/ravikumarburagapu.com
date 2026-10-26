# Operating Reliable Services

An error budget translates a service-level objective into an allowed amount of unreliability. A burn-rate alert compares recent error-budget consumption with the remaining budget. A useful service-level indicator measures an outcome that users can observe. Availability is the fraction of valid requests that receive a successful response. Latency percentiles describe the distribution of response times rather than only the average.

Retries can amplify an outage when many clients repeat failed requests together. Exponential backoff increases the wait between repeated attempts. Jitter spreads retry times across clients. A circuit breaker stops calls to a dependency after repeated failures. A bulkhead limits how much one failing dependency can consume.

Capacity planning compares expected demand with tested system limits. A load test measures behavior under a controlled request volume. A stress test pushes a service beyond its expected operating range. A saturation signal indicates that a resource is approaching exhaustion. Headroom is the capacity kept available for unexpected demand.

An incident commander coordinates response roles during an active incident. A communication lead shares status with affected stakeholders. A scribe records decisions and timestamps. A blameless review focuses on system conditions rather than individual fault. An action item should name an owner and a measurable completion condition.

Observability combines logs, metrics, and traces to help explain system behavior. A trace links spans across a request path. A metric records a numeric measurement over time. A structured log stores named fields with an event. A dashboard should answer a specific operational question. An alert should request an action that a responder can take.
