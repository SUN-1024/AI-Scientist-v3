# Local model endpoints in agent containers

Docker bridge gives each agent its own loopback interface. A provider configured
as `http://127.0.0.1:3425/v1` therefore cannot reach a model service listening on
the launcher's loopback interface without a route between those namespaces.

The container wrapper now connects a selected Codex loopback provider through a
private Unix socket in the agent's shared home. The agent keeps its original
provider URL, profile, model, authentication, and retry policy. Each request opens
a fresh upstream connection, so the existing model client can retry after the
backend becomes available again.

The destination is read from `CONTAINER_AUTH_ROOT/.codex/config.toml` and the
explicitly selected `<profile>.config.toml`. The guest's writable configuration
cannot choose the host-side destination. Only the selected HTTP(S) loopback
endpoint is forwarded. The helper does not parse or log request contents.

The route is created for Codex calls using `container.network=bridge`. It works
with single-use containers, commands in reusable containers, and the isolated
concept-figure renderer. `host`, `none`, and shared-container network modes keep
their existing behavior. In particular, `none` does not gain a model route.

`AGENT_CONTAINER_MODEL_LOOPBACK=auto` is the default. Set it to `off` when a model
service intentionally runs inside the agent container. Explicit `on` requires
a host `CONTAINER_AUTH_ROOT`. An occupied listener or an unsafe configuration
fails before the model command starts. The helper removes its private files and
socket when the call exits and forwards interruption signals to that call.

This route connects an agent to its launcher. It does not keep a laptop-hosted
model service available during sleep, provision an upstream provider, or restart
a remote server. Unattended deployments still need an available model backend
and a supervised connection when that backend is on another machine. Transport
recovery must preserve the study's selected model and completed research stages.
