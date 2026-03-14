// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified agent-aware routing controller.
//!
//! Owns cache control (prefix pinning) and session control (subagent KV isolation)
//! behind a single abstraction. The controller manages:
//!
//! - Lazy-initialized event plane clients for both endpoints
//! - A session affinity table (session_id -> worker_id) so clients don't need to
//!   track backend_instance_id across turns
//! - Post-route actions (pin prefix, close session) deferred to RequestGuard::finish()
//! - Background reaper for expired session affinity entries

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use dashmap::DashMap;
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use tokio::sync::OnceCell;

use crate::preprocessor::PreprocessedRequest;
use crate::protocols::TokenIdType;
use crate::protocols::openai::nvext::SessionAction;

/// Untyped event plane client shared by both cache_control and session_control.
pub type EventPlaneClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

/// Session affinity entry: worker ID + expiry based on session timeout.
struct AffinityEntry {
    worker_id: u64,
    expires_at: Instant,
}

/// Deferred actions captured after routing, executed on RequestGuard::finish().
pub struct PostRouteActions {
    pub pin: Option<PinAction>,
    pub session_close: Option<SessionCloseAction>,
}

/// Deferred prefix pin after generation completes.
pub struct PinAction {
    pub token_ids: Vec<TokenIdType>,
    pub client: EventPlaneClient,
    pub instance_id: u64,
    pub ttl_seconds: u64,
}

/// Deferred session close after generation completes.
pub struct SessionCloseAction {
    pub session_id: String,
    pub client: EventPlaneClient,
    pub instance_id: u64,
}

/// Unified controller for agent-aware routing features.
///
/// Replaces the separate `cache_control_cell` and `session_control_cell` on KvPushRouter.
/// Both clients connect to worker endpoints via the event plane and are lazily initialized
/// on first use.
pub struct AgentController {
    cache_control: OnceCell<EventPlaneClient>,
    session_control: OnceCell<EventPlaneClient>,
    /// session_id -> (worker_id, expiry). Enables router-side affinity resolution
    /// so clients only need to send session_id, not backend_instance_id.
    session_affinity: Arc<DashMap<String, AffinityEntry>>,
    component: Component,
}

impl AgentController {
    pub fn new(component: Component) -> Self {
        let session_affinity = Arc::new(DashMap::new());

        // Spawn reaper that sweeps expired affinity entries every 30s
        let affinity_ref = session_affinity.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                let now = Instant::now();
                affinity_ref.retain(|session_id, entry: &mut AffinityEntry| {
                    let alive = entry.expires_at > now;
                    if !alive {
                        tracing::debug!(%session_id, "Session affinity expired, removing");
                    }
                    alive
                });
            }
        });

        tracing::info!("AgentController initialized (cache_control + session_control, lazy clients)");

        AgentController {
            cache_control: OnceCell::new(),
            session_control: OnceCell::new(),
            session_affinity,
            component,
        }
    }

    /// Resolve a session_id to a pinned worker_id from the affinity table.
    /// Returns None if the session is unknown or expired.
    pub fn resolve_session_worker(&self, request: &PreprocessedRequest) -> Option<u64> {
        let routing = request.routing.as_ref()?;

        // Try session_params.id first (normal multi-turn usage),
        // fall back to session_control.session_id (for close actions where
        // the client might not set backend_instance_id)
        let session_id = routing
            .session_params
            .as_ref()
            .and_then(|sp| sp.id.as_deref())
            .or_else(|| routing.session_control.as_ref().map(|sc| sc.session_id.as_str()))?;

        let entry = self.session_affinity.get(session_id)?;
        if entry.expires_at <= Instant::now() {
            drop(entry);
            self.session_affinity.remove(session_id);
            tracing::debug!(%session_id, "Session affinity expired during resolve");
            return None;
        }
        let worker_id = entry.worker_id;
        tracing::debug!(%session_id, worker_id, "Resolved session affinity");
        Some(worker_id)
    }

    /// Called after worker selection. Fires open_session if needed, captures
    /// deferred actions (pin, close) for RequestGuard::finish().
    ///
    /// Returns Err if session_control.action == Open but the session_control
    /// client cannot be created (fail-fast: don't silently serve without isolation).
    pub async fn on_routed(
        &self,
        request: &PreprocessedRequest,
        instance_id: u64,
        context_id: &str,
    ) -> Result<PostRouteActions> {
        let routing = request.routing.as_ref();

        // Build pin action if cache_control TTL is present
        let pin = async {
            let ttl = routing.and_then(|r| r.cache_control_ttl)?;
            let client = self.get_cache_control_client().await.ok()?;
            Some(PinAction {
                token_ids: request.token_ids.clone(),
                client,
                instance_id,
                ttl_seconds: ttl,
            })
        }
        .await;

        // Handle session control
        let session_control = routing.and_then(|r| r.session_control.clone());
        let mut session_close = None;

        if let Some(ref sc) = session_control {
            match sc.action {
                SessionAction::Open => {
                    // Fail fast if we can't open the session -- don't silently
                    // serve without isolation, as subsequent turns would target
                    // a session that never existed.
                    let client = self.get_session_control_client().await?;

                    // Insert affinity entry
                    let timeout = Duration::from_secs(sc.timeout);
                    self.session_affinity.insert(
                        sc.session_id.clone(),
                        AffinityEntry {
                            worker_id: instance_id,
                            expires_at: Instant::now() + timeout,
                        },
                    );

                    // Fire open_session to worker
                    Self::spawn_session_request(
                        client,
                        serde_json::json!({
                            "action": "open_session",
                            "session_id": sc.session_id,
                            "timeout": sc.timeout,
                            "capacity_of_str_len": 65536,
                        }),
                        instance_id,
                        &sc.session_id,
                        context_id,
                        "open_session",
                    );
                }
                SessionAction::Close => {
                    // Remove affinity entry immediately
                    self.session_affinity.remove(&sc.session_id);
                    // Defer close to after generation completes
                    if let Ok(client) = self.get_session_control_client().await {
                        session_close = Some(SessionCloseAction {
                            session_id: sc.session_id.clone(),
                            client,
                            instance_id,
                        });
                    }
                }
            }
        }

        Ok(PostRouteActions {
            pin,
            session_close,
        })
    }

    /// Execute deferred post-route actions. Called from RequestGuard::finish().
    pub fn execute_post_route(actions: &PostRouteActions, context_id: &str) {
        if let Some(ref pin) = actions.pin {
            Self::spawn_pin_prefix(
                &pin.client,
                &pin.token_ids,
                pin.instance_id,
                context_id,
                pin.ttl_seconds,
            );
        }
        if let Some(ref close) = actions.session_close {
            Self::spawn_session_request(
                close.client.clone(),
                serde_json::json!({
                    "action": "close_session",
                    "session_id": close.session_id,
                }),
                close.instance_id,
                &close.session_id,
                context_id,
                "close_session",
            );
        }
    }

    // -- private helpers --

    async fn get_cache_control_client(&self) -> Result<EventPlaneClient> {
        let client = self
            .cache_control
            .get_or_try_init(|| async {
                let c = self.component.endpoint("cache_control").client().await?;
                EventPlaneClient::from_client(c, RouterMode::KV).await
            })
            .await?;
        Ok(client.clone())
    }

    async fn get_session_control_client(&self) -> Result<EventPlaneClient> {
        let client = self
            .session_control
            .get_or_try_init(|| async {
                let c = self.component.endpoint("session_control").client().await?;
                EventPlaneClient::from_client(c, RouterMode::KV).await
            })
            .await?;
        Ok(client.clone())
    }

    /// Fire-and-forget pin_prefix to the worker that served this request.
    fn spawn_pin_prefix(
        client: &EventPlaneClient,
        token_ids: &[TokenIdType],
        instance_id: u64,
        context_id: &str,
        ttl_seconds: u64,
    ) {
        let client = client.clone();
        let token_ids = token_ids.to_vec();
        let context_id = context_id.to_owned();

        tokio::spawn(async move {
            let request = serde_json::json!({
                "action": "pin_prefix",
                "token_ids": token_ids,
                "ttl_seconds": ttl_seconds,
            });
            match client.direct(SingleIn::new(request), instance_id).await {
                Ok(mut stream) => {
                    if let Some(resp) = stream.next().await {
                        tracing::info!(
                            request_id = %context_id,
                            worker_id = instance_id,
                            ?resp,
                            "pin_prefix response"
                        );
                    }
                    while stream.next().await.is_some() {}
                }
                Err(e) => {
                    tracing::warn!(
                        request_id = %context_id,
                        worker_id = instance_id,
                        "Failed to pin prefix: {e}"
                    );
                }
            }
        });
    }

    /// Fire-and-forget session lifecycle request to a specific worker.
    fn spawn_session_request(
        client: EventPlaneClient,
        request: serde_json::Value,
        instance_id: u64,
        session_id: &str,
        context_id: &str,
        action_label: &str,
    ) {
        let session_id = session_id.to_owned();
        let context_id = context_id.to_owned();
        let action_label = action_label.to_owned();

        tokio::spawn(async move {
            match client.direct(SingleIn::new(request), instance_id).await {
                Ok(mut stream) => {
                    if let Some(resp) = stream.next().await {
                        tracing::info!(
                            request_id = %context_id,
                            worker_id = instance_id,
                            %session_id,
                            ?resp,
                            "{action_label} response"
                        );
                    }
                    while stream.next().await.is_some() {}
                }
                Err(e) => {
                    tracing::warn!(
                        request_id = %context_id,
                        worker_id = instance_id,
                        %session_id,
                        "Failed {action_label}: {e}"
                    );
                }
            }
        });
    }
}
