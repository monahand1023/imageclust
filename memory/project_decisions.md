---
name: project-decisions
description: Key architectural decisions made during the CLIP/Ollama migration with rationale
metadata:
  type: project
---

# imageclust key decisions

**CLIP over DINOv2:**
- CLIP's contrastive text-image training produces semantically-organized embedding space — images of the same concept cluster together even if visually different
- DINOv2 is better for visual similarity / re-identification; CLIP is better for "what is this about" clustering
- **How to apply:** If user reports clusters that seem visually similar but semantically different, DINOv2 could be a swap. If clusters are semantically incoherent, consider fine-tuning or a domain-specific model.

**Direct HTTP for Ollama (no SDK):**
- `github.com/ollama/ollama v0.24.0` requires go 1.26 and pulls in 80+ transitive deps (gin, sqlite, bubbletea)
- Direct `net/http` POST to `/api/generate` is simpler, zero extra deps, fully sufficient for our use
- **How to apply:** Do not add the ollama Go SDK as a dependency. The REST API is stable.

**Ward clustering over HDBSCAN:**
- HDBSCAN produces noise points (unassigned images) and can't enforce max cluster size
- The app guarantees every uploaded image appears in a cluster (user expectation)
- Ward with our existing size-constraint logic satisfies this; the constraint code is in `internal/clustering/clustering.go`
- **How to apply:** HDBSCAN is only appropriate if we drop the min/max size UI and accept noise points.

**AdvancedSession over DynamicAdvancedSession:**
- `AdvancedSession` pre-allocates input/output tensors once; data is updated in-place per inference
- Avoids per-call tensor allocation; inference is serialized via mutex
- `DynamicAdvancedSession.Run(inputs, outputs []Value)` requires both input AND output tensors on every call
- **How to apply:** If adding batch inference, switch to DynamicAdvancedSession and allocate output tensors of shape [batch, 1024].

**onnxruntime_go v1.30.1 API notes:**
- `ArbitraryTensor` is a type alias for `Value`
- `AdvancedSession.Run()` — no arguments, uses pre-bound tensors
- `DynamicAdvancedSession.Run(inputs, outputs []Value) error` — both args required
- `GetInputOutputInfo(modelPath)` — probes model input/output shapes without running inference
- **How to apply:** Always check library source at `$(go env GOPATH)/pkg/mod/github.com/yalue/onnxruntime_go@v1.30.1/` when the API seems unclear.
