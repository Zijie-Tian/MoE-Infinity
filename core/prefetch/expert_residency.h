// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "model/model_topology.h"
#include "prefetch/expert_policy.h"

// Routed experts and generic topology nodes share a uint64_t manager index.
// Keep them in disjoint domains so dense node ids 0,1,... cannot alias
// layer-0 experts 0,1,... . Supported routed layer ids keep the high bit clear.
constexpr std::uint64_t kCanonicalNodeResidencyTag = UINT64_C(1) << 63;

inline std::uint64_t RoutedExpertResidencyKey(int layer_idx, int expert_idx) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(layer_idx))
          << 32) |
         static_cast<std::uint32_t>(expert_idx);
}

inline std::uint64_t CanonicalNodeResidencyKey(std::size_t node_id) {
  return kCanonicalNodeResidencyTag | static_cast<std::uint64_t>(node_id);
}

enum class AdmissionSource : std::uint8_t { DEMAND = 0, PREFETCH = 1 };

enum class AdmissionOutcome : std::uint8_t {
  ADMIT = 0,
  ALREADY_RESIDENT = 1,
  TRANSIENT = 2,
  REJECTED = 3
};

enum class ExpertFormat : std::uint8_t {
  BF16 = 0,
  FP8_E4M3_BLOCK128 = 1,
  MARLIN_INT4_GROUP128 = 2,
  GPT_OSS_MXFP4 = 3,
  GLM_FP8_BLOCK128 = 4,
  DEEPSEEK_V4_FP4 = 5,
  GPTQ = 6,
  AWQ = 7,
};

struct ResidencyVariantKey {
  std::uint64_t logical_expert_key = 0;
  ExpertFormat format = ExpertFormat::BF16;
  std::uint64_t generation = 0;

  bool operator==(const ResidencyVariantKey& other) const {
    return logical_expert_key == other.logical_expert_key &&
           format == other.format && generation == other.generation;
  }
  bool operator<(const ResidencyVariantKey& other) const {
    return std::tie(logical_expert_key, format, generation) <
           std::tie(other.logical_expert_key, other.format, other.generation);
  }
};

struct ResidencyVariant {
  ResidencyVariantKey key;
  NodePtr node;
  std::int64_t payload_bytes = 0;
  std::int64_t aligned_bytes = 0;
  std::int64_t workspace_bytes = 0;
};

struct ExpertExecutionDescriptor {
  ResidencyVariantKey key;
  std::uint8_t execution_kind = 0;
  std::vector<TensorID> tensor_ids;
  std::vector<std::string> tensor_roles;
};

enum class ResidencyState : std::uint8_t {
  HOST_READY = 0,
  ACTIVE = 1,
  RETIRING = 2,
};

enum class LeaseKind : std::uint8_t {
  DEMAND = 0,
  PREFETCH = 1,
  TRANSFER = 2,
  EXECUTION = 3,
  TRANSITION = 4,
};

enum class ResidencyTransactionKind : std::uint8_t {
  ADMISSION = 0,
  TRANSITION = 1,
};

class ResidencyTransferOps {
 public:
  virtual ~ResidencyTransferOps() = default;
  virtual bool MoveToHost(const NodePtr& node) = 0;
};

struct ResidencyTicket {
  std::uint64_t id = 0;
  ResidencyTransactionKind kind = ResidencyTransactionKind::ADMISSION;
  NodePtr incoming;
  NodePtr reserved_victim;
  ResidencyVariant incoming_variant;
  std::optional<ResidencyVariantKey> replaced_key;
  std::optional<ResidencyVariantKey> reserved_victim_key;
  int gpu_id = -1;
  ExpertPhase phase = ExpertPhase::MIXED;
  AdmissionSource source = AdmissionSource::DEMAND;
  AdmissionOutcome outcome = AdmissionOutcome::REJECTED;
  std::int64_t reserved_aligned_bytes = 0;
  std::int64_t reserved_workspace_bytes = 0;
  bool reservation_charged = false;
  bool transient = false;
  bool valid = false;
};

struct ResidencyEntry {
  NodePtr node;
  std::int64_t bytes = 0;
  ResidencyVariantKey key;
  std::int64_t payload_bytes = 0;
  ResidencyState state = ResidencyState::HOST_READY;
  std::uint32_t lease_count = 0;
  cudaEvent_t last_use_event = nullptr;
  // Demand-only LFU state for the current residency epoch.  Prefetch does not
  // touch these fields; an unconsumed prefetched entry therefore has frequency
  // zero.  A true eviction removes the entry and resets its history.
  std::uint64_t demand_frequency = 0;
  std::uint64_t last_demand_sequence = 0;
  bool prefetched_unconsumed = false;
  std::uint64_t residency_epoch = 0;
};

struct ExpertResidencyEvent {
  std::uint64_t sequence = 0;
  std::int64_t timestamp_ns = 0;
  std::uint64_t logical_expert_key = 0;
  std::uint64_t residency_epoch = 0;
  int gpu_id = -1;
  std::int64_t bytes = 0;
  std::string kind;
};

struct LeaseRecord {
  std::uint64_t id = 0;
  NodePtr node;
  ResidencyVariantKey key;
  LeaseKind kind = LeaseKind::DEMAND;
};

struct WorkspaceRecord {
  std::uint64_t transaction_id = 0;
  ResidencyVariantKey key;
  int gpu_id = -1;
  std::int64_t bytes = 0;
  cudaEvent_t completion_event = nullptr;
};

using ExpertPolicyStats = std::unordered_map<std::string, std::int64_t>;

class ExpertResidencyManager {
 public:
  explicit ExpertResidencyManager(
      std::shared_ptr<ResidencyTransferOps> transfer_ops);

  bool ConfigureCapacity(int gpu_id, std::int64_t capacity_bytes);
  bool IsCapacityConfigured(int gpu_id) const;
  std::int64_t CapacityBytes(int gpu_id) const;

  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode,
                                 AdmissionSource source);
  bool EvictReserved(const ResidencyTicket& ticket);
  bool CommitAdmission(const ResidencyTicket& ticket);
  bool AbortAdmission(const ResidencyTicket& ticket);

  void RegisterVariant(const ResidencyVariant& variant);
  std::optional<ResidencyVariant> RegisteredVariant(
      const ResidencyVariantKey& key) const;
  ResidencyTicket BeginAdmission(const ResidencyVariant& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode,
                                 AdmissionSource source);
  ResidencyTicket BeginTransition(const ResidencyVariantKey& current,
                                  const ResidencyVariant& incoming, int gpu_id,
                                  ExpertPhase phase, AdmissionSource source);
  bool CommitTransaction(const ResidencyTicket& transaction);
  bool AbortTransaction(const ResidencyTicket& transaction);

  std::uint64_t AcquireLease(const NodePtr& node, LeaseKind kind);
  std::uint64_t AcquireLease(const ResidencyVariantKey& key, LeaseKind kind);
  bool ReleaseLease(std::uint64_t lease_id);
  bool RecordLastUse(const ResidencyVariantKey& key, cudaEvent_t event);
  bool RecordWorkspaceUse(std::uint64_t transaction_id, cudaEvent_t event);
  std::size_t ReapWorkspace(int gpu_id);
  bool RequestRetirement(const ResidencyVariantKey& key);
  std::size_t ReapRetired(int gpu_id);
  std::vector<ResidencyEntry> ResidentGenerations(int gpu_id) const;
  std::size_t ResidentGenerationCount(int gpu_id) const;
  std::optional<ResidencyVariantKey> ActiveGeneration(
      int gpu_id, std::uint64_t logical_expert_key) const;
  void ReplaceProtectedVariants(
      const std::vector<ResidencyVariantKey>& candidates);

  void ReplaceProtectedCandidates(const NodePtrList& candidates);
  void RecordAccess(const NodePtr& node, ExpertPhase phase, bool hit,
                    bool waited_for_prefetch = false,
                    std::int64_t arrival_ns = 0);
  void RecordDemandH2D(const NodePtr& node, std::int64_t bytes);
  void RecordPrefetchH2D(const NodePtr& node, std::int64_t bytes);
  void RecordPrefetchReleaseBegin(const NodePtr& node);
  void RecordPrefetchReleaseReady(const NodePtr& node);
  std::vector<ExpertResidencyEvent> DrainEvents();

  bool ConfigureEvictionPolicy(ExpertEvictionPolicy policy);
  ExpertEvictionPolicy EvictionPolicy() const;

  void ConfigurePolicy(const PhasePolicyConfig& config);
  bool PolicyEnabled() const;
  AdmissionMode AdmissionFor(ExpertPhase phase) const;
  std::uint32_t StarvationLimit() const;
  void RecordStarvationPromotion();

  ExpertPolicyStats Snapshot() const;
  std::int64_t ResidentBytes(int gpu_id) const;
  std::size_t ResidentCount(int gpu_id) const;

 private:
  struct GpuState {
    bool configured = false;
    std::int64_t capacity_bytes = 0;
    std::int64_t resident_bytes = 0;
    std::int64_t transition_reserved_bytes = 0;
    std::int64_t workspace_bytes = 0;
    std::vector<ResidencyEntry> entries;
  };

  GpuState& StateFor(int gpu_id);
  const GpuState* FindState(int gpu_id) const;
  ResidencyEntry* FindEntry(GpuState& state, std::size_t node_id);
  const ResidencyEntry* FindEntry(const GpuState& state,
                                  std::size_t node_id) const;
  ResidencyEntry* FindEntry(GpuState& state, const ResidencyVariantKey& key);
  const ResidencyEntry* FindEntry(const GpuState& state,
                                  const ResidencyVariantKey& key) const;
  ResidencyEntry* FindEntry(const ResidencyVariantKey& key, int* gpu_id);
  ResidencyEntry* FindActiveLogicalEntry(int gpu_id, std::uint64_t logical_key);
  NodePtr SelectVictim(const GpuState& state) const;
  std::uint32_t NodeLeaseCount(std::size_t node_id) const;
  void AppendEventLocked(const ResidencyEntry& entry, int gpu_id,
                         const std::string& kind,
                         std::int64_t timestamp_ns = 0);

  mutable std::mutex mutex_;
  std::shared_ptr<ResidencyTransferOps> transfer_ops_;
  std::unordered_map<int, GpuState> gpu_states_;
  std::unordered_map<std::uint64_t, ResidencyTicket> pending_tickets_;
  std::unordered_map<std::uint64_t, LeaseRecord> leases_;
  std::vector<ResidencyVariant> registered_variants_;
  std::unordered_map<std::uint64_t, WorkspaceRecord> workspace_records_;
  // Phase-policy metadata is retained independently of cache-local LFU state.
  std::unordered_map<std::size_t, ExpertPolicyMetadata> access_metadata_;
  std::unordered_set<std::size_t> protected_ids_;
  std::vector<ResidencyVariantKey> protected_variants_;
  PhasePolicyConfig config_;
  ExpertPolicyStats counters_;

  std::uint64_t next_ticket_id_ = 1;
  std::uint64_t next_lease_id_ = 1;
  std::uint64_t access_sequence_ = 0;
  std::uint64_t demand_access_sequence_ = 0;
  std::uint64_t routed_demand_accesses_ = 0;
  std::uint64_t generic_demand_accesses_ = 0;
  std::uint64_t demand_h2d_count_ = 0;
  std::uint64_t demand_h2d_bytes_ = 0;
  std::uint64_t prefetch_h2d_count_ = 0;
  std::uint64_t prefetch_h2d_bytes_ = 0;
  std::uint64_t demand_miss_count_ = 0;
  std::uint64_t demand_miss_bytes_ = 0;
  std::uint64_t prefetch_ready_first_use_count_ = 0;
  std::uint64_t prefetch_ready_first_use_bytes_ = 0;
  std::uint64_t prefetch_late_first_use_count_ = 0;
  std::uint64_t prefetch_late_first_use_bytes_ = 0;
  std::uint64_t reuse_hit_count_ = 0;
  std::uint64_t reuse_hit_bytes_ = 0;
  std::uint64_t unused_prefetch_eviction_count_ = 0;
  std::uint64_t unused_prefetch_eviction_bytes_ = 0;
  std::vector<ExpertResidencyEvent> residency_events_;
  std::uint64_t residency_events_dropped_ = 0;
  std::uint64_t next_residency_event_sequence_ = 1;
  std::uint64_t next_residency_epoch_ = 1;
  ExpertEvictionPolicy eviction_policy_ = ExpertEvictionPolicy::LRU;
};

namespace expert_residency_detail {

inline bool HasActiveCanonicalVariant(
    const std::shared_ptr<ExpertResidencyManager>& manager, int gpu_id,
    const NodePtr& node) {
  if (!manager || !node || !node->residency_key.has_value()) return false;
  const ResidencyVariantKey expected{*node->residency_key, ExpertFormat::BF16,
                                     0};
  const auto active = manager->ActiveGeneration(gpu_id, *node->residency_key);
  return active.has_value() && *active == expected;
}

// Claim a node for demand execution while treating a manager eviction claim as
// a short reservation, not a fatal competing executor. Other busy states are
// returned to the caller unchanged so existing prefetch/fatal behavior remains
// under dispatcher control.
inline bool ClaimNodeForDemandExecution(const NodePtr& node,
                                        NodeExecState* competing_state) {
  if (!node) return false;
  while (true) {
    auto expected = NodeExecState::IDLE;
    if (node->exec_state.compare_exchange_strong(
            expected, NodeExecState::FETCHING, std::memory_order_acq_rel)) {
      if (competing_state != nullptr) *competing_state = NodeExecState::IDLE;
      return true;
    }
    if (expected == NodeExecState::RESIZE_RESERVED) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      continue;
    }
    if (competing_state != nullptr) *competing_state = expected;
    return false;
  }
}

// Own one pending manager admission until commit succeeds or the scope exits.
// The guard is also used by unit tests so failure cleanup is exercised through
// the same implementation as the production fetch worker.
class PendingAdmissionGuard {
 public:
  explicit PendingAdmissionGuard(
      std::shared_ptr<ExpertResidencyManager> manager)
      : manager_(std::move(manager)) {}

  void Arm(const ResidencyTicket* ticket) {
    ticket_ = ticket;
    armed_ = ticket_ != nullptr && ticket_->valid && ticket_->id != 0;
  }

  void Abort() noexcept {
    if (!armed_) return;
    try {
      manager_->AbortAdmission(*ticket_);
    } catch (...) {
      // Preserve the original fetch/transfer failure.
    }
    armed_ = false;
    ticket_ = nullptr;
  }

  void Disarm() noexcept {
    armed_ = false;
    ticket_ = nullptr;
  }

  ~PendingAdmissionGuard() { Abort(); }

 private:
  std::shared_ptr<ExpertResidencyManager> manager_;
  const ResidencyTicket* ticket_ = nullptr;
  bool armed_ = false;
};

}  // namespace expert_residency_detail

class ExpertResidencyClient {
 public:
  ExpertResidencyClient(std::shared_ptr<ExpertResidencyManager> manager,
                        AdmissionSource source)
      : manager_(std::move(manager)), source_(source) {}

  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode) {
    return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
  }

  ResidencyTicket BeginAdmission(const ResidencyVariant& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode) {
    return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
  }

  ResidencyTicket BeginTransition(const ResidencyVariantKey& current,
                                  const ResidencyVariant& incoming, int gpu_id,
                                  ExpertPhase phase) {
    return manager_->BeginTransition(current, incoming, gpu_id, phase, source_);
  }

  std::uint64_t AcquireLease(const ResidencyVariantKey& key, LeaseKind kind) {
    return manager_->AcquireLease(key, kind);
  }
  bool ReleaseLease(std::uint64_t lease_id) {
    return manager_->ReleaseLease(lease_id);
  }
  bool RecordLastUse(const ResidencyVariantKey& key, cudaEvent_t event) {
    return manager_->RecordLastUse(key, event);
  }
  bool RecordWorkspaceUse(std::uint64_t transaction_id, cudaEvent_t event) {
    return manager_->RecordWorkspaceUse(transaction_id, event);
  }
  std::size_t ReapWorkspace(int gpu_id) {
    return manager_->ReapWorkspace(gpu_id);
  }
  bool RequestRetirement(const ResidencyVariantKey& key) {
    return manager_->RequestRetirement(key);
  }
  std::size_t ReapRetired(int gpu_id) { return manager_->ReapRetired(gpu_id); }
  ExpertPolicyStats Snapshot() const { return manager_->Snapshot(); }

  std::shared_ptr<ExpertResidencyManager> manager() const { return manager_; }

 private:
  std::shared_ptr<ExpertResidencyManager> manager_;
  AdmissionSource source_;
};

class TopologyMoveToHostOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override;
};

extern std::shared_ptr<ExpertResidencyManager> kExpertResidencyManager;
extern std::unique_ptr<ExpertResidencyClient> kDemandResidencyClient;
extern std::unique_ptr<ExpertResidencyClient> kPrefetchResidencyClient;

void InitExpertResidency();
void ResetExpertResidency();
void ConfigureExpertResidencyCapacityFromTopology();
