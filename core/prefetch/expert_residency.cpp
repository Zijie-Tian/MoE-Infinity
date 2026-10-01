// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "prefetch/expert_residency.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

#include <cuda_runtime_api.h>

namespace {

bool EventComplete(cudaEvent_t event) {
  if (event == nullptr) return true;
  return cudaEventQuery(event) == cudaSuccess;
}

std::int64_t ResidencySteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

ResidencyVariant CanonicalVariant(const NodePtr& node) {
  ResidencyVariant variant;
  if (node) {
    variant.key.logical_expert_key =
        node->residency_key.value_or(CanonicalNodeResidencyKey(node->id));
    variant.key.format = ExpertFormat::BF16;
    variant.key.generation = 0;
    variant.node = node;
    variant.payload_bytes = node->byte_size;
    variant.aligned_bytes = node->byte_size;
  }
  return variant;
}

}  // namespace

std::shared_ptr<ExpertResidencyManager> kExpertResidencyManager = nullptr;
std::unique_ptr<ExpertResidencyClient> kDemandResidencyClient = nullptr;
std::unique_ptr<ExpertResidencyClient> kPrefetchResidencyClient = nullptr;

bool TopologyMoveToHostOps::MoveToHost(const NodePtr& node) {
  if (!node) return false;
  node->SetDevice(node->default_host);
  return true;
}

void InitExpertResidency() {
  if (kExpertResidencyManager != nullptr) return;
  kExpertResidencyManager = std::make_shared<ExpertResidencyManager>(
      std::make_shared<TopologyMoveToHostOps>());
  kDemandResidencyClient = std::make_unique<ExpertResidencyClient>(
      kExpertResidencyManager, AdmissionSource::DEMAND);
  kPrefetchResidencyClient = std::make_unique<ExpertResidencyClient>(
      kExpertResidencyManager, AdmissionSource::PREFETCH);
}

void ResetExpertResidency() {
  kDemandResidencyClient.reset();
  kPrefetchResidencyClient.reset();
  kExpertResidencyManager.reset();
}

void ConfigureExpertResidencyCapacityFromTopology() {
  if (kExpertResidencyManager == nullptr || kTopologyHandle == nullptr) return;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  for (int device = 0; device < device_count; ++device) {
    const std::int64_t limit =
        kTopologyHandle->GetSparseCacheLimit(CUDA_DEVICE(device));
    if (limit > 0) {
      kExpertResidencyManager->ConfigureCapacity(device, limit);
    }
  }
}

ExpertResidencyManager::ExpertResidencyManager(
    std::shared_ptr<ResidencyTransferOps> transfer_ops)
    : transfer_ops_(std::move(transfer_ops)) {}

ExpertResidencyManager::GpuState& ExpertResidencyManager::StateFor(int gpu_id) {
  return gpu_states_[gpu_id];
}

const ExpertResidencyManager::GpuState* ExpertResidencyManager::FindState(
    int gpu_id) const {
  auto it = gpu_states_.find(gpu_id);
  return it == gpu_states_.end() ? nullptr : &it->second;
}

ResidencyEntry* ExpertResidencyManager::FindEntry(GpuState& state,
                                                  std::size_t node_id) {
  for (auto& entry : state.entries) {
    if (entry.node && entry.node->id == node_id) return &entry;
  }
  return nullptr;
}

const ResidencyEntry* ExpertResidencyManager::FindEntry(
    const GpuState& state, std::size_t node_id) const {
  for (const auto& entry : state.entries) {
    if (entry.node && entry.node->id == node_id) return &entry;
  }
  return nullptr;
}

ResidencyEntry* ExpertResidencyManager::FindEntry(
    GpuState& state, const ResidencyVariantKey& key) {
  for (auto& entry : state.entries) {
    if (entry.key == key) return &entry;
  }
  return nullptr;
}

const ResidencyEntry* ExpertResidencyManager::FindEntry(
    const GpuState& state, const ResidencyVariantKey& key) const {
  for (const auto& entry : state.entries) {
    if (entry.key == key) return &entry;
  }
  return nullptr;
}

ResidencyEntry* ExpertResidencyManager::FindEntry(
    const ResidencyVariantKey& key, int* gpu_id) {
  for (auto& state_pair : gpu_states_) {
    if (auto* entry = FindEntry(state_pair.second, key)) {
      if (gpu_id != nullptr) *gpu_id = state_pair.first;
      return entry;
    }
  }
  return nullptr;
}

ResidencyEntry* ExpertResidencyManager::FindActiveLogicalEntry(
    int gpu_id, std::uint64_t logical_key) {
  auto state = gpu_states_.find(gpu_id);
  if (state == gpu_states_.end()) return nullptr;
  for (auto& entry : state->second.entries) {
    if (entry.state == ResidencyState::ACTIVE &&
        entry.key.logical_expert_key == logical_key) {
      return &entry;
    }
  }
  return nullptr;
}

std::uint32_t ExpertResidencyManager::NodeLeaseCount(
    std::size_t node_id) const {
  std::uint32_t count = 0;
  for (const auto& lease : leases_) {
    if (lease.second.node && lease.second.node->id == node_id) ++count;
  }
  return count;
}

void ExpertResidencyManager::AppendEventLocked(const ResidencyEntry& entry,
                                               int gpu_id,
                                               const std::string& kind,
                                               std::int64_t timestamp_ns) {
  constexpr std::size_t kMaxBufferedResidencyEvents = 1U << 20;
  if (residency_events_.size() >= kMaxBufferedResidencyEvents) {
    if (residency_events_dropped_ !=
        std::numeric_limits<std::uint64_t>::max()) {
      ++residency_events_dropped_;
    }
    return;
  }
  residency_events_.push_back(ExpertResidencyEvent{
      next_residency_event_sequence_++,
      timestamp_ns > 0 ? timestamp_ns : ResidencySteadyNowNs(),
      entry.key.logical_expert_key, entry.residency_epoch, gpu_id, entry.bytes,
      kind});
}

NodePtr ExpertResidencyManager::SelectVictim(const GpuState& state) const {
  NodePtr victim;
  std::size_t best_last_access = static_cast<std::size_t>(-1);
  std::uint64_t best_frequency = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t best_sequence = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t best_logical_key = std::numeric_limits<std::uint64_t>::max();
  for (const auto& entry : state.entries) {
    if (!entry.node || entry.state != ResidencyState::ACTIVE) continue;
    // Node execution state spans the complete consumer lifetime for generic
    // topology groups and routed experts. An accounted entry is not an
    // eviction candidate until that shared lifecycle says it is idle; manager
    // leases below remain the independent transaction/execution ownership
    // guard. Device is intentionally not inspected here: the existing external
    // resize path can leave a host-moved entry charged, and selecting it lets
    // EvictReserved reconcile that stale charge.
    if (entry.node->is_prefetching.load(std::memory_order_acquire) ||
        entry.node->exec_state.load(std::memory_order_acquire) !=
            NodeExecState::IDLE ||
        entry.node->pending_dispatches.load(std::memory_order_acquire) != 0) {
      continue;
    }
    if (entry.lease_count > 0 || NodeLeaseCount(entry.node->id) > 0) continue;
    if (protected_ids_.count(entry.node->id) > 0) continue;
    if (std::find(protected_variants_.begin(), protected_variants_.end(),
                  entry.key) != protected_variants_.end()) {
      continue;
    }
    if (eviction_policy_ == ExpertEvictionPolicy::LFU) {
      const bool better = !victim || entry.demand_frequency < best_frequency ||
                          (entry.demand_frequency == best_frequency &&
                           (entry.last_demand_sequence < best_sequence ||
                            (entry.last_demand_sequence == best_sequence &&
                             entry.key.logical_expert_key < best_logical_key)));
      if (better) {
        victim = entry.node;
        best_frequency = entry.demand_frequency;
        best_sequence = entry.last_demand_sequence;
        best_logical_key = entry.key.logical_expert_key;
      }
    } else if (!victim || entry.node->last_access_time < best_last_access ||
               (entry.node->last_access_time == best_last_access &&
                entry.key.logical_expert_key < best_logical_key)) {
      victim = entry.node;
      best_last_access = entry.node->last_access_time;
      best_logical_key = entry.key.logical_expert_key;
    }
  }
  return victim;
}

bool ExpertResidencyManager::ConfigureCapacity(int gpu_id,
                                               std::int64_t capacity_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (capacity_bytes < 0) return false;
  for (const auto& ticket : pending_tickets_) {
    if (ticket.second.gpu_id == gpu_id) return false;
  }
  GpuState& state = StateFor(gpu_id);
  const auto charged = state.resident_bytes + state.transition_reserved_bytes +
                       state.workspace_bytes;
  if (capacity_bytes < charged) return false;
  state.configured = true;
  state.capacity_bytes = capacity_bytes;
  return true;
}

bool ExpertResidencyManager::IsCapacityConfigured(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state != nullptr && state->configured;
}

std::int64_t ExpertResidencyManager::CapacityBytes(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state != nullptr ? state->capacity_bytes : 0;
}

void ExpertResidencyManager::RegisterVariant(const ResidencyVariant& variant) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!variant.node || variant.payload_bytes < 0 ||
      variant.aligned_bytes <= 0 || variant.workspace_bytes < 0 ||
      variant.payload_bytes > variant.aligned_bytes) {
    return;
  }
  auto duplicate =
      std::find_if(registered_variants_.begin(), registered_variants_.end(),
                   [&variant](const ResidencyVariant& existing) {
                     return existing.key == variant.key;
                   });
  if (duplicate == registered_variants_.end()) {
    registered_variants_.push_back(variant);
  }
}

std::optional<ResidencyVariant> ExpertResidencyManager::RegisteredVariant(
    const ResidencyVariantKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = std::find_if(
      registered_variants_.begin(), registered_variants_.end(),
      [&key](const ResidencyVariant& variant) { return variant.key == key; });
  return found == registered_variants_.end()
             ? std::nullopt
             : std::optional<ResidencyVariant>(*found);
}

ResidencyTicket ExpertResidencyManager::BeginAdmission(const NodePtr& incoming,
                                                       int gpu_id,
                                                       ExpertPhase phase,
                                                       AdmissionMode mode,
                                                       AdmissionSource source) {
  return BeginAdmission(CanonicalVariant(incoming), gpu_id, phase, mode,
                        source);
}

ResidencyTicket ExpertResidencyManager::BeginAdmission(
    const ResidencyVariant& incoming, int gpu_id, ExpertPhase phase,
    AdmissionMode mode, AdmissionSource source) {
  std::lock_guard<std::mutex> lock(mutex_);
  ResidencyTicket ticket;
  ticket.kind = ResidencyTransactionKind::ADMISSION;
  ticket.incoming = incoming.node;
  ticket.incoming_variant = incoming;
  ticket.gpu_id = gpu_id;
  ticket.phase = phase;
  ticket.source = source;

  const GpuState* found = FindState(gpu_id);
  if (!incoming.node || incoming.aligned_bytes <= 0 || found == nullptr ||
      !found->configured) {
    return ticket;
  }
  GpuState& state = StateFor(gpu_id);
  if ((incoming.key.logical_expert_key & kCanonicalNodeResidencyTag) == 0) {
    for (const auto& pending : pending_tickets_) {
      if (pending.second.valid &&
          pending.second.incoming_variant.key.logical_expert_key ==
              incoming.key.logical_expert_key) {
        return ticket;
      }
    }
  }
  if (FindEntry(state, incoming.key) != nullptr) {
    ticket.outcome = AdmissionOutcome::ALREADY_RESIDENT;
    return ticket;
  }
  for (const auto& entry : state.entries) {
    if (entry.state == ResidencyState::ACTIVE &&
        entry.key.logical_expert_key == incoming.key.logical_expert_key) {
      return ticket;
    }
  }

  const auto requested = incoming.aligned_bytes + incoming.workspace_bytes;
  const auto charged = state.resident_bytes + state.transition_reserved_bytes +
                       state.workspace_bytes;
  const auto free_bytes = state.capacity_bytes - charged;
  ticket.id = next_ticket_id_++;
  if (requested > free_bytes) {
    NodePtr victim = SelectVictim(state);
    if (!victim) {
      if (mode == AdmissionMode::TRANSIENT_ON_PRESSURE) {
        ticket.outcome = AdmissionOutcome::TRANSIENT;
        ticket.transient = true;
        ticket.valid = true;
        pending_tickets_[ticket.id] = ticket;
      }
      return ticket;
    }
    ticket.reserved_victim = victim;
    if (auto* victim_entry = FindEntry(state, victim->id)) {
      ticket.reserved_victim_key = victim_entry->key;
      ++victim_entry->lease_count;
    }
  }

  ticket.outcome = AdmissionOutcome::ADMIT;
  ticket.reserved_aligned_bytes = incoming.aligned_bytes;
  ticket.reserved_workspace_bytes = incoming.workspace_bytes;
  ticket.valid = true;
  if (!ticket.reserved_victim_key.has_value()) {
    state.transition_reserved_bytes += ticket.reserved_aligned_bytes;
    state.workspace_bytes += ticket.reserved_workspace_bytes;
    ticket.reservation_charged = true;
  }
  pending_tickets_[ticket.id] = ticket;
  return ticket;
}

ResidencyTicket ExpertResidencyManager::BeginTransition(
    const ResidencyVariantKey& current, const ResidencyVariant& incoming,
    int gpu_id, ExpertPhase phase, AdmissionSource source) {
  std::lock_guard<std::mutex> lock(mutex_);
  ResidencyTicket ticket;
  ticket.kind = ResidencyTransactionKind::TRANSITION;
  ticket.incoming = incoming.node;
  ticket.incoming_variant = incoming;
  ticket.replaced_key = current;
  ticket.gpu_id = gpu_id;
  ticket.phase = phase;
  ticket.source = source;

  const GpuState* found = FindState(gpu_id);
  if (!incoming.node || incoming.aligned_bytes <= 0 || found == nullptr ||
      !found->configured ||
      current.logical_expert_key != incoming.key.logical_expert_key) {
    return ticket;
  }
  GpuState& state = StateFor(gpu_id);
  ResidencyEntry* current_entry = FindEntry(state, current);
  if (current_entry == nullptr ||
      current_entry->state != ResidencyState::ACTIVE)
    return ticket;
  for (const auto& pending : pending_tickets_) {
    if (pending.second.kind == ResidencyTransactionKind::TRANSITION &&
        pending.second.replaced_key.has_value() &&
        pending.second.replaced_key->logical_expert_key ==
            current.logical_expert_key) {
      return ticket;
    }
  }
  const auto requested = incoming.aligned_bytes + incoming.workspace_bytes;
  const auto charged = state.resident_bytes + state.transition_reserved_bytes +
                       state.workspace_bytes;
  if (charged + requested > state.capacity_bytes) return ticket;

  ticket.id = next_ticket_id_++;
  ticket.outcome = AdmissionOutcome::ADMIT;
  ticket.reserved_aligned_bytes = incoming.aligned_bytes;
  ticket.reserved_workspace_bytes = incoming.workspace_bytes;
  ticket.valid = true;
  ++current_entry->lease_count;
  state.transition_reserved_bytes += ticket.reserved_aligned_bytes;
  state.workspace_bytes += ticket.reserved_workspace_bytes;
  ticket.reservation_charged = true;
  pending_tickets_[ticket.id] = ticket;
  return ticket;
}

bool ExpertResidencyManager::EvictReserved(const ResidencyTicket& ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pending_it = pending_tickets_.find(ticket.id);
  if (pending_it == pending_tickets_.end() || !pending_it->second.valid ||
      !pending_it->second.reserved_victim_key.has_value()) {
    return false;
  }
  ResidencyTicket& pending = pending_it->second;
  GpuState& state = StateFor(pending.gpu_id);
  const auto requested =
      pending.reserved_aligned_bytes + pending.reserved_workspace_bytes;
  auto charged = [&state]() {
    return state.resident_bytes + state.transition_reserved_bytes +
           state.workspace_bytes;
  };

  while (charged() + requested > state.capacity_bytes) {
    if (!pending.reserved_victim_key.has_value()) {
      NodePtr next = SelectVictim(state);
      if (!next) return false;
      ResidencyEntry* next_entry = FindEntry(state, next->id);
      if (next_entry == nullptr) return false;
      ++next_entry->lease_count;
      pending.reserved_victim = next;
      pending.reserved_victim_key = next_entry->key;
    }

    ResidencyEntry* victim = FindEntry(state, *pending.reserved_victim_key);
    if (victim == nullptr || victim->lease_count != 1) return false;
    NodePtr victim_node = victim->node;
    auto expected = NodeExecState::IDLE;
    if (!victim_node->exec_state.compare_exchange_strong(
            expected, NodeExecState::RESIZE_RESERVED,
            std::memory_order_acq_rel)) {
      return false;
    }
    bool moved = false;
    try {
      moved = !transfer_ops_ || transfer_ops_->MoveToHost(victim_node);
    } catch (...) {
      victim_node->exec_state.store(NodeExecState::IDLE,
                                    std::memory_order_release);
      throw;
    }
    if (!moved) {
      victim_node->exec_state.store(NodeExecState::IDLE,
                                    std::memory_order_release);
      return false;
    }
    if ((victim->key.logical_expert_key & kCanonicalNodeResidencyTag) == 0) {
      AppendEventLocked(*victim, pending.gpu_id, "evict");
    }
    if (victim->prefetched_unconsumed &&
        (victim->key.logical_expert_key & kCanonicalNodeResidencyTag) == 0) {
      if (unused_prefetch_eviction_count_ !=
          std::numeric_limits<std::uint64_t>::max()) {
        ++unused_prefetch_eviction_count_;
      }
      const auto bytes = static_cast<std::uint64_t>(
          std::max<std::int64_t>(victim->bytes, static_cast<std::int64_t>(0)));
      unused_prefetch_eviction_bytes_ =
          bytes > std::numeric_limits<std::uint64_t>::max() -
                      unused_prefetch_eviction_bytes_
              ? std::numeric_limits<std::uint64_t>::max()
              : unused_prefetch_eviction_bytes_ + bytes;
      AppendEventLocked(*victim, pending.gpu_id, "unused_prefetch_evict");
    }
    state.resident_bytes -= victim->bytes;
    const auto key = victim->key;
    state.entries.erase(
        std::remove_if(
            state.entries.begin(), state.entries.end(),
            [&key](const ResidencyEntry& entry) { return entry.key == key; }),
        state.entries.end());
    victim_node->exec_state.store(NodeExecState::IDLE,
                                  std::memory_order_release);
    pending.reserved_victim = nullptr;
    pending.reserved_victim_key.reset();
  }

  // Capacity can become sufficient between BeginAdmission and this call when
  // another pending transaction aborts. In that case the initially reserved
  // victim was never consumed; release its reservation lease before charging
  // the incoming transaction.
  if (pending.reserved_victim_key.has_value()) {
    if (auto* unused = FindEntry(state, *pending.reserved_victim_key)) {
      if (unused->lease_count > 0) --unused->lease_count;
    }
    pending.reserved_victim = nullptr;
    pending.reserved_victim_key.reset();
  }

  state.transition_reserved_bytes += pending.reserved_aligned_bytes;
  state.workspace_bytes += pending.reserved_workspace_bytes;
  pending.reservation_charged = true;
  return true;
}

bool ExpertResidencyManager::CommitTransaction(
    const ResidencyTicket& transaction) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pending_it = pending_tickets_.find(transaction.id);
  if (pending_it == pending_tickets_.end() || !pending_it->second.valid)
    return false;
  ResidencyTicket pending = pending_it->second;
  GpuState& state = StateFor(pending.gpu_id);
  if (pending.reserved_victim_key.has_value()) {
    if (auto* victim = FindEntry(state, *pending.reserved_victim_key)) {
      if (victim->lease_count > 0) --victim->lease_count;
    }
    pending_tickets_.erase(pending_it);
    return false;
  }
  if (pending.transient) {
    pending_tickets_.erase(pending_it);
    return true;
  }
  if (!pending.reservation_charged) {
    pending_tickets_.erase(pending_it);
    return false;
  }

  std::uint64_t carried_frequency = 0;
  std::uint64_t carried_sequence = 0;
  bool carried_prefetched_unconsumed = false;
  std::uint64_t carried_residency_epoch = 0;
  if (pending.kind == ResidencyTransactionKind::TRANSITION &&
      pending.replaced_key.has_value()) {
    ResidencyEntry* replaced = FindEntry(state, *pending.replaced_key);
    if (replaced == nullptr || replaced->state != ResidencyState::ACTIVE) {
      pending_tickets_.erase(pending_it);
      state.transition_reserved_bytes -= pending.reserved_aligned_bytes;
      state.workspace_bytes -= pending.reserved_workspace_bytes;
      return false;
    }
    carried_frequency = replaced->demand_frequency;
    carried_sequence = replaced->last_demand_sequence;
    carried_prefetched_unconsumed = replaced->prefetched_unconsumed;
    carried_residency_epoch = replaced->residency_epoch;
    if (replaced->lease_count > 0) --replaced->lease_count;
    replaced->state = ResidencyState::RETIRING;
  }

  ResidencyEntry entry;
  entry.node = pending.incoming_variant.node;
  entry.bytes = pending.incoming_variant.aligned_bytes;
  entry.key = pending.incoming_variant.key;
  entry.payload_bytes = pending.incoming_variant.payload_bytes;
  entry.state = ResidencyState::ACTIVE;
  entry.demand_frequency = carried_frequency;
  entry.last_demand_sequence = carried_sequence;
  entry.prefetched_unconsumed = carried_prefetched_unconsumed ||
                                pending.source == AdmissionSource::PREFETCH;
  entry.residency_epoch = carried_residency_epoch != 0
                              ? carried_residency_epoch
                              : next_residency_epoch_++;
  state.entries.push_back(entry);
  state.resident_bytes += entry.bytes;
  state.transition_reserved_bytes -= pending.reserved_aligned_bytes;
  if (pending.reserved_workspace_bytes > 0) {
    workspace_records_[pending.id] = WorkspaceRecord{
        pending.id, pending.incoming_variant.key, pending.gpu_id,
        pending.reserved_workspace_bytes, nullptr};
  }
  pending_tickets_.erase(pending_it);
  if (pending.source == AdmissionSource::PREFETCH &&
      (entry.key.logical_expert_key & kCanonicalNodeResidencyTag) == 0) {
    AppendEventLocked(state.entries.back(), pending.gpu_id, "prefetch_commit");
  }
  return true;
}

bool ExpertResidencyManager::CommitAdmission(const ResidencyTicket& ticket) {
  return CommitTransaction(ticket);
}

bool ExpertResidencyManager::AbortTransaction(
    const ResidencyTicket& transaction) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pending_it = pending_tickets_.find(transaction.id);
  if (pending_it == pending_tickets_.end()) return false;
  ResidencyTicket pending = pending_it->second;
  GpuState& state = StateFor(pending.gpu_id);
  if (pending.reserved_victim_key.has_value()) {
    if (auto* victim = FindEntry(state, *pending.reserved_victim_key)) {
      if (victim->lease_count > 0) --victim->lease_count;
    }
  }
  if (pending.kind == ResidencyTransactionKind::TRANSITION &&
      pending.replaced_key.has_value()) {
    if (auto* replaced = FindEntry(state, *pending.replaced_key)) {
      if (replaced->lease_count > 0) --replaced->lease_count;
    }
  }
  if (pending.reservation_charged) {
    state.transition_reserved_bytes -= pending.reserved_aligned_bytes;
    state.workspace_bytes -= pending.reserved_workspace_bytes;
  }
  pending_tickets_.erase(pending_it);
  return true;
}

bool ExpertResidencyManager::AbortAdmission(const ResidencyTicket& ticket) {
  return AbortTransaction(ticket);
}

std::uint64_t ExpertResidencyManager::AcquireLease(const NodePtr& node,
                                                   LeaseKind kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node) return 0;
  for (auto& state : gpu_states_) {
    if (auto* entry = FindEntry(state.second, node->id)) {
      LeaseRecord record{next_lease_id_++, node, entry->key, kind};
      ++entry->lease_count;
      leases_[record.id] = record;
      return record.id;
    }
  }
  return 0;
}

std::uint64_t ExpertResidencyManager::AcquireLease(
    const ResidencyVariantKey& key, LeaseKind kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  int gpu_id = -1;
  ResidencyEntry* entry = FindEntry(key, &gpu_id);
  if (entry == nullptr) return 0;
  LeaseRecord record{next_lease_id_++, entry->node, key, kind};
  ++entry->lease_count;
  leases_[record.id] = record;
  return record.id;
}

bool ExpertResidencyManager::ReleaseLease(std::uint64_t lease_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto lease_it = leases_.find(lease_id);
  if (lease_it == leases_.end()) return false;
  int gpu_id = -1;
  if (auto* entry = FindEntry(lease_it->second.key, &gpu_id)) {
    if (entry->lease_count > 0) --entry->lease_count;
  }
  leases_.erase(lease_it);
  return true;
}

bool ExpertResidencyManager::RecordLastUse(const ResidencyVariantKey& key,
                                           cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(mutex_);
  int gpu_id = -1;
  auto* entry = FindEntry(key, &gpu_id);
  if (entry == nullptr) return false;
  entry->last_use_event = event;
  return true;
}

bool ExpertResidencyManager::RecordWorkspaceUse(std::uint64_t transaction_id,
                                                cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = workspace_records_.find(transaction_id);
  if (it == workspace_records_.end()) return false;
  it->second.completion_event = event;
  return true;
}

std::size_t ExpertResidencyManager::ReapWorkspace(int gpu_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t reaped = 0;
  for (auto it = workspace_records_.begin(); it != workspace_records_.end();) {
    if (it->second.gpu_id == gpu_id &&
        EventComplete(it->second.completion_event)) {
      StateFor(gpu_id).workspace_bytes -= it->second.bytes;
      it = workspace_records_.erase(it);
      ++reaped;
    } else {
      ++it;
    }
  }
  return reaped;
}

bool ExpertResidencyManager::RequestRetirement(const ResidencyVariantKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  int gpu_id = -1;
  auto* entry = FindEntry(key, &gpu_id);
  if (entry == nullptr || entry->state != ResidencyState::ACTIVE) return false;
  entry->state = ResidencyState::RETIRING;
  return true;
}

std::size_t ExpertResidencyManager::ReapRetired(int gpu_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  GpuState& state = StateFor(gpu_id);
  std::size_t reaped = 0;
  for (auto it = state.entries.begin(); it != state.entries.end();) {
    if (it->state == ResidencyState::RETIRING && it->lease_count == 0 &&
        EventComplete(it->last_use_event) &&
        (!transfer_ops_ || transfer_ops_->MoveToHost(it->node))) {
      state.resident_bytes -= it->bytes;
      it = state.entries.erase(it);
      ++reaped;
    } else {
      ++it;
    }
  }
  return reaped;
}

std::vector<ResidencyEntry> ExpertResidencyManager::ResidentGenerations(
    int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state == nullptr ? std::vector<ResidencyEntry>{} : state->entries;
}

std::size_t ExpertResidencyManager::ResidentGenerationCount(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state == nullptr ? 0 : state->entries.size();
}

std::optional<ResidencyVariantKey> ExpertResidencyManager::ActiveGeneration(
    int gpu_id, std::uint64_t logical_expert_key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  if (state == nullptr) return std::nullopt;
  for (const auto& entry : state->entries) {
    if (entry.state == ResidencyState::ACTIVE &&
        entry.key.logical_expert_key == logical_expert_key) {
      return entry.key;
    }
  }
  return std::nullopt;
}

void ExpertResidencyManager::ReplaceProtectedVariants(
    const std::vector<ResidencyVariantKey>& candidates) {
  std::lock_guard<std::mutex> lock(mutex_);
  protected_variants_ = candidates;
}

void ExpertResidencyManager::ReplaceProtectedCandidates(
    const NodePtrList& candidates) {
  std::lock_guard<std::mutex> lock(mutex_);
  protected_ids_.clear();
  for (const auto& node : candidates) {
    if (node) protected_ids_.insert(node->id);
  }
}

void ExpertResidencyManager::ConfigurePolicy(const PhasePolicyConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  counters_["enabled"] = config.enabled ? 1 : 0;
}

bool ExpertResidencyManager::PolicyEnabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.enabled;
}

AdmissionMode ExpertResidencyManager::AdmissionFor(ExpertPhase phase) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return EffectivePhase(phase) == ExpertPhase::PREFILL
             ? config_.prefill_admission
             : config_.decode_admission;
}

std::uint32_t ExpertResidencyManager::StarvationLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.starvation_limit;
}

void ExpertResidencyManager::RecordStarvationPromotion() {
  std::lock_guard<std::mutex> lock(mutex_);
  counters_["starvation_promotions"] += 1;
}

void ExpertResidencyManager::RecordAccess(const NodePtr& node,
                                          ExpertPhase phase, bool hit,
                                          bool waited_for_prefetch,
                                          std::int64_t arrival_ns) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node) return;
  (void)hit;

  // Preserve the existing phase-policy accounting independently of the
  // selected cache eviction policy.
  ExpertPolicyMetadata& meta = access_metadata_[node->id];
  const std::uint64_t phase_sequence = ++access_sequence_;
  const ExpertPhase effective = EffectivePhase(phase);
  if (effective == ExpertPhase::PREFILL) {
    ++meta.prefill_accesses;
    meta.last_prefill_sequence = phase_sequence;
  } else {
    ++meta.decode_accesses;
    meta.last_decode_sequence = phase_sequence;
  }

  auto& access_count = node->residency_key.has_value()
                           ? routed_demand_accesses_
                           : generic_demand_accesses_;
  if (access_count != std::numeric_limits<std::uint64_t>::max()) {
    ++access_count;
  }

  const auto logical_key =
      node->residency_key.value_or(CanonicalNodeResidencyKey(node->id));
  const int gpu_id = node->device.is_cuda() ? node->device.index() : -1;
  ResidencyEntry* entry = FindActiveLogicalEntry(gpu_id, logical_key);
  if (entry == nullptr) return;
  const bool routed = node->residency_key.has_value();
  const auto entry_bytes = static_cast<std::uint64_t>(
      std::max<std::int64_t>(entry->bytes, static_cast<std::int64_t>(0)));
  auto saturating_add = [](std::uint64_t* value, std::uint64_t increment) {
    *value = increment > std::numeric_limits<std::uint64_t>::max() - *value
                 ? std::numeric_limits<std::uint64_t>::max()
                 : *value + increment;
  };
  if (routed) {
    if (entry->prefetched_unconsumed) {
      const bool late = waited_for_prefetch || !hit;
      auto* count = late ? &prefetch_late_first_use_count_
                         : &prefetch_ready_first_use_count_;
      auto* bytes = late ? &prefetch_late_first_use_bytes_
                         : &prefetch_ready_first_use_bytes_;
      saturating_add(count, 1);
      saturating_add(bytes, entry_bytes);
      AppendEventLocked(
          *entry, gpu_id,
          late ? "prefetch_late_first_use" : "prefetch_ready_first_use",
          arrival_ns);
      entry->prefetched_unconsumed = false;
    } else if (hit) {
      saturating_add(&reuse_hit_count_, 1);
      saturating_add(&reuse_hit_bytes_, entry_bytes);
      AppendEventLocked(*entry, gpu_id, "reuse_hit", arrival_ns);
    } else {
      saturating_add(&demand_miss_count_, 1);
      saturating_add(&demand_miss_bytes_, entry_bytes);
      AppendEventLocked(*entry, gpu_id, "demand_miss", arrival_ns);
    }
  }
  if (entry->demand_frequency != std::numeric_limits<std::uint64_t>::max()) {
    ++entry->demand_frequency;
  }
  if (demand_access_sequence_ != std::numeric_limits<std::uint64_t>::max()) {
    ++demand_access_sequence_;
  }
  entry->last_demand_sequence = demand_access_sequence_;
}

void ExpertResidencyManager::RecordDemandH2D(const NodePtr& node,
                                             std::int64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (demand_h2d_count_ != std::numeric_limits<std::uint64_t>::max()) {
    ++demand_h2d_count_;
  }
  const auto increment =
      static_cast<std::uint64_t>(std::max<std::int64_t>(bytes, 0));
  demand_h2d_bytes_ =
      increment > std::numeric_limits<std::uint64_t>::max() - demand_h2d_bytes_
          ? std::numeric_limits<std::uint64_t>::max()
          : demand_h2d_bytes_ + increment;
  if (node && node->residency_key.has_value() && node->device.is_cuda()) {
    if (auto* entry = FindActiveLogicalEntry(node->device.index(),
                                             *node->residency_key)) {
      AppendEventLocked(*entry, node->device.index(), "demand_h2d");
    } else if (residency_events_.size() < (1U << 20)) {
      residency_events_.push_back(ExpertResidencyEvent{
          next_residency_event_sequence_++, ResidencySteadyNowNs(),
          *node->residency_key, 0, node->device.index(), bytes,
          "demand_h2d_uncommitted"});
    } else if (residency_events_dropped_ !=
               std::numeric_limits<std::uint64_t>::max()) {
      ++residency_events_dropped_;
    }
  }
}

void ExpertResidencyManager::RecordPrefetchH2D(const NodePtr& node,
                                               std::int64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (prefetch_h2d_count_ != std::numeric_limits<std::uint64_t>::max()) {
    ++prefetch_h2d_count_;
  }
  const auto increment =
      static_cast<std::uint64_t>(std::max<std::int64_t>(bytes, 0));
  prefetch_h2d_bytes_ = increment > std::numeric_limits<std::uint64_t>::max() -
                                        prefetch_h2d_bytes_
                            ? std::numeric_limits<std::uint64_t>::max()
                            : prefetch_h2d_bytes_ + increment;
  if (node && node->residency_key.has_value() && node->device.is_cuda()) {
    if (auto* entry = FindActiveLogicalEntry(node->device.index(),
                                             *node->residency_key)) {
      AppendEventLocked(*entry, node->device.index(), "prefetch_h2d");
    } else if (residency_events_.size() < (1U << 20)) {
      residency_events_.push_back(ExpertResidencyEvent{
          next_residency_event_sequence_++, ResidencySteadyNowNs(),
          *node->residency_key, 0, node->device.index(), bytes,
          "prefetch_h2d_discarded"});
    } else if (residency_events_dropped_ !=
               std::numeric_limits<std::uint64_t>::max()) {
      ++residency_events_dropped_;
    }
  }
}

void ExpertResidencyManager::RecordPrefetchReleaseBegin(const NodePtr& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node || !node->residency_key.has_value() || !node->device.is_cuda()) {
    return;
  }
  if (auto* entry =
          FindActiveLogicalEntry(node->device.index(), *node->residency_key)) {
    AppendEventLocked(*entry, node->device.index(), "prefetch_release_begin");
  }
}

void ExpertResidencyManager::RecordPrefetchReleaseReady(const NodePtr& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node || !node->residency_key.has_value() || !node->device.is_cuda()) {
    return;
  }
  if (auto* entry =
          FindActiveLogicalEntry(node->device.index(), *node->residency_key)) {
    AppendEventLocked(*entry, node->device.index(), "prefetch_release_ready");
  }
}

std::vector<ExpertResidencyEvent> ExpertResidencyManager::DrainEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ExpertResidencyEvent> drained;
  drained.swap(residency_events_);
  return drained;
}

bool ExpertResidencyManager::ConfigureEvictionPolicy(
    ExpertEvictionPolicy policy) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (policy != ExpertEvictionPolicy::LRU &&
      policy != ExpertEvictionPolicy::LFU) {
    return false;
  }
  eviction_policy_ = policy;
  return true;
}

ExpertEvictionPolicy ExpertResidencyManager::EvictionPolicy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return eviction_policy_;
}

ExpertPolicyStats ExpertResidencyManager::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  ExpertPolicyStats stats;
  std::int64_t resident_bytes = 0;
  std::int64_t resident_payload_bytes = 0;
  std::int64_t capacity_bytes = 0;
  std::int64_t transition_reserved_bytes = 0;
  std::int64_t workspace_bytes = 0;
  std::int64_t resident_generations = 0;
  std::int64_t retiring_generations = 0;
  std::set<std::pair<int, std::uint64_t>> resident_experts;
  for (const auto& gpu : gpu_states_) {
    resident_bytes += gpu.second.resident_bytes;
    capacity_bytes += gpu.second.capacity_bytes;
    transition_reserved_bytes += gpu.second.transition_reserved_bytes;
    workspace_bytes += gpu.second.workspace_bytes;
    resident_generations +=
        static_cast<std::int64_t>(gpu.second.entries.size());
    for (const auto& entry : gpu.second.entries) {
      resident_payload_bytes += entry.payload_bytes;
      if (entry.state == ResidencyState::ACTIVE) {
        resident_experts.emplace(gpu.first, entry.key.logical_expert_key);
      } else if (entry.state == ResidencyState::RETIRING) {
        ++retiring_generations;
      }
    }
  }
  stats["resident_bytes"] = resident_bytes;
  stats["resident_payload_bytes"] = resident_payload_bytes;
  stats["alignment_padding_bytes"] = resident_bytes - resident_payload_bytes;
  stats["resident_count"] = static_cast<std::int64_t>(resident_experts.size());
  stats["resident_experts"] =
      static_cast<std::int64_t>(resident_experts.size());
  stats["resident_generations"] = resident_generations;
  stats["retiring_generations"] = retiring_generations;
  stats["capacity_bytes"] = capacity_bytes;
  stats["transition_reserved_bytes"] = transition_reserved_bytes;
  stats["workspace_bytes"] = workspace_bytes;
  stats["registered_variants"] =
      static_cast<std::int64_t>(registered_variants_.size());
  stats["active_leases"] = static_cast<std::int64_t>(leases_.size());
  for (const auto& name :
       {"demand", "prefetch", "transfer", "execution", "transition"}) {
    stats[std::string("leases_") + name] = 0;
  }
  for (const auto& lease : leases_) {
    const char* name = "demand";
    switch (lease.second.kind) {
      case LeaseKind::PREFETCH:
        name = "prefetch";
        break;
      case LeaseKind::TRANSFER:
        name = "transfer";
        break;
      case LeaseKind::EXECUTION:
        name = "execution";
        break;
      case LeaseKind::TRANSITION:
        name = "transition";
        break;
      case LeaseKind::DEMAND:
        break;
    }
    ++stats[std::string("leases_") + name];
  }
  stats["pending_tickets"] = static_cast<std::int64_t>(pending_tickets_.size());
  stats["pending_transactions"] =
      static_cast<std::int64_t>(pending_tickets_.size());
  stats["peak_accounted_bytes"] =
      resident_bytes + transition_reserved_bytes + workspace_bytes;
  stats["eviction_policy"] = static_cast<std::int64_t>(eviction_policy_);
  const auto clamp_counter = [](std::uint64_t value) {
    return value > static_cast<std::uint64_t>(
                       std::numeric_limits<std::int64_t>::max())
               ? std::numeric_limits<std::int64_t>::max()
               : static_cast<std::int64_t>(value);
  };
  stats["routed_demand_accesses"] = clamp_counter(routed_demand_accesses_);
  stats["generic_demand_accesses"] = clamp_counter(generic_demand_accesses_);
  const auto total =
      routed_demand_accesses_ > std::numeric_limits<std::uint64_t>::max() -
                                    generic_demand_accesses_
          ? std::numeric_limits<std::uint64_t>::max()
          : routed_demand_accesses_ + generic_demand_accesses_;
  stats["demand_accesses"] = clamp_counter(total);
  stats["demand_h2d_count"] = clamp_counter(demand_h2d_count_);
  stats["demand_h2d_bytes"] = clamp_counter(demand_h2d_bytes_);
  stats["prefetch_h2d_count"] = clamp_counter(prefetch_h2d_count_);
  stats["prefetch_h2d_bytes"] = clamp_counter(prefetch_h2d_bytes_);
  stats["total_h2d_count"] = clamp_counter(
      demand_h2d_count_ >
              std::numeric_limits<std::uint64_t>::max() - prefetch_h2d_count_
          ? std::numeric_limits<std::uint64_t>::max()
          : demand_h2d_count_ + prefetch_h2d_count_);
  stats["total_h2d_bytes"] = clamp_counter(
      demand_h2d_bytes_ >
              std::numeric_limits<std::uint64_t>::max() - prefetch_h2d_bytes_
          ? std::numeric_limits<std::uint64_t>::max()
          : demand_h2d_bytes_ + prefetch_h2d_bytes_);
  stats["demand_miss_count"] = clamp_counter(demand_miss_count_);
  stats["demand_miss_bytes"] = clamp_counter(demand_miss_bytes_);
  stats["prefetch_ready_first_use_count"] =
      clamp_counter(prefetch_ready_first_use_count_);
  stats["prefetch_ready_first_use_bytes"] =
      clamp_counter(prefetch_ready_first_use_bytes_);
  stats["prefetch_late_first_use_count"] =
      clamp_counter(prefetch_late_first_use_count_);
  stats["prefetch_late_first_use_bytes"] =
      clamp_counter(prefetch_late_first_use_bytes_);
  stats["reuse_hit_count"] = clamp_counter(reuse_hit_count_);
  stats["reuse_hit_bytes"] = clamp_counter(reuse_hit_bytes_);
  stats["unused_prefetch_eviction_count"] =
      clamp_counter(unused_prefetch_eviction_count_);
  stats["unused_prefetch_eviction_bytes"] =
      clamp_counter(unused_prefetch_eviction_bytes_);
  const auto classified_accesses =
      demand_miss_count_ + prefetch_ready_first_use_count_ +
      prefetch_late_first_use_count_ + reuse_hit_count_;
  stats["routed_access_partition_ok"] =
      classified_accesses == routed_demand_accesses_ ? 1 : 0;
  std::uint64_t unused_resident_count = 0;
  std::uint64_t unused_resident_bytes = 0;
  for (const auto& gpu : gpu_states_) {
    for (const auto& entry : gpu.second.entries) {
      if (!entry.prefetched_unconsumed ||
          (entry.key.logical_expert_key & kCanonicalNodeResidencyTag) != 0) {
        continue;
      }
      if (unused_resident_count != std::numeric_limits<std::uint64_t>::max()) {
        ++unused_resident_count;
      }
      const auto bytes =
          static_cast<std::uint64_t>(std::max<std::int64_t>(entry.bytes, 0));
      unused_resident_bytes =
          bytes > std::numeric_limits<std::uint64_t>::max() -
                      unused_resident_bytes
              ? std::numeric_limits<std::uint64_t>::max()
              : unused_resident_bytes + bytes;
    }
  }
  stats["unused_prefetch_resident_count"] =
      clamp_counter(unused_resident_count);
  stats["unused_prefetch_resident_bytes"] =
      clamp_counter(unused_resident_bytes);
  stats["residency_events_buffered"] =
      static_cast<std::int64_t>(residency_events_.size());
  stats["residency_events_dropped"] = clamp_counter(residency_events_dropped_);
  return stats;
}

std::int64_t ExpertResidencyManager::ResidentBytes(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state == nullptr ? 0 : state->resident_bytes;
}

std::size_t ExpertResidencyManager::ResidentCount(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  if (state == nullptr) return 0;
  std::set<std::uint64_t> logical_experts;
  for (const auto& entry : state->entries) {
    if (entry.state == ResidencyState::ACTIVE) {
      logical_experts.insert(entry.key.logical_expert_key);
    }
  }
  return logical_experts.size();
}
