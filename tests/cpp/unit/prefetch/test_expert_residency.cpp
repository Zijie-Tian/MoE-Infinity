// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_residency_test_fixture.h"

#include <algorithm>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>

namespace {

class RefusingTransferOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override {
    observed_state = node->exec_state.load(std::memory_order_acquire);
    return false;
  }
  NodeExecState observed_state = NodeExecState::IDLE;
};

class ThrowingTransferOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override {
    observed_state = node->exec_state.load(std::memory_order_acquire);
    throw std::runtime_error("injected transfer failure");
  }
  NodeExecState observed_state = NodeExecState::IDLE;
};

class ClaimObservingTransferOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override {
    observed_state = node->exec_state.load(std::memory_order_acquire);
    node->device = node->default_host;
    return true;
  }
  NodeExecState observed_state = NodeExecState::IDLE;
};

class BlockingClaimTransferOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override {
    std::unique_lock<std::mutex> lock(mutex_);
    observed_state = node->exec_state.load(std::memory_order_acquire);
    entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
    node->device = node->default_host;
    return true;
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return entered_; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

  NodeExecState observed_state = NodeExecState::IDLE;

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_ = false;
  bool released_ = false;
};

}  // namespace

TEST_F(ExpertResidencyManagerTest, AdmissionRejectsUntilCapacityConfigured) {
  auto node = MakeNode(1, 40);
  auto rejected =
      manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.outcome, AdmissionOutcome::REJECTED);
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto admitted =
      manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_TRUE(admitted.valid);
  EXPECT_TRUE(manager->AbortAdmission(admitted));
}

TEST_F(ExpertResidencyManagerTest, CapacityAndVictimTransactionAreAtomic) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 60);
  auto first_ticket =
      manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(first_ticket));
  auto second = MakeNode(2, 60);
  auto second_ticket =
      manager->BeginAdmission(second, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::PREFETCH);
  ASSERT_EQ(second_ticket.reserved_victim, first);
  ASSERT_TRUE(manager->EvictReserved(second_ticket));
  second->device = second->default_device;
  ASSERT_TRUE(manager->CommitAdmission(second_ticket));
  EXPECT_EQ(manager->ResidentBytes(0), 60);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest, LeaseBlocksEvictionUntilReleased) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 100);
  auto ticket =
      manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  const auto lease = manager->AcquireLease(first, LeaseKind::DEMAND);
  EXPECT_FALSE(manager
                   ->BeginAdmission(MakeNode(2, 100), 0, ExpertPhase::DECODE,
                                    AdmissionMode::CACHE,
                                    AdmissionSource::DEMAND)
                   .valid);
  EXPECT_TRUE(manager->ReleaseLease(lease));
  EXPECT_FALSE(manager->ReleaseLease(lease));
  auto retry =
      manager->BeginAdmission(MakeNode(3, 100), 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_EQ(retry.reserved_victim, first);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}

TEST_F(ExpertResidencyManagerTest, DemandAndPrefetchClientsShareAccounting) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  ExpertResidencyClient demand(manager, AdmissionSource::DEMAND);
  ExpertResidencyClient prefetch(manager, AdmissionSource::PREFETCH);
  auto node = MakeNode(1, 40);
  auto admitted =
      demand.BeginAdmission(node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  node->device = node->default_device;
  ASSERT_TRUE(manager->CommitAdmission(admitted));
  auto duplicate = prefetch.BeginAdmission(node, 0, ExpertPhase::PREFILL,
                                           AdmissionMode::CACHE);
  EXPECT_EQ(duplicate.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_EQ(demand.manager().get(), prefetch.manager().get());
  EXPECT_EQ(manager->ResidentBytes(0), 40);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest,
       DenseNodeZeroCannotAliasLayerZeroExpertZero) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));

  auto dense = MakeNode(0, 40);
  auto dense_ticket = prefetch_client->BeginAdmission(
      dense, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
  ASSERT_TRUE(dense_ticket.valid);
  dense->device = dense->default_device;
  ASSERT_TRUE(manager->CommitAdmission(dense_ticket));

  const auto expert_key = RoutedExpertResidencyKey(0, 0);
  EXPECT_FALSE(manager->ActiveGeneration(0, expert_key).has_value());

  auto expert = MakeNode(1, 40);
  expert->residency_key = expert_key;
  auto expert_ticket = demand_client->BeginAdmission(
      expert, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(expert_ticket.valid);
  EXPECT_EQ(expert_ticket.outcome, AdmissionOutcome::ADMIT);
  expert->device = expert->default_device;
  ASSERT_TRUE(manager->CommitAdmission(expert_ticket));

  const auto active = manager->ActiveGeneration(0, expert_key);
  ASSERT_TRUE(active.has_value());
  EXPECT_EQ(active->logical_expert_key, expert_key);

  const auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 2);
  EXPECT_NE(entries[0].key.logical_expert_key,
            entries[1].key.logical_expert_key);
  EXPECT_TRUE(entries[0].key.logical_expert_key ==
                  CanonicalNodeResidencyKey(dense->id) ||
              entries[1].key.logical_expert_key ==
                  CanonicalNodeResidencyKey(dense->id));
}

TEST_F(ExpertResidencyManagerTest,
       SparsePrefetchAndRoutedDemandShareExpertIdentity) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto expert = MakeNode(7, 40);
  const auto expert_key = RoutedExpertResidencyKey(0, 0);
  expert->residency_key = expert_key;

  auto prefetched = prefetch_client->BeginAdmission(
      expert, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
  ASSERT_TRUE(prefetched.valid);
  expert->device = expert->default_device;
  ASSERT_TRUE(manager->CommitAdmission(prefetched));

  ResidencyVariant demand_variant{
      ResidencyVariantKey{expert_key, ExpertFormat::BF16, 0}, expert,
      expert->byte_size, expert->byte_size, 0};
  auto demand = demand_client->BeginAdmission(
      demand_variant, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  EXPECT_EQ(demand.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_EQ(manager->ResidentBytes(0), expert->byte_size);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest,
       LargeAdmissionEvictsMultipleVictimsInLruOrder) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 40);
  auto second = MakeNode(2, 40);
  auto third = MakeNode(3, 20);
  first->last_access_time = 1;
  second->last_access_time = 2;
  third->last_access_time = 3;

  for (const auto& node : {first, second, third}) {
    auto tx = demand_client->BeginAdmission(
        node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
    ASSERT_TRUE(tx.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(tx));
  }
  ASSERT_EQ(manager->ResidentBytes(0), 100);

  auto incoming = MakeNode(4, 90);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, first);
  ASSERT_TRUE(manager->EvictReserved(admission));
  auto reserved = manager->Snapshot();
  EXPECT_EQ(reserved.at("resident_bytes"), 0);
  EXPECT_EQ(reserved.at("transition_reserved_bytes"), 90);
  EXPECT_LE(reserved.at("resident_bytes") +
                reserved.at("transition_reserved_bytes") +
                reserved.at("workspace_bytes"),
            reserved.at("capacity_bytes"));
  EXPECT_EQ(transfer_ops->moved_ids,
            (std::vector<std::size_t>{1, 2, 3}));

  incoming->device = incoming->default_device;
  ASSERT_TRUE(manager->CommitAdmission(admission));
  EXPECT_EQ(manager->ResidentBytes(0), 90);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest,
       PartialMultiVictimFailureAbortsWithoutReservationUnderflow) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 40);
  auto leased_second = MakeNode(2, 40);
  auto leased_third = MakeNode(3, 20);
  first->last_access_time = 1;
  leased_second->last_access_time = 2;
  leased_third->last_access_time = 3;

  for (const auto& node : {first, leased_second, leased_third}) {
    auto tx = demand_client->BeginAdmission(
        node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
    ASSERT_TRUE(tx.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(tx));
  }
  const auto second_lease =
      manager->AcquireLease(leased_second, LeaseKind::EXECUTION);
  const auto third_lease =
      manager->AcquireLease(leased_third, LeaseKind::EXECUTION);
  ASSERT_NE(second_lease, 0);
  ASSERT_NE(third_lease, 0);

  auto incoming = MakeNode(4, 90);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, first);
  EXPECT_FALSE(manager->EvictReserved(admission));
  EXPECT_TRUE(manager->AbortAdmission(admission));

  auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("resident_bytes"), 60);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 0);
  EXPECT_EQ(snapshot.at("workspace_bytes"), 0);
  EXPECT_EQ(snapshot.at("pending_transactions"), 0);
  EXPECT_EQ(transfer_ops->moved_ids, (std::vector<std::size_t>{1}));
  EXPECT_TRUE(manager->ReleaseLease(second_lease));
  EXPECT_TRUE(manager->ReleaseLease(third_lease));
}

TEST_F(ExpertResidencyManagerTest,
       CapacityFreedBeforeEvictionReleasesUnusedVictimReservation) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto resident = MakeNode(1, 60);
  resident->last_access_time = 1;
  auto resident_ticket = demand_client->BeginAdmission(
      resident, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(resident_ticket.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(manager->CommitAdmission(resident_ticket));

  // This transaction temporarily consumes all remaining capacity.
  auto temporary = MakeNode(2, 40);
  auto temporary_ticket = demand_client->BeginAdmission(
      temporary, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(temporary_ticket.valid);

  auto incoming = MakeNode(3, 40);
  incoming->last_access_time = 2;
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, resident);

  // Freeing the other reservation makes eviction unnecessary. The victim's
  // reservation lease still must be released by EvictReserved.
  ASSERT_TRUE(manager->AbortAdmission(temporary_ticket));
  ASSERT_TRUE(manager->EvictReserved(admission));
  EXPECT_TRUE(transfer_ops->moved_ids.empty());
  auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].node, resident);
  EXPECT_EQ(entries[0].lease_count, 0);

  incoming->device = incoming->default_device;
  ASSERT_TRUE(manager->CommitAdmission(admission));
  EXPECT_EQ(manager->ResidentBytes(0), 100);

  // A subsequent admission can reserve the original LRU resident, proving
  // the unused reservation did not leave it permanently ineligible.
  auto retry = demand_client->BeginAdmission(
      MakeNode(4, 1), 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(retry.valid);
  EXPECT_EQ(retry.reserved_victim, resident);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}

TEST_F(ExpertResidencyManagerTest,
       CommitAfterPartialEvictionFailureCleansUnchargedTransaction) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto evictable = MakeNode(1, 40);
  auto leased = MakeNode(2, 60);
  evictable->last_access_time = 1;
  leased->last_access_time = 2;

  for (const auto& node : {evictable, leased}) {
    auto tx = demand_client->BeginAdmission(
        node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
    ASSERT_TRUE(tx.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(tx));
  }
  const auto execution_lease =
      manager->AcquireLease(leased, LeaseKind::EXECUTION);
  ASSERT_NE(execution_lease, 0);

  auto incoming = MakeNode(3, 90);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, evictable);
  EXPECT_FALSE(manager->EvictReserved(admission));

  // Commit must reject an uncharged transaction and fully remove its ticket;
  // no reservation bytes or manager-owned victim lease remain to abort.
  EXPECT_FALSE(manager->CommitAdmission(admission));
  const auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("resident_bytes"), 60);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 0);
  EXPECT_EQ(snapshot.at("workspace_bytes"), 0);
  EXPECT_EQ(snapshot.at("pending_transactions"), 0);
  EXPECT_FALSE(manager->AbortAdmission(admission));

  EXPECT_TRUE(manager->ReleaseLease(execution_lease));
  auto retry = demand_client->BeginAdmission(
      MakeNode(4, 40), 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(retry.valid);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}

TEST_F(ExpertResidencyManagerTest, ManagedLfuDivergesFromLruForHotOldEntry) {
  const auto selected_victim = [this](ExpertEvictionPolicy policy) {
    auto local_ops = std::make_shared<RecordingTransferOps>();
    auto local_manager = std::make_shared<ExpertResidencyManager>(local_ops);
    EXPECT_TRUE(local_manager->ConfigureEvictionPolicy(policy));
    EXPECT_TRUE(local_manager->ConfigureCapacity(0, 2));

    auto hot_old = MakeNode(101, 1);
    auto cold_recent = MakeNode(102, 1);
    auto incoming = MakeNode(103, 1);
    hot_old->residency_key = RoutedExpertResidencyKey(1, 1);
    cold_recent->residency_key = RoutedExpertResidencyKey(1, 2);
    incoming->residency_key = RoutedExpertResidencyKey(1, 3);
    hot_old->last_access_time = 1;
    cold_recent->last_access_time = 2;

    for (const auto& node : {hot_old, cold_recent}) {
      auto ticket = local_manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                                  AdmissionMode::CACHE,
                                                  AdmissionSource::DEMAND);
      EXPECT_TRUE(ticket.valid);
      node->device = node->default_device;
      EXPECT_TRUE(local_manager->CommitAdmission(ticket));
      local_manager->RecordAccess(node, ExpertPhase::DECODE, false);
    }
    local_manager->RecordAccess(hot_old, ExpertPhase::DECODE, true);
    local_manager->RecordAccess(hot_old, ExpertPhase::DECODE, true);

    auto ticket = local_manager->BeginAdmission(
        incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE,
        AdmissionSource::DEMAND);
    EXPECT_TRUE(ticket.valid);
    const auto victim_id =
        ticket.reserved_victim == nullptr ? 0 : ticket.reserved_victim->id;
    EXPECT_TRUE(local_manager->AbortAdmission(ticket));
    return victim_id;
  };

  EXPECT_EQ(selected_victim(ExpertEvictionPolicy::LRU), 101);
  EXPECT_EQ(selected_victim(ExpertEvictionPolicy::LFU), 102);
}

TEST_F(ExpertResidencyManagerTest,
       ManagedLfuTieBreaksByOldestDemandThenLogicalKey) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 2));
  auto first = MakeNode(1, 1);
  auto second = MakeNode(2, 1);
  first->residency_key = RoutedExpertResidencyKey(4, 1);
  second->residency_key = RoutedExpertResidencyKey(4, 2);
  for (const auto& node : {first, second}) {
    auto ticket = demand_client->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                                AdmissionMode::CACHE);
    ASSERT_TRUE(ticket.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(ticket));
    manager->RecordAccess(node, ExpertPhase::DECODE, false);
  }

  auto incoming = MakeNode(3, 1);
  incoming->residency_key = RoutedExpertResidencyKey(4, 3);
  auto ticket = demand_client->BeginAdmission(incoming, 0, ExpertPhase::DECODE,
                                              AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  EXPECT_EQ(ticket.reserved_victim, first);
  EXPECT_TRUE(manager->AbortAdmission(ticket));
}

TEST_F(ExpertResidencyManagerTest,
       ManagedLfuUntouchedPrefetchTieUsesLogicalKey) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 2));
  auto larger_key = MakeNode(1, 1);
  auto smaller_key = MakeNode(2, 1);
  larger_key->residency_key = RoutedExpertResidencyKey(10, 2);
  smaller_key->residency_key = RoutedExpertResidencyKey(10, 1);
  // Admit in reverse key order. Both remain frequency/sequence zero because
  // prefetch is not a demand touch.
  for (const auto& node : {larger_key, smaller_key}) {
    auto ticket = prefetch_client->BeginAdmission(node, 0, ExpertPhase::PREFILL,
                                                  AdmissionMode::CACHE);
    ASSERT_TRUE(ticket.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(ticket));
  }

  auto incoming = MakeNode(3, 1);
  incoming->residency_key = RoutedExpertResidencyKey(10, 3);
  auto ticket = demand_client->BeginAdmission(incoming, 0, ExpertPhase::DECODE,
                                              AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  EXPECT_EQ(ticket.reserved_victim, smaller_key);
  EXPECT_TRUE(manager->AbortAdmission(ticket));
}

TEST_F(ExpertResidencyManagerTest, ManagedLruTimestampTieUsesLogicalKey) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 2));
  auto larger_key = MakeNode(1, 1);
  auto smaller_key = MakeNode(2, 1);
  larger_key->residency_key = RoutedExpertResidencyKey(12, 2);
  smaller_key->residency_key = RoutedExpertResidencyKey(12, 1);
  larger_key->last_access_time = 7;
  smaller_key->last_access_time = 7;
  for (const auto& node : {larger_key, smaller_key}) {
    auto ticket = demand_client->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                                AdmissionMode::CACHE);
    ASSERT_TRUE(ticket.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(ticket));
    manager->RecordAccess(node, ExpertPhase::DECODE, false);
  }
  auto incoming = MakeNode(3, 1);
  incoming->residency_key = RoutedExpertResidencyKey(12, 3);
  auto ticket = demand_client->BeginAdmission(incoming, 0, ExpertPhase::DECODE,
                                              AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  EXPECT_EQ(ticket.reserved_victim, smaller_key);
  EXPECT_TRUE(manager->AbortAdmission(ticket));
}

TEST_F(ExpertResidencyManagerTest, ManagedLfuResetsAfterTrueEviction) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto first = MakeNode(1, 1);
  auto second = MakeNode(2, 1);
  first->residency_key = RoutedExpertResidencyKey(5, 1);
  second->residency_key = RoutedExpertResidencyKey(5, 2);

  auto first_ticket = demand_client->BeginAdmission(
      first, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(first_ticket.valid);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(first_ticket));
  for (int i = 0; i < 3; ++i) {
    manager->RecordAccess(first, ExpertPhase::DECODE, i != 0);
  }

  auto second_ticket = demand_client->BeginAdmission(
      second, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(second_ticket.valid);
  ASSERT_TRUE(manager->EvictReserved(second_ticket));
  second->device = second->default_device;
  ASSERT_TRUE(manager->CommitAdmission(second_ticket));
  manager->RecordAccess(second, ExpertPhase::DECODE, false);

  auto readmit = demand_client->BeginAdmission(first, 0, ExpertPhase::DECODE,
                                               AdmissionMode::CACHE);
  ASSERT_TRUE(readmit.valid);
  ASSERT_TRUE(manager->EvictReserved(readmit));
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(readmit));
  manager->RecordAccess(first, ExpertPhase::DECODE, false);

  const auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].key.logical_expert_key, *first->residency_key);
  EXPECT_EQ(entries[0].demand_frequency, 1);
  EXPECT_EQ(manager->Snapshot().at("routed_demand_accesses"), 5);
}

TEST_F(ExpertResidencyManagerTest, PrefetchAdmissionDoesNotInflateLfu) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto node = MakeNode(1, 1);
  node->residency_key = RoutedExpertResidencyKey(6, 1);
  auto ticket = prefetch_client->BeginAdmission(node, 0, ExpertPhase::PREFILL,
                                                AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  node->device = node->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  for (int i = 0; i < 3; ++i) {
    auto duplicate = prefetch_client->BeginAdmission(
        node, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
    EXPECT_EQ(duplicate.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  }

  const auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].demand_frequency, 0);
  EXPECT_EQ(entries[0].last_demand_sequence, 0);
  EXPECT_EQ(manager->Snapshot().at("routed_demand_accesses"), 0);
}

TEST_F(ExpertResidencyManagerTest, AdmissionRetriesAndAbortDoNotInflateLfu) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto resident = MakeNode(1, 1);
  resident->residency_key = RoutedExpertResidencyKey(7, 1);
  auto ticket = demand_client->BeginAdmission(resident, 0, ExpertPhase::DECODE,
                                              AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  manager->RecordAccess(resident, ExpertPhase::DECODE, false);
  const auto lease = manager->AcquireLease(resident, LeaseKind::EXECUTION);
  ASSERT_NE(lease, 0);

  for (int i = 0; i < 3; ++i) {
    auto rejected = demand_client->BeginAdmission(
        MakeNode(10 + i, 1), 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
    EXPECT_FALSE(rejected.valid);
  }
  ASSERT_TRUE(manager->ReleaseLease(lease));
  const auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].demand_frequency, 1);
  EXPECT_EQ(manager->Snapshot().at("routed_demand_accesses"), 1);
}

TEST_F(ExpertResidencyManagerTest, ManagedLfuSkipsLeasedLowFrequencyEntry) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 2));
  auto low = MakeNode(1, 1);
  auto high = MakeNode(2, 1);
  low->residency_key = RoutedExpertResidencyKey(8, 1);
  high->residency_key = RoutedExpertResidencyKey(8, 2);
  for (const auto& node : {low, high}) {
    auto ticket = demand_client->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                                AdmissionMode::CACHE);
    ASSERT_TRUE(ticket.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(ticket));
    manager->RecordAccess(node, ExpertPhase::DECODE, false);
  }
  manager->RecordAccess(high, ExpertPhase::DECODE, true);
  const auto lease = manager->AcquireLease(low, LeaseKind::EXECUTION);
  ASSERT_NE(lease, 0);

  auto incoming = MakeNode(3, 1);
  incoming->residency_key = RoutedExpertResidencyKey(8, 3);
  auto ticket = demand_client->BeginAdmission(incoming, 0, ExpertPhase::DECODE,
                                              AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  EXPECT_EQ(ticket.reserved_victim, high);
  EXPECT_TRUE(manager->AbortAdmission(ticket));
  ASSERT_TRUE(manager->ReleaseLease(lease));

  manager->ReplaceProtectedCandidates(NodePtrList{low});
  auto retry = demand_client->BeginAdmission(incoming, 0, ExpertPhase::DECODE,
                                             AdmissionMode::CACHE);
  ASSERT_TRUE(retry.valid);
  EXPECT_EQ(retry.reserved_victim, high);
  EXPECT_TRUE(manager->AbortAdmission(retry));

  manager->ReplaceProtectedCandidates(NodePtrList{});
  auto unprotected = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(unprotected.valid);
  EXPECT_EQ(unprotected.reserved_victim, low);
  EXPECT_TRUE(manager->AbortAdmission(unprotected));
}

TEST_F(ExpertResidencyManagerTest,
       VariantTransitionCarriesLfuStateForLogicalExpert) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 4));
  auto current = MakeVariant(9, 1, ExpertFormat::BF16, 1, 1, 0);
  auto replacement = MakeVariant(9, 1, ExpertFormat::GPTQ, 2, 1, 0);
  current.node->residency_key = current.key.logical_expert_key;
  replacement.node->residency_key = replacement.key.logical_expert_key;
  CommitResident(current, ExpertPhase::DECODE);
  manager->RecordAccess(current.node, ExpertPhase::DECODE, false);
  manager->RecordAccess(current.node, ExpertPhase::DECODE, true);

  auto transition = demand_client->BeginTransition(current.key, replacement, 0,
                                                   ExpertPhase::DECODE);
  ASSERT_TRUE(transition.valid);
  replacement.node->device = replacement.node->default_device;
  ASSERT_TRUE(manager->CommitTransaction(transition));

  auto entries = manager->ResidentGenerations(0);
  auto active = std::find_if(entries.begin(), entries.end(), [](const auto& e) {
    return e.state == ResidencyState::ACTIVE;
  });
  ASSERT_NE(active, entries.end());
  EXPECT_TRUE(active->key == replacement.key);
  EXPECT_EQ(active->demand_frequency, 2);
  const auto carried_sequence = active->last_demand_sequence;

  manager->RecordAccess(replacement.node, ExpertPhase::DECODE, true);
  entries = manager->ResidentGenerations(0);
  active = std::find_if(entries.begin(), entries.end(), [](const auto& e) {
    return e.state == ResidencyState::ACTIVE;
  });
  ASSERT_NE(active, entries.end());
  EXPECT_EQ(active->demand_frequency, 3);
  EXPECT_GT(active->last_demand_sequence, carried_sequence);
}

TEST_F(ExpertResidencyManagerTest,
       GenericManagedGroupUsesTaggedIdentityAndCountsOnce) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto generic = MakeNode(0, 1);
  auto ticket = prefetch_client->BeginAdmission(
      generic, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  generic->device = generic->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  manager->RecordAccess(generic, ExpertPhase::MIXED, true);

  const auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].key.logical_expert_key,
            CanonicalNodeResidencyKey(generic->id));
  EXPECT_EQ(entries[0].demand_frequency, 1);
  const auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("generic_demand_accesses"), 1);
  EXPECT_EQ(snapshot.at("routed_demand_accesses"), 0);
  EXPECT_EQ(snapshot.at("eviction_policy"),
            static_cast<std::int64_t>(ExpertEvictionPolicy::LFU));
}

TEST_F(ExpertResidencyManagerTest,
       RecordAccessUpdatesMatchingGpuForReplicatedLogicalKey) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  ASSERT_TRUE(manager->ConfigureCapacity(1, 1));
  const auto logical_key = RoutedExpertResidencyKey(11, 1);
  auto gpu0_node = MakeNode(1, 1);
  auto gpu1_node = MakeNode(2, 1);
  gpu0_node->residency_key = logical_key;
  gpu1_node->residency_key = logical_key;
  gpu1_node->default_device = torch::Device(torch::kCUDA, 1);

  for (const auto& item :
       std::vector<std::pair<NodePtr, int>>{{gpu0_node, 0}, {gpu1_node, 1}}) {
    auto ticket =
        manager->BeginAdmission(item.first, item.second, ExpertPhase::DECODE,
                                AdmissionMode::CACHE, AdmissionSource::DEMAND);
    ASSERT_TRUE(ticket.valid);
    item.first->device = item.first->default_device;
    ASSERT_TRUE(manager->CommitAdmission(ticket));
  }

  manager->RecordAccess(gpu1_node, ExpertPhase::DECODE, false);
  const auto gpu0_entries = manager->ResidentGenerations(0);
  const auto gpu1_entries = manager->ResidentGenerations(1);
  ASSERT_EQ(gpu0_entries.size(), 1);
  ASSERT_EQ(gpu1_entries.size(), 1);
  EXPECT_EQ(gpu0_entries[0].demand_frequency, 0);
  EXPECT_EQ(gpu1_entries[0].demand_frequency, 1);
}

TEST_F(ExpertResidencyManagerTest, BusyNodeFlagsBlockManagedEviction) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto resident = MakeNode(1, 1);
  resident->residency_key = RoutedExpertResidencyKey(13, 1);
  auto ticket = demand_client->BeginAdmission(resident, 0, ExpertPhase::DECODE,
                                              AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  manager->RecordAccess(resident, ExpertPhase::DECODE, false);

  const auto expect_blocked = [&] {
    auto rejected = demand_client->BeginAdmission(
        MakeNode(2, 1), 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
    EXPECT_FALSE(rejected.valid);
    EXPECT_EQ(rejected.outcome, AdmissionOutcome::REJECTED);
  };

  resident->is_prefetching.store(true, std::memory_order_release);
  expect_blocked();
  resident->is_prefetching.store(false, std::memory_order_release);

  resident->exec_state.store(NodeExecState::FETCHING,
                             std::memory_order_release);
  expect_blocked();
  resident->exec_state.store(NodeExecState::IDLE, std::memory_order_release);

  resident->pending_dispatches.store(1, std::memory_order_release);
  expect_blocked();
  resident->pending_dispatches.store(0, std::memory_order_release);

  auto eligible = demand_client->BeginAdmission(
      MakeNode(3, 1), 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(eligible.valid);
  EXPECT_EQ(eligible.reserved_victim, resident);
  EXPECT_TRUE(manager->AbortAdmission(eligible));
}

TEST_F(ExpertResidencyManagerTest,
       HostStaleActiveEntryCanBeReclaimedFromManagerAccounting) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto stale = MakeNode(1, 1);
  auto first = demand_client->BeginAdmission(stale, 0, ExpertPhase::DECODE,
                                             AdmissionMode::CACHE);
  ASSERT_TRUE(first.valid);
  stale->device = stale->default_device;
  ASSERT_TRUE(manager->CommitAdmission(first));
  ASSERT_EQ(manager->ResidentBytes(0), 1);

  // Simulate the existing external resize path, which moves topology nodes to
  // host without first reconciling the residency manager.
  stale->device = stale->default_host;
  auto incoming = MakeNode(2, 1);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, stale);
  ASSERT_TRUE(manager->EvictReserved(admission));
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  EXPECT_EQ(manager->ResidentCount(0), 0);
  incoming->device = incoming->default_device;
  EXPECT_TRUE(manager->CommitAdmission(admission));
}

TEST_F(ExpertResidencyManagerTest,
       FetchingGenericLfuEntryIsSkippedThenEligibleWhenIdle) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 2));
  auto generic = MakeNode(1, 1);
  auto routed = MakeNode(2, 1);
  routed->residency_key = RoutedExpertResidencyKey(14, 1);

  auto generic_ticket = prefetch_client->BeginAdmission(
      generic, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
  ASSERT_TRUE(generic_ticket.valid);
  generic->device = generic->default_device;
  ASSERT_TRUE(manager->CommitAdmission(generic_ticket));

  auto routed_ticket = demand_client->BeginAdmission(
      routed, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(routed_ticket.valid);
  routed->device = routed->default_device;
  ASSERT_TRUE(manager->CommitAdmission(routed_ticket));
  manager->RecordAccess(routed, ExpertPhase::DECODE, false);
  manager->RecordAccess(routed, ExpertPhase::DECODE, true);

  generic->exec_state.store(NodeExecState::FETCHING, std::memory_order_release);
  auto incoming = MakeNode(3, 1);
  incoming->residency_key = RoutedExpertResidencyKey(14, 2);
  auto while_fetching = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(while_fetching.valid);
  EXPECT_EQ(while_fetching.reserved_victim, routed);
  EXPECT_TRUE(manager->AbortAdmission(while_fetching));

  generic->exec_state.store(NodeExecState::IDLE, std::memory_order_release);
  auto when_idle = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(when_idle.valid);
  EXPECT_EQ(when_idle.reserved_victim, generic);
  EXPECT_TRUE(manager->AbortAdmission(when_idle));
}

TEST_F(ExpertResidencyManagerTest,
       ConsumerClaimAfterReservationPreventsMoveToHost) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto resident = MakeNode(1, 1);
  auto first = demand_client->BeginAdmission(resident, 0, ExpertPhase::DECODE,
                                             AdmissionMode::CACHE);
  ASSERT_TRUE(first.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(manager->CommitAdmission(first));

  auto incoming = MakeNode(2, 1);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, resident);
  resident->exec_state.store(NodeExecState::FETCHING,
                             std::memory_order_release);

  expert_residency_detail::PendingAdmissionGuard guard(manager);
  guard.Arm(&admission);
  EXPECT_FALSE(manager->EvictReserved(admission));
  guard.Abort();
  EXPECT_TRUE(transfer_ops->moved_ids.empty());
  EXPECT_TRUE(resident->device.is_cuda());
  EXPECT_EQ(manager->ResidentBytes(0), 1);
  EXPECT_EQ(manager->Snapshot().at("pending_transactions"), 0);
  EXPECT_EQ(resident->exec_state.load(std::memory_order_acquire),
            NodeExecState::FETCHING);
  resident->exec_state.store(NodeExecState::IDLE, std::memory_order_release);

  auto retry = demand_client->BeginAdmission(incoming, 0, ExpertPhase::DECODE,
                                             AdmissionMode::CACHE);
  ASSERT_TRUE(retry.valid);
  EXPECT_EQ(retry.reserved_victim, resident);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}

TEST_F(ExpertResidencyManagerTest, RefusedMoveRestoresIdleClaim) {
  auto refusing = std::make_shared<RefusingTransferOps>();
  auto local_manager = std::make_shared<ExpertResidencyManager>(refusing);
  ExpertResidencyClient client(local_manager, AdmissionSource::DEMAND);
  ASSERT_TRUE(local_manager->ConfigureCapacity(0, 1));
  auto resident = MakeNode(1, 1);
  auto first = client.BeginAdmission(resident, 0, ExpertPhase::DECODE,
                                     AdmissionMode::CACHE);
  ASSERT_TRUE(first.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(local_manager->CommitAdmission(first));
  auto admission = client.BeginAdmission(MakeNode(2, 1), 0, ExpertPhase::DECODE,
                                         AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);

  EXPECT_FALSE(local_manager->EvictReserved(admission));
  EXPECT_EQ(refusing->observed_state, NodeExecState::RESIZE_RESERVED);
  EXPECT_EQ(resident->exec_state.load(std::memory_order_acquire),
            NodeExecState::IDLE);
  EXPECT_TRUE(resident->device.is_cuda());
  EXPECT_EQ(local_manager->ResidentBytes(0), 1);
  EXPECT_TRUE(local_manager->AbortAdmission(admission));
}

TEST_F(ExpertResidencyManagerTest, ThrowingMoveRestoresIdleClaim) {
  auto throwing = std::make_shared<ThrowingTransferOps>();
  auto local_manager = std::make_shared<ExpertResidencyManager>(throwing);
  ExpertResidencyClient client(local_manager, AdmissionSource::DEMAND);
  ASSERT_TRUE(local_manager->ConfigureCapacity(0, 1));
  auto resident = MakeNode(1, 1);
  auto first = client.BeginAdmission(resident, 0, ExpertPhase::DECODE,
                                     AdmissionMode::CACHE);
  ASSERT_TRUE(first.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(local_manager->CommitAdmission(first));
  auto admission = client.BeginAdmission(MakeNode(2, 1), 0, ExpertPhase::DECODE,
                                         AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);

  EXPECT_THROW(local_manager->EvictReserved(admission), std::runtime_error);
  EXPECT_EQ(throwing->observed_state, NodeExecState::RESIZE_RESERVED);
  EXPECT_EQ(resident->exec_state.load(std::memory_order_acquire),
            NodeExecState::IDLE);
  EXPECT_TRUE(resident->device.is_cuda());
  EXPECT_EQ(local_manager->ResidentBytes(0), 1);
  EXPECT_TRUE(local_manager->AbortAdmission(admission));
}

TEST_F(ExpertResidencyManagerTest,
       SuccessfulMoveHoldsClaimThroughAccountingAndErase) {
  auto observing = std::make_shared<ClaimObservingTransferOps>();
  auto local_manager = std::make_shared<ExpertResidencyManager>(observing);
  ExpertResidencyClient client(local_manager, AdmissionSource::DEMAND);
  ASSERT_TRUE(local_manager->ConfigureCapacity(0, 1));
  auto resident = MakeNode(1, 1);
  auto first = client.BeginAdmission(resident, 0, ExpertPhase::DECODE,
                                     AdmissionMode::CACHE);
  ASSERT_TRUE(first.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(local_manager->CommitAdmission(first));
  auto incoming = MakeNode(2, 1);
  auto admission = client.BeginAdmission(incoming, 0, ExpertPhase::DECODE,
                                         AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);

  ASSERT_TRUE(local_manager->EvictReserved(admission));
  EXPECT_EQ(observing->observed_state, NodeExecState::RESIZE_RESERVED);
  EXPECT_EQ(resident->exec_state.load(std::memory_order_acquire),
            NodeExecState::IDLE);
  EXPECT_TRUE(resident->device.is_cpu());
  EXPECT_EQ(local_manager->ResidentBytes(0), 0);
  EXPECT_EQ(local_manager->ResidentCount(0), 0);
  incoming->device = incoming->default_device;
  EXPECT_TRUE(local_manager->CommitAdmission(admission));
}

TEST_F(ExpertResidencyManagerTest,
       DemandClaimWaitsForEvictionClaimThenObservesHost) {
  auto blocking = std::make_shared<BlockingClaimTransferOps>();
  auto local_manager = std::make_shared<ExpertResidencyManager>(blocking);
  ExpertResidencyClient client(local_manager, AdmissionSource::DEMAND);
  ASSERT_TRUE(local_manager->ConfigureCapacity(0, 1));
  auto resident = MakeNode(1, 1);
  auto first = client.BeginAdmission(resident, 0, ExpertPhase::DECODE,
                                     AdmissionMode::CACHE);
  ASSERT_TRUE(first.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(local_manager->CommitAdmission(first));
  auto admission = client.BeginAdmission(MakeNode(2, 1), 0, ExpertPhase::DECODE,
                                         AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);

  auto eviction = std::async(std::launch::async, [&] {
    return local_manager->EvictReserved(admission);
  });
  blocking->WaitUntilEntered();
  EXPECT_EQ(blocking->observed_state, NodeExecState::RESIZE_RESERVED);

  NodeExecState competing = NodeExecState::IDLE;
  auto demand = std::async(std::launch::async, [&] {
    return expert_residency_detail::ClaimNodeForDemandExecution(resident,
                                                                &competing);
  });
  EXPECT_EQ(demand.wait_for(std::chrono::milliseconds(10)),
            std::future_status::timeout);

  blocking->Release();
  ASSERT_TRUE(eviction.get());
  ASSERT_TRUE(demand.get());
  EXPECT_EQ(competing, NodeExecState::IDLE);
  EXPECT_TRUE(resident->device.is_cpu());
  EXPECT_EQ(resident->exec_state.load(std::memory_order_acquire),
            NodeExecState::FETCHING);
  resident->exec_state.store(NodeExecState::IDLE, std::memory_order_release);
  EXPECT_TRUE(local_manager->AbortAdmission(admission));
  const auto snapshot = local_manager->Snapshot();
  EXPECT_EQ(snapshot.at("pending_transactions"), 0);
  EXPECT_EQ(snapshot.at("active_leases"), 0);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 0);
}

TEST_F(ExpertResidencyManagerTest,
       PendingAdmissionGuardAbortsDuringExceptionUnwind) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto ticket = demand_client->BeginAdmission(
      MakeNode(1, 1), 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  EXPECT_THROW(
      {
        expert_residency_detail::PendingAdmissionGuard guard(manager);
        guard.Arm(&ticket);
        throw std::runtime_error("injected guarded failure");
      },
      std::runtime_error);
  const auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("pending_transactions"), 0);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 0);
  EXPECT_EQ(snapshot.at("workspace_bytes"), 0);
}

TEST_F(ExpertResidencyManagerTest,
       PendingAdmissionGuardDisarmPreservesCommittedEntry) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto node = MakeNode(1, 1);
  auto ticket = demand_client->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                              AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  {
    expert_residency_detail::PendingAdmissionGuard guard(manager);
    guard.Arm(&ticket);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(ticket));
    guard.Disarm();
  }
  const auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("pending_transactions"), 0);
  EXPECT_EQ(snapshot.at("resident_bytes"), 1);
  EXPECT_EQ(snapshot.at("resident_count"), 1);
}

TEST_F(ExpertResidencyManagerTest,
       PendingAdmissionGuardAbortsVariantTransition) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 4));
  auto current = MakeVariant(15, 1, ExpertFormat::BF16, 1, 1, 0);
  auto replacement = MakeVariant(15, 1, ExpertFormat::GPTQ, 2, 1, 1);
  CommitResident(current, ExpertPhase::DECODE);
  auto transition = demand_client->BeginTransition(current.key, replacement, 0,
                                                   ExpertPhase::DECODE);
  ASSERT_TRUE(transition.valid);
  EXPECT_THROW(
      {
        expert_residency_detail::PendingAdmissionGuard guard(manager);
        guard.Arm(&transition);
        throw std::runtime_error("injected transition failure");
      },
      std::runtime_error);

  const auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("pending_transactions"), 0);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 0);
  EXPECT_EQ(snapshot.at("workspace_bytes"), 0);
  const auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].state, ResidencyState::ACTIVE);
  EXPECT_EQ(entries[0].lease_count, 0);
}

TEST_F(ExpertResidencyManagerTest,
       RoutedPendingAdmissionHasSingleOwnerAndCanRetry) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 2));
  auto node = MakeNode(1, 1);
  node->residency_key = RoutedExpertResidencyKey(16, 1);
  auto prefetch = prefetch_client->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                                  AdmissionMode::CACHE);
  ASSERT_TRUE(prefetch.valid);
  auto duplicate = demand_client->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                                 AdmissionMode::CACHE);
  EXPECT_FALSE(duplicate.valid);
  EXPECT_EQ(duplicate.outcome, AdmissionOutcome::REJECTED);
  EXPECT_TRUE(manager->AbortAdmission(prefetch));
  auto retry = demand_client->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                             AdmissionMode::CACHE);
  EXPECT_TRUE(retry.valid);
  EXPECT_TRUE(manager->AbortAdmission(retry));

  auto demand = demand_client->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                              AdmissionMode::CACHE);
  ASSERT_TRUE(demand.valid);
  auto duplicate_prefetch = prefetch_client->BeginAdmission(
      node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  EXPECT_FALSE(duplicate_prefetch.valid);
  EXPECT_TRUE(manager->AbortAdmission(demand));
  EXPECT_EQ(manager->Snapshot().at("pending_transactions"), 0);
}

TEST_F(ExpertResidencyManagerTest,
       RoutedDemandProvenancePartitionsMissReadyLateAndReuse) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 5));

  auto ready = MakeNode(1, 1);
  ready->residency_key = RoutedExpertResidencyKey(17, 1);
  auto ready_ticket = prefetch_client->BeginAdmission(
      ready, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(ready_ticket.valid);
  ready->device = ready->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ready_ticket));
  manager->RecordPrefetchH2D(ready, ready->byte_size);
  manager->RecordAccess(ready, ExpertPhase::DECODE, true, false, 100);
  manager->RecordAccess(ready, ExpertPhase::DECODE, true, false, 200);

  auto late = MakeNode(2, 1);
  late->residency_key = RoutedExpertResidencyKey(17, 2);
  auto late_ticket = prefetch_client->BeginAdmission(
      late, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(late_ticket.valid);
  late->device = late->default_device;
  ASSERT_TRUE(manager->CommitAdmission(late_ticket));
  manager->RecordPrefetchH2D(late, late->byte_size);
  manager->RecordAccess(late, ExpertPhase::DECODE, true, true, 300);

  auto completed_after_arrival = MakeNode(4, 1);
  completed_after_arrival->residency_key = RoutedExpertResidencyKey(17, 4);
  auto gap_ticket = prefetch_client->BeginAdmission(
      completed_after_arrival, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(gap_ticket.valid);
  completed_after_arrival->device = completed_after_arrival->default_device;
  ASSERT_TRUE(manager->CommitAdmission(gap_ticket));
  manager->RecordPrefetchH2D(completed_after_arrival,
                             completed_after_arrival->byte_size);
  manager->RecordAccess(completed_after_arrival, ExpertPhase::DECODE, false,
                        false, 350);

  auto missed = MakeNode(3, 1);
  missed->residency_key = RoutedExpertResidencyKey(17, 3);
  auto miss_ticket = demand_client->BeginAdmission(
      missed, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(miss_ticket.valid);
  missed->device = missed->default_device;
  ASSERT_TRUE(manager->CommitAdmission(miss_ticket));
  manager->RecordDemandH2D(missed, missed->byte_size);
  manager->RecordAccess(missed, ExpertPhase::DECODE, false, false, 400);

  auto demand_owned = MakeNode(5, 1);
  demand_owned->residency_key = RoutedExpertResidencyKey(17, 5);
  auto demand_owned_ticket = demand_client->BeginAdmission(
      demand_owned, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(demand_owned_ticket.valid);
  demand_owned->device = demand_owned->default_device;
  ASSERT_TRUE(manager->CommitAdmission(demand_owned_ticket));
  manager->RecordDemandH2D(demand_owned, demand_owned->byte_size);
  auto no_op_prefetch = prefetch_client->BeginAdmission(
      demand_owned, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  EXPECT_EQ(no_op_prefetch.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  manager->RecordAccess(demand_owned, ExpertPhase::DECODE, false, true, 450);

  const auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("routed_demand_accesses"), 6);
  EXPECT_EQ(snapshot.at("prefetch_ready_first_use_count"), 1);
  EXPECT_EQ(snapshot.at("prefetch_late_first_use_count"), 2);
  EXPECT_EQ(snapshot.at("reuse_hit_count"), 1);
  EXPECT_EQ(snapshot.at("demand_miss_count"), 2);
  EXPECT_EQ(snapshot.at("prefetch_h2d_count"), 3);
  EXPECT_EQ(snapshot.at("demand_h2d_count"), 2);
  EXPECT_EQ(snapshot.at("unused_prefetch_resident_count"), 0);

  const auto events = manager->DrainEvents();
  ASSERT_FALSE(events.empty());
  for (std::size_t index = 1; index < events.size(); ++index) {
    EXPECT_EQ(events[index].sequence, events[index - 1].sequence + 1);
  }
  EXPECT_EQ(manager->Snapshot().at("residency_events_buffered"), 0);
  EXPECT_EQ(manager->Snapshot().at("residency_events_dropped"), 0);
}

TEST_F(ExpertResidencyManagerTest,
       UnconsumedPrefetchEvictionIsCountedAndTraced) {
  ASSERT_TRUE(manager->ConfigureEvictionPolicy(ExpertEvictionPolicy::LFU));
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1));
  auto prefetched = MakeNode(1, 1);
  prefetched->residency_key = RoutedExpertResidencyKey(18, 1);
  auto ticket = prefetch_client->BeginAdmission(
      prefetched, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(ticket.valid);
  prefetched->device = prefetched->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  manager->RecordPrefetchH2D(prefetched, prefetched->byte_size);

  auto incoming = MakeNode(2, 1);
  incoming->residency_key = RoutedExpertResidencyKey(18, 2);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_TRUE(manager->EvictReserved(admission));
  EXPECT_TRUE(manager->AbortAdmission(admission));
  const auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("unused_prefetch_eviction_count"), 1);
  EXPECT_EQ(snapshot.at("unused_prefetch_eviction_bytes"), 1);
  EXPECT_EQ(snapshot.at("unused_prefetch_resident_count"), 0);
  const auto events = manager->DrainEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const auto& event) {
    return event.kind == "unused_prefetch_evict";
  }));
}

TEST_F(ExpertResidencyManagerTest,
       CanonicalActivePrefetchNoOpCheckIsExactAndStateAware) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 4));
  auto canonical = MakeNode(1, 1);
  canonical->residency_key = RoutedExpertResidencyKey(19, 1);
  auto canonical_ticket = prefetch_client->BeginAdmission(
      canonical, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(canonical_ticket.valid);
  canonical->device = canonical->default_device;
  ASSERT_TRUE(manager->CommitAdmission(canonical_ticket));
  EXPECT_TRUE(expert_residency_detail::HasActiveCanonicalVariant(manager, 0,
                                                                 canonical));

  auto noncanonical = MakeVariant(19, 2, ExpertFormat::GPTQ, 7, 1, 0);
  noncanonical.node->residency_key = noncanonical.key.logical_expert_key;
  CommitResident(noncanonical, ExpertPhase::DECODE);
  EXPECT_FALSE(expert_residency_detail::HasActiveCanonicalVariant(
      manager, 0, noncanonical.node));

  ASSERT_TRUE(manager->RequestRetirement(
      ResidencyVariantKey{*canonical->residency_key, ExpertFormat::BF16, 0}));
  EXPECT_FALSE(expert_residency_detail::HasActiveCanonicalVariant(manager, 0,
                                                                  canonical));
}
